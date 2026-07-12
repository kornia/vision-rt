//! A tiny live-view server: H.264-over-WebSocket (WebCodecs) + MJPEG fallback.
//!
//! Endpoints:
//! - `/` — index page: a `<canvas>` client that decodes an **H.264** WebSocket stream
//!   with the browser's WebCodecs `VideoDecoder` and plays it through a small **jitter
//!   buffer** (paced render + drop-stale), so uneven network arrival doesn't look
//!   choppy. H.264 is inter-frame compressed → ~10–20× less bandwidth than MJPEG.
//! - `/ws` — one WebSocket carrying both views. Each binary message is `[kind, tag] +
//!   payload`: `kind` ∈ `C`(avcC config) / `K`(keyframe) / `D`(delta), `tag` ∈ `M`/`B`.
//! - `/main`, `/bev` — plain `multipart/x-mixed-replace` MJPEG fallback (only live if
//!   the producer also calls [`MjpegServer::publish`]).
//!
//! The producer calls [`MjpegServer::publish_h264_frame`] / `publish_h264_config` with
//! encoded access units; each viewer is pushed frames in order the instant they land,
//! and a viewer that falls a full buffer behind is dropped (its browser reconnects and
//! re-syncs on the next keyframe).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Index page: a WebSocket + WebCodecs canvas client with a jitter buffer. Two canvases
/// (main over BEV); each stream has its own `VideoDecoder` configured from the avcC
/// `description`, and decoded frames are drawn on a fixed ~15 fps tick once `MINBUF` are
/// buffered, dropping past `MAXBUF` so latency stays bounded. Auto-reconnects and
/// re-syncs on the next keyframe if the socket drops.
const INDEX_HTML: &str = "<!doctype html><html><head><meta name=viewport \
content='width=device-width,initial-scale=1'><style>body{margin:0;background:#111}\
canvas{display:block;width:100%;height:auto}#e{color:#f66;font:13px monospace;padding:6px}\
</style></head><body><canvas id=m></canvas><canvas id=b></canvas><div id=e></div><script>\
const MAXBUF=30,MINBUF=6;\
function mk(id){const c=document.getElementById(id);return{c,x:c.getContext('2d'),q:[],dec:null,cfg:null,started:false,waitkey:true};}\
const S={M:mk('m'),B:mk('b')};\
function codecStr(a){return 'avc1.'+[a[1],a[2],a[3]].map(b=>b.toString(16).padStart(2,'0')).join('');}\
function ensureDec(s){if(s.dec&&s.dec.state!=='closed')return;\
s.dec=new VideoDecoder({output:f=>{s.q.push(f);while(s.q.length>MAXBUF)s.q.shift().close();},\
error:e=>{document.getElementById('e').textContent=''+e;s.dec=null;s.waitkey=true;}});\
const cfg={optimizeForLatency:true,codec:s.cfg?codecStr(s.cfg):'avc1.42e01f'};\
if(s.cfg)cfg.description=s.cfg;try{s.dec.configure(cfg);}catch(err){document.getElementById('e').textContent=''+err;}}\
let ws;function connect(){ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');\
ws.binaryType='arraybuffer';\
ws.onmessage=e=>{const d=new Uint8Array(e.data);const kind=d[0],tag=d[1]===66?'B':'M';const s=S[tag];const pl=d.subarray(2);\
if(kind===67){s.cfg=pl.slice();if(s.dec){try{s.dec.close();}catch(_){}}s.dec=null;s.waitkey=true;ensureDec(s);return;}\
ensureDec(s);const key=kind===75;if(s.waitkey){if(!key)return;s.waitkey=false;}\
try{s.dec.decode(new EncodedVideoChunk({type:key?'key':'delta',timestamp:performance.now()*1000,data:pl}));}catch(_){}}; \
ws.onclose=()=>{for(const t in S)S[t].waitkey=true;setTimeout(connect,500);};}\
connect();\
function tick(){for(const t of['M','B']){const s=S[t];if(!s.started&&s.q.length>=MINBUF)s.started=true;\
if(s.started&&s.q.length){const f=s.q.shift();if(s.c.width!==f.displayWidth){s.c.width=f.displayWidth;s.c.height=f.displayHeight;}\
s.x.drawImage(f,0,0);f.close();if(!s.q.length)s.started=false;}}}\
setInterval(tick,66);</script></body></html>";

// ─────────────────────────── MJPEG (fallback) ───────────────────────────

/// A shared latest-JPEG slot with a monotonic version; a consumer waits on the
/// [`Condvar`] until the version advances, delivering each frame exactly once.
struct Shared {
    frame: Mutex<(Arc<Vec<u8>>, u64)>,
    cv: Condvar,
}

impl Shared {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            frame: Mutex::new((Arc::new(Vec::new()), 0)),
            cv: Condvar::new(),
        })
    }

    fn publish(&self, jpeg: Vec<u8>) {
        {
            let mut g = self.frame.lock().unwrap_or_else(|e| e.into_inner());
            g.0 = Arc::new(jpeg);
            g.1 += 1;
        }
        self.cv.notify_all();
    }

    fn wait_newer(&self, last: u64) -> (Arc<Vec<u8>>, u64) {
        let g = self.frame.lock().unwrap_or_else(|e| e.into_inner());
        let (g, _) = self
            .cv
            .wait_timeout_while(g, Duration::from_millis(500), |f| f.1 == last)
            .unwrap_or_else(|e| e.into_inner());
        (g.0.clone(), g.1)
    }
}

type Slot = Arc<Shared>;

// ─────────────────────────── H.264 broadcast ───────────────────────────

/// A stream's `(tag, avcC codec-config)`.
type StreamConfig = (u8, Arc<Vec<u8>>);

/// One H.264 message fanned out to WebSocket viewers.
#[derive(Clone)]
enum H264Msg {
    /// avcC codec-config for stream `tag` (viewer configures its decoder from this).
    Config { tag: u8, data: Arc<Vec<u8>> },
    /// One access unit for stream `tag`; `key` marks an IDR (a viewer starts on one).
    Frame {
        tag: u8,
        key: bool,
        data: Arc<Vec<u8>>,
    },
}

/// Fan-out of H.264 access units to connected viewers, caching the latest per-stream
/// codec-config so a new viewer can configure its decoder before the next keyframe. A
/// viewer whose bounded queue fills (a full buffer behind) is dropped — its browser
/// reconnects and re-syncs on the next keyframe.
struct H264Cast {
    clients: Mutex<Vec<SyncSender<H264Msg>>>,
    configs: Mutex<Vec<StreamConfig>>,
}

impl H264Cast {
    fn new() -> Self {
        Self {
            clients: Mutex::new(Vec::new()),
            configs: Mutex::new(Vec::new()),
        }
    }

    fn broadcast(&self, msg: H264Msg) {
        let mut cs = self.clients.lock().unwrap_or_else(|e| e.into_inner());
        cs.retain(|tx| tx.try_send(msg.clone()).is_ok());
    }

    /// Cache + broadcast a stream's codec-config, skipping an unchanged repeat.
    fn publish_config(&self, tag: u8, data: &[u8]) {
        let arc = {
            let mut cfgs = self.configs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(slot) = cfgs.iter_mut().find(|(t, _)| *t == tag) {
                if slot.1.as_slice() == data {
                    return;
                }
                slot.1 = Arc::new(data.to_vec());
                slot.1.clone()
            } else {
                let arc = Arc::new(data.to_vec());
                cfgs.push((tag, arc.clone()));
                arc
            }
        };
        self.broadcast(H264Msg::Config { tag, data: arc });
    }

    fn publish_frame(&self, tag: u8, key: bool, data: Arc<Vec<u8>>) {
        self.broadcast(H264Msg::Frame { tag, key, data });
    }

    /// Register a new viewer; returns its receiver plus the cached configs to send it
    /// first (so its decoder is configured before the first keyframe arrives).
    fn subscribe(&self) -> (Receiver<H264Msg>, Vec<StreamConfig>) {
        let (tx, rx) = sync_channel(120);
        self.clients
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(tx);
        let cfgs = self
            .configs
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        (rx, cfgs)
    }
}

// ─────────────────────────── server ───────────────────────────

/// A running live-view server. Clone-free handle over the JPEG slots + H.264 cast.
pub struct MjpegServer {
    main: Slot,
    bev: Slot,
    h264: Arc<H264Cast>,
}

impl MjpegServer {
    /// Bind `0.0.0.0:port` and spawn the accept loop. Returns once bound.
    pub fn spawn(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        let (main, bev, h264) = (Shared::new(), Shared::new(), Arc::new(H264Cast::new()));
        let (m, b, h) = (main.clone(), bev.clone(), h264.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (m, b, h) = (m.clone(), b.clone(), h.clone());
                std::thread::spawn(move || {
                    let _ = serve_client(stream, &m, &b, &h);
                });
            }
        });
        Ok(Self { main, bev, h264 })
    }

    /// Publish the latest JPEG for each stream (MJPEG fallback path).
    pub fn publish(&self, main_jpeg: Vec<u8>, bev_jpeg: Vec<u8>) {
        self.main.publish(main_jpeg);
        self.bev.publish(bev_jpeg);
    }

    /// Publish (or refresh) a stream's H.264 avcC codec-config. `tag` is `b'M'`/`b'B'`.
    pub fn publish_h264_config(&self, tag: u8, avcc: &[u8]) {
        self.h264.publish_config(tag, avcc);
    }

    /// Publish one H.264 access unit for a stream. `tag` is `b'M'`/`b'B'`.
    pub fn publish_h264_frame(&self, tag: u8, key: bool, data: Vec<u8>) {
        self.h264.publish_frame(tag, key, Arc::new(data));
    }
}

/// Route one client by request path: `/ws` upgrades to the H.264 WebSocket, `/main` /
/// `/bev` stream MJPEG, else the index page.
fn serve_client(mut s: TcpStream, main: &Slot, bev: &Slot, h264: &H264Cast) -> std::io::Result<()> {
    let _ = s.set_nodelay(true); // flush frames immediately — no Nagle coalescing latency
    let mut req = [0u8; 2048];
    let n = s.read(&mut req).unwrap_or(0);
    let reqs = std::str::from_utf8(&req[..n]).unwrap_or("");
    let path = reqs.split_whitespace().nth(1).unwrap_or("/");
    match path {
        "/ws" => match ws_key(reqs) {
            Some(key) => serve_ws_h264(s, &key, h264),
            None => Ok(()),
        },
        "/main" => serve_mjpeg(s, main),
        "/bev" => serve_mjpeg(s, bev),
        _ => write!(
            s,
            "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            INDEX_HTML.len(),
            INDEX_HTML
        ),
    }
}

/// Push each newly-published frame once, as `multipart/x-mixed-replace` JPEG.
fn serve_mjpeg(mut s: TcpStream, latest: &Slot) -> std::io::Result<()> {
    s.write_all(
        b"HTTP/1.0 200 OK\r\nConnection: close\r\nCache-Control: no-cache\r\n\
          Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n",
    )?;
    let mut last = 0u64;
    loop {
        let (jpeg, version) = latest.wait_newer(last);
        if version == last {
            continue;
        }
        last = version;
        if !jpeg.is_empty() {
            write!(
                s,
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                jpeg.len()
            )?;
            s.write_all(&jpeg)?;
            s.write_all(b"\r\n")?;
        }
    }
}

/// Complete the WebSocket handshake, then relay H.264: cached configs first, then each
/// access unit in order (skipping deltas until the first keyframe per stream).
fn serve_ws_h264(mut s: TcpStream, key: &str, cast: &H264Cast) -> std::io::Result<()> {
    let accept = base64(&sha1(
        format!("{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11").as_bytes(),
    ));
    write!(
        s,
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
         Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )?;
    let (rx, cfgs) = cast.subscribe();
    for (tag, data) in cfgs {
        ws_send(&mut s, &[b'C', tag], &data)?;
    }
    let mut started = [false; 2]; // per stream: seen a keyframe yet (M=0, B=1)
    for msg in rx {
        match msg {
            H264Msg::Config { tag, data } => ws_send(&mut s, &[b'C', tag], &data)?,
            H264Msg::Frame { tag, key, data } => {
                let idx = (tag == b'B') as usize;
                if !started[idx] {
                    if !key {
                        continue;
                    }
                    started[idx] = true;
                }
                ws_send(&mut s, &[if key { b'K' } else { b'D' }, tag], &data)?;
            }
        }
    }
    Ok(())
}

/// Send one unmasked binary WebSocket frame: `header` bytes + `payload`.
fn ws_send(s: &mut TcpStream, header: &[u8], payload: &[u8]) -> std::io::Result<()> {
    let len = header.len() + payload.len();
    let mut hdr = Vec::with_capacity(10 + header.len());
    hdr.push(0x82); // FIN + binary opcode
    if len < 126 {
        hdr.push(len as u8);
    } else if len <= 0xFFFF {
        hdr.push(126);
        hdr.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        hdr.push(127);
        hdr.extend_from_slice(&(len as u64).to_be_bytes());
    }
    hdr.extend_from_slice(header);
    s.write_all(&hdr)?;
    s.write_all(payload)
}

/// Extract the `Sec-WebSocket-Key` header value from a raw request (case-insensitive).
fn ws_key(req: &str) -> Option<String> {
    req.split("\r\n").find_map(|line| {
        let (name, val) = line.split_once(':')?;
        name.trim()
            .eq_ignore_ascii_case("sec-websocket-key")
            .then(|| val.trim().to_string())
    })
}

/// Standard Base64 (RFC 4648) with `=` padding.
fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18 & 63) as usize] as char);
        out.push(T[(n >> 12 & 63) as usize] as char);
        out.push(if c.len() > 1 {
            T[(n >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// SHA-1 (RFC 3174) — needed only for the WebSocket accept key.
fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let ml = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&ml.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for (i, wi) in w.iter_mut().enumerate().take(16) {
            *wi = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let tmp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = tmp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, hi) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&hi.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_known_vector() {
        let hex: String = sha1(b"abc").iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn base64_padding() {
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
    }

    #[test]
    fn ws_accept_rfc6455_example() {
        let acc = base64(&sha1(
            b"dGhlIHNhbXBsZSBub25jZQ==258EAFA5-E914-47DA-95CA-C5AB0DC85B11",
        ));
        assert_eq!(acc, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn ws_key_parse_case_insensitive() {
        let req = "GET /ws HTTP/1.1\r\nHost: x\r\nSec-WebSocket-Key: abc123==\r\n\r\n";
        assert_eq!(ws_key(req).as_deref(), Some("abc123=="));
    }

    #[test]
    fn h264cast_caches_and_dedups_config() {
        let cast = H264Cast::new();
        cast.publish_config(b'M', &[1, 2, 3]);
        let (_rx, cfgs) = cast.subscribe();
        assert_eq!(cfgs.len(), 1);
        assert_eq!(cfgs[0].1.as_slice(), &[1, 2, 3]);
        // Same bytes → no new config entry / no duplicate.
        cast.publish_config(b'M', &[1, 2, 3]);
        assert_eq!(cast.configs.lock().unwrap().len(), 1);
    }
}
