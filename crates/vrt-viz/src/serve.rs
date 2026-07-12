//! A tiny MJPEG + WebSocket server for live viewing on a phone browser.
//!
//! Endpoints:
//! - `/` — index page: a `<canvas>` client that pulls frames over **WebSocket** and
//!   plays them through a small **jitter buffer** (paced render + drop-stale) so
//!   uneven network arrival (cellular / Tailscale) doesn't look choppy.
//! - `/ws` — one WebSocket stream carrying both views; each binary message is a
//!   1-byte tag (`M`/`B`) + a JPEG.
//! - `/main`, `/bev` — plain `multipart/x-mixed-replace` MJPEG (fallback / direct).
//!
//! The render loop calls [`MjpegServer::publish`] with the latest encoded frames;
//! every client is pushed the newest JPEG the instant it is published.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Index page: a WebSocket + canvas client with a jitter buffer. Two canvases (main
/// over BEV); frames are decoded off-thread (`createImageBitmap`) into per-stream
/// queues and drawn on a fixed ~15 fps tick once `MINBUF` frames are buffered, dropping
/// the oldest beyond `MAXBUF` so latency stays bounded. Falls back nowhere — if the
/// socket drops the canvas just holds the last frame.
const INDEX_HTML: &str = "<!doctype html><html><head><meta name=viewport \
content='width=device-width,initial-scale=1'><style>body{margin:0;background:#111}\
canvas{display:block;width:100%;height:auto}</style></head><body>\
<canvas id=m></canvas><canvas id=b></canvas><script>\
const cv={M:document.getElementById('m'),B:document.getElementById('b')};\
const cx={M:cv.M.getContext('2d'),B:cv.B.getContext('2d')};\
const q={M:[],B:[]},started={M:false,B:false},MAXBUF=5,MINBUF=2;\
const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');\
ws.binaryType='arraybuffer';\
ws.onmessage=async e=>{const d=new Uint8Array(e.data);const tag=d[0]===77?'M':'B';\
const bmp=await createImageBitmap(new Blob([d.subarray(1)],{type:'image/jpeg'}));\
const qq=q[tag];qq.push(bmp);while(qq.length>MAXBUF){const o=qq.shift();o.close&&o.close();}};\
function tick(){for(const tag of ['M','B']){const qq=q[tag];\
if(!started[tag]&&qq.length>=MINBUF)started[tag]=true;\
if(started[tag]&&qq.length){const bmp=qq.shift(),c=cv[tag];\
if(c.width!==bmp.width){c.width=bmp.width;c.height=bmp.height;}\
cx[tag].drawImage(bmp,0,0);bmp.close&&bmp.close();if(!qq.length)started[tag]=false;}}}\
setInterval(tick,66);</script></body></html>";

/// A shared latest-JPEG slot: the newest encoded frame plus a monotonically
/// increasing version. A consumer waits on the [`Condvar`] until the version moves
/// past the one it last sent, so it delivers each frame **exactly once, when it is
/// published** — no fixed-timer polling that would beat against the producer's cadence
/// and duplicate/drop frames. The inner `Arc<Vec<u8>>` lets a consumer clone the frame
/// with a refcount bump (not a deep copy) and drop the lock immediately.
struct Shared {
    /// `(latest JPEG, version)`; `version == 0` means nothing published yet.
    frame: Mutex<(Arc<Vec<u8>>, u64)>,
    /// Signalled on every [`publish`](MjpegServer::publish).
    cv: Condvar,
}

impl Shared {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            frame: Mutex::new((Arc::new(Vec::new()), 0)),
            cv: Condvar::new(),
        })
    }

    /// Swap in a new JPEG, bump the version, and wake every waiting consumer.
    fn publish(&self, jpeg: Vec<u8>) {
        {
            let mut g = self.frame.lock().unwrap_or_else(|e| e.into_inner());
            g.0 = Arc::new(jpeg);
            g.1 += 1;
        }
        self.cv.notify_all();
    }

    /// Block until a version newer than `last` is available (bounded so a stalled
    /// producer doesn't wedge the thread forever), then return `(jpeg, version)`. A
    /// spurious timeout returns the current pair; the caller re-checks `version` and
    /// only sends when it actually advanced.
    fn wait_newer(&self, last: u64) -> (Arc<Vec<u8>>, u64) {
        let g = self.frame.lock().unwrap_or_else(|e| e.into_inner());
        let (g, _) = self
            .cv
            .wait_timeout_while(g, Duration::from_millis(500), |f| f.1 == last)
            .unwrap_or_else(|e| e.into_inner());
        (g.0.clone(), g.1)
    }

    /// The current JPEG without waiting (for pairing the other stream on a WS push).
    fn latest(&self) -> Arc<Vec<u8>> {
        self.frame
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .0
            .clone()
    }
}

type Slot = Arc<Shared>;

/// A running MJPEG + WebSocket server. Clone-free handle over the two shared
/// latest-JPEG slots.
pub struct MjpegServer {
    main: Slot,
    bev: Slot,
}

impl MjpegServer {
    /// Bind `0.0.0.0:port` and spawn the accept loop. Returns once bound; serving runs
    /// on background threads.
    pub fn spawn(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        let (main, bev) = (Shared::new(), Shared::new());
        let (m, b) = (main.clone(), bev.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (m, b) = (m.clone(), b.clone());
                std::thread::spawn(move || {
                    let _ = serve_client(stream, &m, &b);
                });
            }
        });
        Ok(Self { main, bev })
    }

    /// Publish the latest encoded JPEG for each stream (call every frame). Swaps in a
    /// new `Arc` + bumps the version and wakes waiting consumers — no copy.
    pub fn publish(&self, main_jpeg: Vec<u8>, bev_jpeg: Vec<u8>) {
        self.main.publish(main_jpeg);
        self.bev.publish(bev_jpeg);
    }
}

/// Route one client by request path: `/ws` upgrades to WebSocket, `/main` / `/bev`
/// stream MJPEG, else the index page.
fn serve_client(mut s: TcpStream, main: &Slot, bev: &Slot) -> std::io::Result<()> {
    let _ = s.set_nodelay(true); // flush each JPEG immediately — no Nagle coalescing latency
    let mut req = [0u8; 2048];
    let n = s.read(&mut req).unwrap_or(0);
    let reqs = std::str::from_utf8(&req[..n]).unwrap_or("");
    let path = reqs.split_whitespace().nth(1).unwrap_or("/");
    match path {
        "/ws" => match ws_key(reqs) {
            Some(key) => serve_ws(s, &key, main, bev),
            None => Ok(()),
        },
        "/main" => serve_mjpeg(s, main),
        "/bev" => serve_mjpeg(s, bev),
        _ => {
            write!(
                s,
                "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                INDEX_HTML.len(),
                INDEX_HTML
            )
        }
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
            continue; // spurious wake / producer stalled — keep waiting, don't resend
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

/// Complete the WebSocket handshake, then push both streams as binary frames tagged
/// with a leading `M`/`B` byte. Cadence is driven by the main slot; the BEV's latest is
/// paired on each tick (they are published back-to-back).
fn serve_ws(mut s: TcpStream, key: &str, main: &Slot, bev: &Slot) -> std::io::Result<()> {
    let accept = base64(&sha1(
        format!("{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11").as_bytes(),
    ));
    write!(
        s,
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
         Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )?;
    let mut last = 0u64;
    loop {
        let (mj, version) = main.wait_newer(last);
        if version == last {
            continue;
        }
        last = version;
        let bj = bev.latest();
        if !mj.is_empty() {
            ws_send(&mut s, b'M', &mj)?;
        }
        if !bj.is_empty() {
            ws_send(&mut s, b'B', &bj)?;
        }
    }
}

/// Send one unmasked binary WebSocket frame: `tag` byte + `payload`.
fn ws_send(s: &mut TcpStream, tag: u8, payload: &[u8]) -> std::io::Result<()> {
    let len = payload.len() + 1; // + the tag byte
    let mut hdr = Vec::with_capacity(11);
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
    hdr.push(tag);
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
        // RFC 6455 §1.3: key "dGhlIHNhbXBsZSBub25jZQ==" → accept below.
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
}
