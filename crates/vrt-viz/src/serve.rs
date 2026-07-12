//! A tiny live-view server: H.264 over WebSocket, decoded in the browser via WebCodecs.
//!
//! Endpoints:
//! - `/` — index page ([`client.html`]): a `<canvas>` client that decodes the H.264
//!   WebSocket stream with the browser's WebCodecs `VideoDecoder` and plays it through a
//!   small **jitter buffer** (paced render + drop-stale), so uneven network arrival
//!   doesn't look choppy. H.264 is inter-frame compressed → ~10–20× less bandwidth than
//!   MJPEG.
//! - `/ws` — one WebSocket carrying both views. Each binary message is `[kind, tag] +
//!   payload`: `kind` ∈ `C`(avcC config) / `K`(keyframe) / `D`(delta), `tag` ∈ `M`/`B`.
//!
//! The producer calls [`StreamServer::publish_h264_frame`] / `publish_h264_config` with
//! encoded access units; each viewer is pushed frames in order the instant they land,
//! and a viewer that falls a full buffer behind is dropped (its browser reconnects and
//! re-syncs on the next keyframe).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

/// The WebCodecs `<canvas>` client (kept as a real file so it can be edited/linted as
/// JS/HTML instead of an inline Rust string).
const INDEX_HTML: &str = include_str!("client.html");

/// Which of the two composited views a frame belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    /// The main annotated camera view.
    Main,
    /// The top-down BEV.
    Bev,
}

impl Stream {
    /// Wire tag byte sent to the browser client (`M` / `B`).
    fn tag(self) -> u8 {
        match self {
            Stream::Main => b'M',
            Stream::Bev => b'B',
        }
    }
    /// Dense index for per-stream arrays (`Main = 0`, `Bev = 1`).
    fn index(self) -> usize {
        self as usize
    }
}

/// A stream's `(view, avcC codec-config)`.
type StreamConfig = (Stream, Arc<Vec<u8>>);

/// One H.264 message fanned out to WebSocket viewers.
#[derive(Clone)]
enum H264Msg {
    /// avcC codec-config for a `stream` (viewer configures its decoder from this).
    Config { stream: Stream, data: Arc<Vec<u8>> },
    /// One access unit for a `stream`; `key` marks an IDR (a viewer starts on one).
    Frame {
        stream: Stream,
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
    fn publish_config(&self, stream: Stream, data: &[u8]) {
        let arc = {
            let mut cfgs = self.configs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(slot) = cfgs.iter_mut().find(|(s, _)| *s == stream) {
                if slot.1.as_slice() == data {
                    return;
                }
                slot.1 = Arc::new(data.to_vec());
                slot.1.clone()
            } else {
                let arc = Arc::new(data.to_vec());
                cfgs.push((stream, arc.clone()));
                arc
            }
        };
        self.broadcast(H264Msg::Config { stream, data: arc });
    }

    fn publish_frame(&self, stream: Stream, key: bool, data: Arc<Vec<u8>>) {
        self.broadcast(H264Msg::Frame { stream, key, data });
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

/// A running H.264-over-WebSocket live-view server. Clone-free handle over the fan-out.
pub struct StreamServer {
    h264: Arc<H264Cast>,
}

impl StreamServer {
    /// Bind `0.0.0.0:port` and spawn the accept loop. Returns once bound.
    pub fn spawn(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        let h264 = Arc::new(H264Cast::new());
        let h = h264.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let h = h.clone();
                std::thread::spawn(move || {
                    let _ = serve_client(stream, &h);
                });
            }
        });
        Ok(Self { h264 })
    }

    /// Publish (or refresh) a [`Stream`]'s H.264 avcC codec-config.
    pub fn publish_h264_config(&self, stream: Stream, avcc: &[u8]) {
        self.h264.publish_config(stream, avcc);
    }

    /// Publish one H.264 access unit for a [`Stream`].
    pub fn publish_h264_frame(&self, stream: Stream, key: bool, data: Vec<u8>) {
        self.h264.publish_frame(stream, key, Arc::new(data));
    }
}

/// Route one client by request path: `/ws` upgrades to the H.264 WebSocket, else serve
/// the index page.
fn serve_client(mut s: TcpStream, cast: &H264Cast) -> std::io::Result<()> {
    let _ = s.set_nodelay(true); // flush frames immediately — no Nagle coalescing latency
    let mut req = [0u8; 2048];
    let n = s.read(&mut req).unwrap_or(0);
    let reqs = std::str::from_utf8(&req[..n]).unwrap_or("");
    let path = reqs.split_whitespace().nth(1).unwrap_or("/");
    match (path, ws_key(reqs)) {
        ("/ws", Some(key)) => serve_ws_h264(s, &key, cast),
        ("/ws", None) => Ok(()),
        _ => write!(
            s,
            "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\nCache-Control: no-cache\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            INDEX_HTML.len(),
            INDEX_HTML
        ),
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
    for (stream, data) in cfgs {
        ws_send(&mut s, &[b'C', stream.tag()], &data)?;
    }
    let mut started = [false; 2]; // per stream: seen a keyframe yet (indexed by Stream)
    for msg in rx {
        match msg {
            H264Msg::Config { stream, data } => ws_send(&mut s, &[b'C', stream.tag()], &data)?,
            H264Msg::Frame { stream, key, data } => {
                let idx = stream.index();
                if !started[idx] {
                    if !key {
                        continue;
                    }
                    started[idx] = true;
                }
                ws_send(
                    &mut s,
                    &[if key { b'K' } else { b'D' }, stream.tag()],
                    &data,
                )?;
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

/// Standard Base64 (RFC 4648) with `=` padding. Hand-rolled to keep the WebSocket
/// handshake dependency-free (the companion [`sha1`] has no crate in the tree either).
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
        cast.publish_config(Stream::Main, &[1, 2, 3]);
        let (_rx, cfgs) = cast.subscribe();
        assert_eq!(cfgs.len(), 1);
        assert_eq!(cfgs[0].1.as_slice(), &[1, 2, 3]);
        cast.publish_config(Stream::Main, &[1, 2, 3]);
        assert_eq!(cast.configs.lock().unwrap().len(), 1);
    }
}
