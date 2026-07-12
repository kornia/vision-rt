//! A tiny MJPEG-over-HTTP server for live viewing on a phone browser (same LAN).
//!
//! Two JPEG streams — `/main` and `/bev` — plus an index page that stacks them.
//! The render loop calls [`MjpegServer::publish`] with the latest encoded frames;
//! each connected client is pushed the newest JPEG the instant it is published
//! (`multipart/x-mixed-replace`).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Index page: the two streams stacked vertically.
const INDEX_HTML: &str = "<!doctype html><html><head><meta name=viewport \
content='width=device-width,initial-scale=1'><style>body{margin:0;background:#111}\
img{display:block;width:100%;height:auto}</style></head><body>\
<img src=/main><img src=/bev></body></html>";

/// A shared latest-JPEG slot: the newest encoded frame plus a monotonically
/// increasing version. A client waits on the [`Condvar`] until the version moves
/// past the one it last sent, so it delivers each frame **exactly once, when it is
/// published** — no fixed-timer polling that would beat against the producer's
/// cadence and duplicate/drop frames (visible judder). The inner `Arc<Vec<u8>>` lets
/// a client clone the frame with a refcount bump (not a deep copy) and drop the lock
/// immediately.
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

    /// Swap in a new JPEG, bump the version, and wake every waiting client.
    fn publish(&self, jpeg: Vec<u8>) {
        {
            let mut g = self.frame.lock().unwrap_or_else(|e| e.into_inner());
            g.0 = Arc::new(jpeg);
            g.1 += 1;
        }
        self.cv.notify_all();
    }

    /// Block until a version newer than `last` is available (bounded so a stalled
    /// producer doesn't wedge the thread forever), then return `(jpeg, version)`.
    /// A spurious timeout returns the current `(jpeg, version)`; the caller re-checks
    /// `version` and only sends when it actually advanced.
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

/// A running MJPEG server. Clone-free handle over the two shared latest-JPEG slots.
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
    /// new `Arc` + bumps the version and wakes waiting clients — no copy.
    pub fn publish(&self, main_jpeg: Vec<u8>, bev_jpeg: Vec<u8>) {
        self.main.publish(main_jpeg);
        self.bev.publish(bev_jpeg);
    }
}

/// Route one client by request path: `/main` / `/bev` stream MJPEG, else the index.
fn serve_client(mut s: TcpStream, main: &Slot, bev: &Slot) -> std::io::Result<()> {
    let _ = s.set_nodelay(true); // flush each JPEG immediately — no Nagle coalescing latency
    let mut req = [0u8; 1024];
    let n = s.read(&mut req).unwrap_or(0);
    let path = std::str::from_utf8(&req[..n])
        .ok()
        .and_then(|r| r.split_whitespace().nth(1))
        .unwrap_or("/")
        .to_string();
    let latest = match path.as_str() {
        "/main" => main,
        "/bev" => bev,
        _ => {
            write!(
                s,
                "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                INDEX_HTML.len(),
                INDEX_HTML
            )?;
            return Ok(());
        }
    };
    s.write_all(
        b"HTTP/1.0 200 OK\r\nConnection: close\r\nCache-Control: no-cache\r\n\
          Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n",
    )?;
    // Push each newly-published frame exactly once, the moment it lands.
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
