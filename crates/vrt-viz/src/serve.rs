//! A tiny MJPEG-over-HTTP server for live viewing on a phone browser (same LAN).
//!
//! Two JPEG streams — `/main` and `/bev` — plus an index page that stacks them.
//! The render loop calls [`MjpegServer::publish`] with the latest encoded frames;
//! connected clients are each pushed the newest JPEG as `multipart/x-mixed-replace`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

/// Index page: the two streams stacked vertically.
const INDEX_HTML: &str = "<!doctype html><html><head><meta name=viewport \
content='width=device-width,initial-scale=1'><style>body{margin:0;background:#111}\
img{display:block;width:100%;height:auto}</style></head><body>\
<img src=/main><img src=/bev></body></html>";

/// A running MJPEG server. Clone-free handle over the two shared latest-JPEG slots.
pub struct MjpegServer {
    main: Arc<Mutex<Vec<u8>>>,
    bev: Arc<Mutex<Vec<u8>>>,
}

impl MjpegServer {
    /// Bind `0.0.0.0:port` and spawn the accept loop. Returns once bound; serving runs
    /// on background threads.
    pub fn spawn(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        let (main, bev) = (
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        );
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

    /// Publish the latest encoded JPEG for each stream (call every frame).
    pub fn publish(&self, main_jpeg: Vec<u8>, bev_jpeg: Vec<u8>) {
        *self.main.lock().unwrap_or_else(|e| e.into_inner()) = main_jpeg;
        *self.bev.lock().unwrap_or_else(|e| e.into_inner()) = bev_jpeg;
    }
}

/// Route one client by request path: `/main` / `/bev` stream MJPEG, else the index.
fn serve_client(
    mut s: TcpStream,
    main: &Arc<Mutex<Vec<u8>>>,
    bev: &Arc<Mutex<Vec<u8>>>,
) -> std::io::Result<()> {
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
    loop {
        let jpeg = latest.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if !jpeg.is_empty() {
            write!(
                s,
                "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                jpeg.len()
            )?;
            s.write_all(&jpeg)?;
            s.write_all(b"\r\n")?;
        }
        std::thread::sleep(std::time::Duration::from_millis(66)); // ~15 fps push
    }
}
