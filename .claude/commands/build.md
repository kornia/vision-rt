# build

Build the workspace or a specific crate in release mode.

```bash
cargo build --release -p rtsp_xfeat 2>&1 | tail -20
```

To build everything:
```bash
cargo build --release 2>&1 | tail -30
```

To check without linking (faster):
```bash
cargo check 2>&1 | tail -30
```
