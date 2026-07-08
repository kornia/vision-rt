# gpu

Check and configure Jetson GPU/power state.

## Current power mode and clocks

```bash
sudo nvpmodel -q && sudo jetson_clocks --show 2>/dev/null | head -20
```

## Set MAXN_SUPER (3× GPU perf vs default 15W)

```bash
sudo nvpmodel -m 2 && sudo jetson_clocks
```

## GPU utilization live

```bash
watch -n1 'tegrastats | tr " " "\n" | grep -A1 "GR3D\|GPU"'
```

## CUDA/TRT versions

```bash
nvcc --version 2>/dev/null; python3 -c "import tensorrt; print('TRT', tensorrt.__version__)" 2>/dev/null; ls /usr/lib/aarch64-linux-gnu/libnvinfer.so.* | head -3
```
