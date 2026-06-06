# rav2d-cli

Command-line AV2 video decoder built on the [`rav2d`](../rav2d/) library — a memory-safe, bit-exact (vs the dav2d C reference) AV2 decoder supporting 8-bit and 10-bit streams.

Reads IVF container files and decodes AV2 bitstreams, optionally writing decoded frames as Y4M output.

## Usage

```sh
# Decode to Y4M
rav2d input.ivf -o output.y4m

# Decode-only benchmark (no file output)
rav2d input.ivf

# Decode with options
rav2d input.ivf -o output.y4m --threads 4 --limit 100 --no-grain
```

## Options

| Flag | Description |
|------|-------------|
| `-o, --output <FILE>` | Output file (Y4M format). Omit for decode-only benchmark |
| `-t, --threads <N>` | Number of threads (0 = auto) |
| `-l, --limit <N>` | Maximum frames to decode |
| `--no-grain` | Skip film grain synthesis |

## Supported Formats

- **Input**: IVF container with AV01/AV02 codec
- **Output**: Y4M (YUV4MPEG2) with 4:2:0 chroma subsampling

## Building

```sh
cargo build -p rav2d-cli --release
```

## License

BSD-2-Clause
