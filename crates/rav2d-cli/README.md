# rav2d-cli

Command-line AV2 video decoder built on the [`rav2d`](../rav2d/) library — a memory-safe, bit-exact (vs the dav2d C reference) AV2 decoder supporting 8-bit and 10-bit streams.

Reads an IVF file or a raw OBU stream (the format of the AV2 test vectors) and optionally writes the decoded frames as Y4M.

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
| `-t, --threads <N>` | Number of threads (0 = auto); today only the output copy and film grain run in parallel |
| `-l, --limit <N>` | Maximum frames to decode |
| `--no-grain` | Skip film grain synthesis |

## Supported Formats

- **Input**: IVF, or a raw OBU stream (anything that does not start with `DKIF`)
- **Output**: Y4M (YUV4MPEG2) in the stream's own layout and bit depth: `420jpeg`/`420p10`, `mono`, `422`, `444`

## Building

```sh
cargo install rav2d-cli          # from crates.io
cargo build -p rav2d-cli --release   # from a checkout
```

Requires Rust 1.88+.

## License

BSD-2-Clause
