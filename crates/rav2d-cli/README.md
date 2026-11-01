# rav2d-cli

Command-line AV2 decoder built on the [`rav2d`](https://crates.io/crates/rav2d) library, a memory-safe Rust port of dav2d. Reads IVF or a raw OBU stream (the format of the AV2 test vectors) and writes Y4M, raw YUV or an MD5 of the decoded frames.

## Install

Prebuilt binaries for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows (x86_64) are attached to each [GitHub release](https://github.com/stukenov/rav2d/releases). With a Rust toolchain (1.88+):

```sh
cargo install rav2d-cli
```

The binary is called `rav2d`.

## Usage

```sh
rav2d in.obu -o out.y4m                 # decode to Y4M
rav2d in.obu -o out.yuv                 # raw planes, no headers
rav2d in.ivf -o - | ffplay -            # Y4M on stdout
rav2d in.obu --md5                      # the hash `avmdec --md5` prints
rav2d in.obu --verify in.obu.md5        # compare with it; exit 1 if it differs
rav2d in.obu --info                     # one line per frame, then the stream
rav2d in.obu --skip 10 -l 1 -o f10.yuv  # just the eleventh frame
rav2d in.obu                            # decode only, prints the speed
cat in.obu | rav2d - --md5              # read stdin
```

## Options

| Flag | Description |
|------|-------------|
| `-o, --output <FILE>` | Output file, `-` for stdout. The format follows the extension: `.y4m`, `.yuv`, `.md5` |
| `-m, --muxer <y4m\|yuv\|md5>` | Output format, overriding the extension; stdout defaults to Y4M |
| `--md5` | Print the MD5 of the decoded planes |
| `--verify <HASH\|FILE>` | Compare the MD5 with a hash, or with the hash in a `.md5` file |
| `--info` | Describe each frame and the stream instead of writing frames |
| `-l, --limit <N>` | Stop after N frames have been output |
| `--skip <N>` | Decode but do not output the first N frames |
| `-t, --threads <N>` | Threads (0 = auto); today only the output copy and film grain run in parallel |
| `--no-grain` | Skip film grain synthesis |
| `--operating-point <N>` | Scalability operating point (0-31) |
| `--all-layers <BOOL>` | Output every spatial layer (default) or only the highest |
| `--size-limit <PIXELS>` | Refuse frames larger than this |
| `--strict` | Fail on streams that break the specification |
| `--keep-going` | Continue after a frame fails to decode (still exits with 1) |
| `--no-asm` | Use the Rust code instead of the NEON assembly |
| `-q, --quiet` | Print nothing but errors |

The MD5 covers the decoded planes in display order, 16-bit little-endian above 8 bits, which is what `avmdec --md5` hashes. The `.yuv` output is the same bytes.

Y4M has one frame size per file, so a stream that changes resolution stops with an error; write `.yuv` or `.md5` for those.

## Exit status

| Code | Meaning |
|------|---------|
| 0 | Decoded, and the MD5 matched if `--verify` was given |
| 1 | The stream failed to decode, or the MD5 did not match |
| 2 | Bad arguments, unreadable input or unwritable output |

## License

BSD-2-Clause
