#!/usr/bin/env python3
"""Cut a test clip out of an 8-bit 4:2:0 Y4M source.

The window, frame count, bit depth and chroma layout are what the vector
needs; the pixels are only ever the source's, shifted up for high bit depth
and repeated for denser chroma, so a clip never contains synthetic content.
"""

import argparse
import sys

LAYOUT_TAG = {"420": "420", "422": "422", "444": "444", "400": "mono"}


def read_frames(path):
    with open(path, "rb") as f:
        header = f.readline().decode().split()
        params = {tok[0]: tok[1:] for tok in header[1:]}
        w, h = int(params["W"]), int(params["H"])
        if params.get("C", "420").split("p")[0] not in ("420", "420jpeg", "420mpeg2"):
            sys.exit("source must be 8-bit 4:2:0")
        y_size, c_size = w * h, (w // 2) * (h // 2)
        while True:
            line = f.readline()
            if not line:
                return
            data = f.read(y_size + 2 * c_size)
            if len(data) < y_size + 2 * c_size:
                return
            yield w, h, data[:y_size], data[y_size:y_size + c_size], data[y_size + c_size:]


def crop(plane, stride, x, y, w, h):
    return [plane[(y + r) * stride + x:(y + r) * stride + x + w] for r in range(h)]


def write_sample_rows(out, rows, bitdepth):
    for row in rows:
        if bitdepth == 8:
            out.write(bytes(row))
        else:
            shift = bitdepth - 8
            buf = bytearray()
            for v in row:
                s = v << shift
                buf += bytes((s & 0xFF, s >> 8))
            out.write(buf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--size", required=True, help="WxH of the clip")
    p.add_argument("--offset", default="0x0", help="XxY of the window in the source")
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--bitdepth", type=int, default=8, choices=(8, 10, 12))
    p.add_argument("--layout", default="420", choices=sorted(LAYOUT_TAG))
    a = p.parse_args()

    w, h = (int(v) for v in a.size.split("x"))
    ox, oy = (int(v) for v in a.offset.split("x"))
    if ox % 2 or oy % 2:
        sys.exit("offset must be even so chroma stays aligned")
    tag = LAYOUT_TAG[a.layout] + ("" if a.bitdepth == 8 else f"p{a.bitdepth}")

    with open(a.dst, "wb") as out:
        out.write(f"YUV4MPEG2 W{w} H{h} F30:1 Ip A1:1 C{tag}\n".encode())
        n = 0
        for sw, sh, ys, us, vs in read_frames(a.src):
            if n == a.frames:
                break
            if ox + w > sw or oy + h > sh:
                sys.exit("window does not fit the source")
            out.write(b"FRAME\n")
            write_sample_rows(out, crop(ys, sw, ox, oy, w, h), a.bitdepth)
            if a.layout == "400":
                n += 1
                continue
            cw, ch = (w + 1) // 2, (h + 1) // 2
            for c in (us, vs):
                rows = crop(c, sw // 2, ox // 2, oy // 2, cw, ch)
                if a.layout in ("422", "444"):
                    rows = [r for r in rows for _ in (0, 1)][:h]
                if a.layout == "444":
                    rows = [bytes(v for v in r for _ in (0, 1))[:w] for r in rows]
                write_sample_rows(out, rows, a.bitdepth)
            n += 1
        if n < a.frames:
            sys.exit(f"source has only {n} frames")


if __name__ == "__main__":
    main()
