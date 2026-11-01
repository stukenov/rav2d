#!/usr/bin/env python3
# half_y4m.py in.y4m out.y4m frames: 2x2 box downscale of an 8-bit 4:2:0 y4m
import sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
f = open(src, 'rb')
hdr = f.readline().decode().split()
W = int(next(t[1:] for t in hdr if t.startswith('W'))); H = int(next(t[1:] for t in hdr if t.startswith('H')))
out = open(dst, 'wb')
out.write((' '.join(f'W{W//2}' if t[0]=='W' and t!='YUV4MPEG2' else f'H{H//2}' if t[0]=='H' else t for t in hdr) + '\n').encode())
def half(p, w, h):
    o = bytearray((w//2)*(h//2))
    for y in range(h//2):
        r0 = y*2*w; r1 = r0 + w
        for x in range(w//2):
            o[y*(w//2)+x] = (p[r0+2*x] + p[r0+2*x+1] + p[r1+2*x] + p[r1+2*x+1] + 2) >> 2
    return o
for _ in range(n):
    out.write(f.readline())
    for w, h in ((W, H), (W//2, H//2), (W//2, H//2)):
        out.write(half(f.read(w*h), w, h))
