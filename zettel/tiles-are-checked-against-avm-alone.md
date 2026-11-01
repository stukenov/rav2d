# Multi-tile streams are checked against avm alone

dav2d cannot decode a stream with more than one tile: it fails to parse
avmenc-tiles and stops after a frame or two of tilecols and tilerows. So
for tiles the frame-by-frame dav2d oracle is off and only avmdec's md5
gates them (`AVMENC_AVMDEC_ONLY`). The tile code in the port came from
dav2d, so it was never right either.

Finding the four bugs took two tools, both kept out of the repository:

- Variants of a vector with one encoder tool switched off each
  (`--enable-cdef=0`, `--enable-refmvbank=0`, ...). The variant that
  starts matching points at the tool.
- An avmdec with a few `fprintf`s behind an environment variable, diffed
  against `RAV2D_BLK_TRACE`. It showed the MVs agreed, so the bug was
  prediction (BAWP), not candidate lists.

The vectors must come from the av2-normative avm the others came from
(`tools/vectors/SOURCE`). avm main is a different bitstream.
