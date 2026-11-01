# Small vectors miss what a 1080p frame exercises

Every shipped vector is at most 128x128, and all of them matched avmdec,
yet the first 1080p clip (crowd_run, 4 frames, `--qp=150`) differed in
three places:

- implicit CfL on a block that crosses the bottom of the frame read the
  block's own column instead of the left neighbour for the padded rows;
- BAWP on a 128x64 block derived a model per 64x64 unit instead of one
  for the block;
- the COMP_SEG mask of a 64x128 block was rebuilt per 64x64 unit, so the
  chroma, blended once for the whole block, saw only the last unit's mask.

The last two share a cause: blocks over 64px are predicted in 64x64
units, and per-block state (the BAWP model, the mask) must span the units.

Crops of the same clip (512x264 at qp 150, 190, 230) did not reproduce
any of them: at that size the encoder chooses differently. The CfL case
is pinned by a unit test; the 1080p streams stay outside the repository
(390 KB each, about 7000 s to encode on a runner without multithreading).
Run them by hand before a release: `tools/vectors` has no recipe for them
yet, the clip is `crowd_run_1080p50` from Xiph derf.
