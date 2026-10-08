# A reference is read over its 8-aligned decoded area, not its visible size

Reconstruction writes whole 8x8 luma blocks, so a frame of height 66 holds
decoded samples down to row 72. dav2d clips reference reads to `f->bw * 4` /
`f->bh * 4` and reads that overhang like any other sample.

rav2d cut its reference views at the visible `p.h` and clipped motion
compensation to the visible width and height. The first made a BAWP read panic
(fuzz crash `decode_settings/crash-5233181b…`); the second silently replicated
the last visible row or column where dav2d reads decoded data, so every frame
whose width or height is not a multiple of 8 — 854x480 is one — decoded wrong
once motion reached the edge. `avmenc-odd.obu` (66x50) is the vector that pins
it. Now `ref_plane_rows` gives every reference view the 8-aligned height and the
unscaled MC paths clip to the frame's 8-aligned extent, as dav2d does.

The memory is there: the decoder writes those rows itself into planes padded
to 128 rows. See [[scaled-warp-is-not-in-dav2d]] for the scaled case.
