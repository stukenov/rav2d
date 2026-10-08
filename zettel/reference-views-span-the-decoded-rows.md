# A reference plane view spans the decoded 8-aligned rows, not the visible height

Reconstruction writes whole 8x8 luma blocks, so a frame of height 66 holds
decoded samples down to row 72. dav2d clips reference reads to `f->bh * 4` and
reads that overhang; a Rust slice cut at `p.h` panicked there instead
(fuzz crash `decode_settings/crash-5233181b…`, BAWP template read).

`ref_plane_rows` in `decode.rs` gives every reference view the same 8-aligned
height. The memory is there: the decoder already writes those rows into the
current frame, whose planes are padded to 128 rows.

Open question: the unscaled MC path still clamps to the reference's visible
size (`imin(right, ref_pw)`), while dav2d clamps to `f->bw * 4`/`f->bh * 4`.
On frames whose width or height is not a multiple of 8 (854x480, 426x240)
that can differ from dav2d. No vector in the corpus has such a size; one has to
be made before touching it. See [[scaled-warp-is-not-in-dav2d]].
