# avm states its chroma rules for 4:2:0; 4:2:2 and 4:4:4 follow from the exceptions

dav2d and the port stopped 4:2:2 and 4:4:4 at the sequence header (profiles
3 and 4 were rejected), so none of their chroma paths had been run. Making
the two vectors match avmdec turned up one shape over and over: avm writes a
rule in luma block sizes as 4:2:0 would see it, then adds a real-subsampling
check, and the port had one or the other.

- Chroma-tree partitions: the size rules (8x8 is a leaf, 8x16 splits only
  horizontally, no extended split at or below 16x16) are luma sizes for
  every layout, but a split whose first sub-block has no chroma block size
  at the real subsampling is also out (4:2:2 16x64 VERT would be 4x64, and
  avm has no 4x64 although the table here does). Shared trees get the same
  check.
- MHCCP luma is the filter of the layout (copy for 4:4:4, horizontal pair
  for 4:2:2), and avm predicts from the neighbour buffer, so the block's
  left tap is not clamped as CfL's is.
- Size gates written with `imax` or "chroma is merged" in mind break once
  every block owns its chroma: the TIP flag, MHCCP corners, the single-ref
  chroma of thin compound blocks, and OPFL chroma sub-blocks, which in
  4:4:4 and 4:2:2 take the co-located luma sub-block's MV.

The 80-stream 4:2:0 corpus is the check that none of this moved 4:2:0. See
[[rav2d-follows-avm-where-dav2d-differs]].
