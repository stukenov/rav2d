# At the top superblock edge, neighbours come from the row above, not the live context

Three port slips with one shape, found by avmenc streams at `--cpu-used=2`
and `4` (CIF, 8 frames):

- SMOOTH interintra forced the top-right count `n_tr` to 0 on the first
  row of a superblock, marked "simplified". dav2d takes the whole width
  there (`n_tr = w`): the row above is complete. The chroma of a 16x64
  wedge-interintra block came out up to 3 off.
- The NEWMV_NEWMV warp-causal check read the above neighbour's refs from
  the live `a` context. On the first row of a superblock avm reads the line
  buffer at 8x8 resolution, and blocks already decoded in this superblock
  have overwritten `a` (here the left neighbour at column 84). The flag
  went unread and the stream desynced.

- The WARP_NEWMV warp-extend/causal gate (`has_cs_ext`) had the same live
  read, with a comment calling the boundary case a follow-up. A 10-bit CIF
  stream desynced on it after four frames.

`recon.a_sb_cache` is the snapshot taken before each superblock; any
top-neighbour read at `by & (sb_step - 1) == 0` should use it. The warp
context already did. See [[sdp-chroma-restoration-follows-the-luma-tree]].
