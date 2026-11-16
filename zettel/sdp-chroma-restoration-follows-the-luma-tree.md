# With SDP and 64x64 superblocks, chroma restoration units come after the luma tree

avm reads a plane's loop-restoration units at the root of the partition
tree that covers the plane. In an intra frame with SDP the luma and chroma
trees split at 64x64, so with 128x128 superblocks the root is shared and
all three planes are read together, but with 64x64 superblocks the split is
the root: luma units, the whole luma tree, then chroma units and the chroma
tree. rav2d and dav2d read all three planes before the split. Every vector
in the repository uses 128x128 superblocks, so nothing caught it until the
avmenc streams of issue 9, which desync on the first chroma unit.

The split is now done in `decode_tile_sbrow_entropy` itself, with the
chroma units read between the two `decode_sb` calls. Both calls must share
`dir`: the chroma root reads the luma root's split direction from it to
decide whether CfL is allowed. A fresh `dir` turned CfL off and desynced a
few blocks later. See [[rav2d-follows-avm-where-dav2d-differs]] and
[[small-vectors-miss-big-block-paths]].

The nine issue 9 streams are the regression check for this and for
[[chroma-restoration-units-can-be-32-wide]], [[integer-and-does-not-short-circuit]]
and the IntraBC CCTX case. They are cut from third-party test sequences, so
they stay out of the repository and are run by hand before a release
(qa-r3 `~/build/issue9`, `rav2d <stream> --verify <md5>` with the md5s in the
issue). Encodes of bus_cif with `--sb-size=64`, screen tuning and
`--cpu-used=6` did not reach any of these paths.
