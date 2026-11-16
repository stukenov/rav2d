# 4:0:0 keeps avm's 4:2:0 chroma grouping, though it has no chroma

avm sets 4:0:0 subsampling to 1,1 and builds the chroma reference info
(`chroma_ref_info`, `offset_started`) as for 4:2:0. Blocks with a 4-sample
side are still grouped as if they shared one chroma block, and that grouping
reaches the bitstream: `read_is_inter_block` returns inter without a symbol
for a grouped block, and `is_tip_allowed_bsize` refuses TIP to it.

dav2d (and the port before) gave 4:0:0 no chroma size at all, so it read an
intra/inter symbol and a TIP flag where avm reads neither. The stream desyncs
in the first inter frame that has such a block. `cov-monochrome-128x128.obu`
had none in the luma that mattered for the old gate: dav2d and rav2d agreed
with each other and both differed from avmdec. That clip is now checked
against avmdec's md5.

The fix keeps the chroma size through the partition walk for 4:0:0, uses it
only for the grouping (`shares_chroma` in `decode_b`), and drops it before
anything that would code or predict chroma. See
[[chroma-rules-are-stated-for-420]], [[rav2d-follows-avm-where-dav2d-differs]].
