# Where dav2d and avm disagree, rav2d follows avm

dav2d is the port's guide, but it is not the reference: avmdec is. Two
header rules in dav2d HEAD are wrong against avm, and the `unitsb` and
`tiles` vectors show it:

- The GDF adaptive bit. avm reads it when the frame holds more than one GDF
  unit, counted per tile (`init_gdf`, `gdf_block_num > 1`). dav2d reads it
  when `max(w, h) > 128`, which is wrong for SB-sized units and for frames
  split into tiles. Misreading it shifts every later header bit.
- The tile parity mask that shrinks CCSO/GDF units. avm tests the parity of
  each tile size but the last; dav2d ORs the starts `0..cols-1`, which skips
  the last inner start, so two 64px tiles keep 128px units.

`AVMENC_AVMDEC_ONLY` in the conformance tests lists vectors where rav2d is
checked against avmdec only, because the dav2d comparison would demand the
bug. See [[port-follows-dav2d-of-2-may]].
