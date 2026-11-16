# Where dav2d and avm disagree, rav2d follows avm

dav2d is the port's guide, but it is not the reference: avmdec is. These
header rules in dav2d HEAD are wrong against avm (the `unitsb` and
`tiles` vectors show the first two, a fuzz crash showed the third):

- The GDF adaptive bit. avm reads it when the frame holds more than one GDF
  unit, counted per tile (`init_gdf`, `gdf_block_num > 1`). dav2d reads it
  when `max(w, h) > 128`, which is wrong for SB-sized units and for frames
  split into tiles. Misreading it shifts every later header bit.
- The tile parity mask that shrinks CCSO/GDF units. avm tests the parity of
  each tile size but the last; dav2d ORs the starts `0..cols-1`, which skips
  the last inner start, so two 64px tiles keep 128px units.
- The NS-Wiener class count. avm reads its 3 bits only where more than
  one class is allowed (`max_num_classes`), which is luma alone. dav2d
  reads them for chroma too, so chroma frame filters shift the header,
  and a corrupt stream could give chroma several classes and run the
  multi-class kernel past the 2-row chroma line store.

`AVMENC_AVMDEC_ONLY` in the conformance tests lists vectors where rav2d is
checked against avmdec only, because the dav2d comparison would demand the
bug. See [[port-follows-dav2d-of-2-may]].
