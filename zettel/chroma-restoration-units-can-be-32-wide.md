# Chroma restoration units can be 32 wide, narrower than the 64-column walk

With 4:2:0 and a 64 luma unit, the chroma unit is 32 samples. The
restoration pass walks a superblock row 64 columns at a time and picks one
unit per block, so the second 32 columns were filtered with the first
unit's coefficients. dav2d walks the same way. `lr_sbrow_plane` now steps
by `min(64, unit_size)`.

The step falls back to 64 when the unit size is below 32: a frame with GDF
but no restoration on the plane still runs this walk, and its unit size is
unset (log2 0), which made the step 1 and broke every GDF vector.

Next to it was a port slip of the same family: the unit row was taken from
the stripe start plus the 8-row offset only on the first superblock row of
a tile row, the reverse of dav2d's `y + 8 * !first_sby_in_tile_row`. Frames
with a single unit row cannot tell the two apart, which is why the
repository vectors (at most 128 rows) never did.
