# Warp from a reference of another size is valid AV2 that dav2d does not decode

AVM (`av2-normative`, `warped_motion.c`) warps across scales with
`av2_ext_highbd_warp_affine_scaled_c`. dav2d has no such path — its
`recon_tmpl.c` says "no scaled support" — and clips the warp window to the
current frame, so it reads past a smaller reference buffer.

rav2d cannot be bit-exact with dav2d here, so it only stays in bounds:
`warp_ref_extent` clips the window to the reference when `svc[ref][0].scale`
is set (fuzz crash `crash-b5bf300c…`). For unscaled references nothing changes.

The real fix is porting AVM's scaled warp, checked against avmdec rather than
dav2d. See [[reference-views-span-the-decoded-rows]].
