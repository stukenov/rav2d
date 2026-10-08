# IntraBC chroma inherits the luma transform type, so the intra path must record it

dav2d reconstructs IntraBC blocks in the intra function, but `decode_coefs`
treats IntraBC chroma as inter: it does not derive the type from `uv_mode`,
it keeps the type it was given. That input is the luma type recorded in
`txtp_map` (recon_tmpl.c:2549 writes it, :3623 reads it). The port wrote
`txtp_map` only on the inter path and started intra chroma at DCT_DCT, so
an IntraBC block whose luma used identity or ADST read V coefficients with
the wrong scan and lost entropy sync a few blocks later.

Only the 128x128 superblock vector hit it
(`avmenc-b128`); the 64x64 vectors never chose IntraBC in an inter frame.
Ordinary intra chroma ignores the seed, which is why nothing else noticed.
See [[port-follows-dav2d-of-2-may]].
