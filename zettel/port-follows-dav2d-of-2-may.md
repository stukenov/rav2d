# The Rust decode logic followed dav2d of 2 May, not the pinned submodule

The submodule was moved from `bbbf12ec` (2 May) to `fafb909e` (19 August) in
"Catch the port up to the current dav2d", but only the film grain change our
vectors caught was ported. That commit says so itself: the tools upstream added
in between were not implemented. The oracle stayed green because no vector
reaches them.

So "bit-exact with dav2d" held for the corpus, not for the decoder. The
submodule now points at dav2d HEAD and the port is being brought up to it
vector by vector. Ported so far, with a vector proving it: quantizer matrices,
opfl_refine_type 2, compound warp in the warp bank, high-priority TMVPs,
no_cross_frame_context, reduced_ref_frame_mvs_mode (see the qm1 commit).
Known still missing: parity hiding (dav2d `f6859b3`, the flag is parsed and
the coefficient path ignores it), single-reference TIP, `max_dpb_size=16`,
`frame_offset` wraparound, the explicit_ref_frame_map bit per frame.

A port step is only verified when a vector exercises it. Porting from the C
diff without one moves code, not correctness — so new tools need new vectors
(`tools/vectors/`), and the oracle has to be built from the same dav2d
revision the port targets. See [[scaled-warp-is-not-in-dav2d]].
