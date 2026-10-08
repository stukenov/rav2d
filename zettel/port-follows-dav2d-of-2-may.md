# The Rust decode logic follows dav2d of 2 May, not the pinned submodule

The submodule was moved from `bbbf12ec` (2 May) to `fafb909e` (19 August) in
"Catch the port up to the current dav2d", but only the film grain change our
vectors caught was ported. That commit says so itself: the tools upstream added
in between were not implemented. The oracle stayed green because no vector
reaches them.

So "bit-exact with dav2d" holds for the corpus, not for the decoder. What is
known to be missing: parity hiding (dav2d `f6859b3`, 28 May — the flag is
parsed, the coefficient path ignores it), single-reference TIP, high-priority
TMVPs, `no_cross_frame_context`, `reduced_ref_frame_mvs_mode`,
`max_dpb_size=16`, and everything after `fafb909e`. Quantizer matrices were
missing too, although they predate May: tables existed but were never loaded.

A port step is only verified when a vector exercises it. Porting from the C
diff without one moves code, not correctness — so new tools need new vectors
(avmenc), and the oracle has to be built from the same dav2d revision the port
targets. See [[scaled-warp-is-not-in-dav2d]].
