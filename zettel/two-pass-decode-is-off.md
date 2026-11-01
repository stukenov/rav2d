# The two-pass decode is off until a scheduler uses it

Since July 2026 any `n_threads > 1`, which `Settings::default()` picks on
every multi-core machine, decoded each frame in two passes: entropy first,
then reconstruction replayed from staged data. Nothing ran in parallel on
top of it, and it was wrong. Between 1 and 4 threads it changed the output
of 28 of 41 test vectors, by 1 or 2 in a few chroma samples per frame.

`mt_matches_single_thread` missed it because it checked a hand-picked list
written before the qm1 and avmenc vectors existed. The new CLI's md5 test
caught it, since the CLI decodes with the default thread count.

0.4.1 always runs the fused single pass, and the determinism test now walks
every vector that decodes correctly. The two-pass code stays in place. Turn
it back on only together with the sbrow scheduler it was built for, after
finding the staging bug with that test.
