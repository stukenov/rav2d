# A frame without a primary reference stores its default CDFs, not "none"

dav2d keeps a static CDF context as `{ref = NULL, qcat, rtxp}`, so a
reference refreshed without CDF update remembers the quantizer category and
`reduced_tx_part_set` it started from. The port stored `None`, and whoever
read that slot later rebuilt defaults with its own category. Now
`init_from_defaults(qcat, reduced_tx_part_set)` is materialized into
`in_cdf` up front, which costs one CDF clone per such frame and removes the
special case. Same commit as [[pri-sec-cdf-average-clamps-the-count]].
