# The primary/secondary CDF average keeps dav2d's arithmetic and saturates the count

dav2d averages each CDF entry's count/para word as one u16,
`(a*7 + b + 4) >> 3`, on the assumption that both references share the para
byte. They do not when the references started from defaults of different
quantizer categories: para is per category (75 against 3 for the same
entry), and the fraction of the para difference lands in the count byte,
past the 32 that `MSAC_RATE` is sized for. dav2d asserts in debug and reads
past its rate table in release.

Averaging count and para separately would have been cleaner, but it
diverges from dav2d in the one case where dav2d's result is still defined
(para difference of 1 mod 8 with both counts 0 gives count 32). So the u16
arithmetic stays and only a count over 32 is saturated: bit-exact wherever
dav2d is, no panic where it is not. Found by the October 2026 fuzz run.
See [[rav2d-follows-avm-where-dav2d-differs]].
