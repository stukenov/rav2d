# `cond as u32 & read()` reads the symbol even when the condition is false

dav2d writes `const int dir = imax(w, h) < 64 && dav2d_msac_decode_bool_equi(msac);`.
The port turned it into `(imax(..) < 64) as u32 & msac.decode_bool_bypass()`,
and `&` on integers evaluates both sides, so a palette block 64 wide or
tall read a direction bit that is not in the stream. It only shows with
screen content tools and a 64-sample palette block, which came first in
issue 9's Vertical_Bayshore qp210 stream, 52 frames in.

A condition guarding a symbol read must be an `if`, not an arithmetic
`&` or `*`. A grep for `as u32 & msac` and similar found no other case.
