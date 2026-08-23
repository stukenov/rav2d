/*
 * Copyright © 2018, VideoLAN and dav2d authors
 * Copyright © 2018, Janne Grunau
 * Copyright © 2018, Martin Storsjo
 * Copyright © 2026, Arm Limited
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "src/arm/asm.S"
#include "util.S"

.macro avg dst, t0, t1, t2, t3
        ld1             {\t0\().8h,\t1\().8h},   [x2],  #32
        ld1             {\t2\().8h,\t3\().8h},   [x3],  #32
        add             \t0\().8h,   \t0\().8h,   \t2\().8h
        add             \t1\().8h,   \t1\().8h,   \t3\().8h
        sqrshrun        \dst\().8b,  \t0\().8h,   #5
        sqrshrun2       \dst\().16b, \t1\().8h,   #5
.endm

.macro w_avg dst, t0, t1, t2, t3
        ld1             {\t0\().8h,\t1\().8h},   [x2],  #32
        ld1             {\t2\().8h,\t3\().8h},   [x3],  #32
        sub             \t2\().8h,   \t0\().8h,   \t2\().8h
        sub             \t3\().8h,   \t1\().8h,   \t3\().8h
        sqdmulh         \t2\().8h,   \t2\().8h,   v30.8h
        sqdmulh         \t3\().8h,   \t3\().8h,   v30.8h
        add             \t0\().8h,   \t2\().8h,   \t0\().8h
        add             \t1\().8h,   \t3\().8h,   \t1\().8h
        sqrshrun        \dst\().8b,  \t0\().8h,   #4
        sqrshrun2       \dst\().16b, \t1\().8h,   #4
.endm

.macro mask dst, t0, t1, t2, t3
        ld1             {v30.16b}, [x6],  #16
        ld1             {\t0\().8h,\t1\().8h},   [x2],  #32
        mul             v30.16b, v30.16b, v31.16b
        ld1             {\t2\().8h,\t3\().8h},   [x3],  #32
        shll            v28.8h, v30.8b,  #8
        shll2           v29.8h, v30.16b, #8
        sub             \t0\().8h,   \t2\().8h,   \t0\().8h
        sub             \t1\().8h,   \t3\().8h,   \t1\().8h
        sqdmulh         \t0\().8h,   \t0\().8h,   v28.8h
        sqdmulh         \t1\().8h,   \t1\().8h,   v29.8h
        add             \t0\().8h,   \t2\().8h,   \t0\().8h
        add             \t1\().8h,   \t3\().8h,   \t1\().8h
        sqrshrun        \dst\().8b,  \t0\().8h,   #4
        sqrshrun2       \dst\().16b, \t1\().8h,   #4
.endm

.macro bidir_fn type
function \type\()_8bpc_neon, export=1
        clz             w4,  w4
.ifc \type, w_avg
        // (a * weight + b * (16 - weight) + 128) >> 8
        // = ((a - b) * weight + (b << 4) + 128) >> 8
        // = ((a - b) * (weight-16) + (a << 4) + 128) >> 8
        // = ((b - a) * -weight + (b << 4) + 128) >> 8
        // = ((((b - a) * (-weight << 11) * 2) >> 16) + b + 8) >> 4
        // = ((((a - b) * ((weight-16) << 11) * 2) >> 16) + a + 8) >> 4
        movi            v31.8h, #16
        cmp             w6, #16
        dup             v30.8h, w6
        b.lt            1f
        // if weight >= 16
        sub             v30.8h, v30.8h, v31.8h   // weight - 16
        b               2f
1:
        mov             x7, x2                   // swap tmp1/tmp2
        mov             x2, x3
        mov             x3, x7
        neg             v30.8h, v30.8h           // -weight
2:
        shl             v30.8h, v30.8h, #11
.endif
.ifc \type, mask
        movi            v31.16b, #256-2
.endif
        movrel          x7,  \type\()_tbl
        sub             w4,  w4,  #25
        ldrsw           x4,  [x7, x4, lsl #2]
        \type           v4,  v0,  v1,  v2,  v3
        add             x7,  x7,  x4
        br              x7
40:
        AARCH64_VALID_JUMP_TARGET
        add             x7,  x0,  x1
        lsl             x1,  x1,  #1
4:
        subs            w5,  w5,  #4
        st1             {v4.s}[0],  [x0], x1
        st1             {v4.s}[1],  [x7], x1
        st1             {v4.s}[2],  [x0], x1
        st1             {v4.s}[3],  [x7], x1
        b.le            0f
        \type           v5,  v0,  v1,  v2,  v3
        subs            w5,  w5,  #4
        st1             {v5.s}[0],  [x0], x1
        st1             {v5.s}[1],  [x7], x1
        st1             {v5.s}[2],  [x0], x1
        st1             {v5.s}[3],  [x7], x1
        b.le            0f
        \type           v4,  v0,  v1,  v2,  v3
41:
        subs            w5,  w5,  #8
        st1             {v4.s}[0],  [x0], x1
        st1             {v4.s}[1],  [x7], x1
        \type           v5,  v0,  v1,  v2,  v3
        st1             {v4.s}[2],  [x0], x1
        st1             {v4.s}[3],  [x7], x1
        st1             {v5.s}[0],  [x0], x1
        st1             {v5.s}[1],  [x7], x1
        st1             {v5.s}[2],  [x0], x1
        st1             {v5.s}[3],  [x7], x1
        b.le            0f
        \type           v4,  v0,  v1,  v2,  v3
        b               41b
80:
        AARCH64_VALID_JUMP_TARGET
        add             x7,  x0,  x1
        lsl             x1,  x1,  #1
8:
        st1             {v4.8b},    [x0], x1
        \type           v5,  v0,  v1,  v2,  v3
        st1             {v4.d}[1],  [x7], x1
        st1             {v5.8b},    [x0], x1
        subs            w5,  w5,  #4
        st1             {v5.d}[1],  [x7], x1
        b.le            0f
        \type           v4,  v0,  v1,  v2,  v3
        b               8b
160:
        AARCH64_VALID_JUMP_TARGET
16:
        \type           v5,  v0,  v1,  v2,  v3
        st1             {v4.16b}, [x0], x1
        \type           v6,  v0,  v1,  v2,  v3
        st1             {v5.16b}, [x0], x1
        \type           v7,  v0,  v1,  v2,  v3
        st1             {v6.16b}, [x0], x1
        subs            w5,  w5,  #4
        st1             {v7.16b}, [x0], x1
        b.le            0f
        \type           v4,  v0,  v1,  v2,  v3
        b               16b
320:
        AARCH64_VALID_JUMP_TARGET
        add             x7,  x0,  x1
        lsl             x1,  x1,  #1
32:
        \type           v5,  v0,  v1,  v2,  v3
        \type           v6,  v0,  v1,  v2,  v3
        st1             {v4.16b,v5.16b}, [x0], x1
        \type           v7,  v0,  v1,  v2,  v3
        subs            w5,  w5,  #2
        st1             {v6.16b,v7.16b}, [x7], x1
        b.le            0f
        \type           v4,  v0,  v1,  v2,  v3
        b               32b
640:
        AARCH64_VALID_JUMP_TARGET
        add             x7,  x0,  x1
        lsl             x1,  x1,  #1
64:
        \type           v5,  v0,  v1,  v2,  v3
        \type           v6,  v0,  v1,  v2,  v3
        \type           v7,  v0,  v1,  v2,  v3
        \type           v16, v0,  v1,  v2,  v3
        \type           v17, v0,  v1,  v2,  v3
        st1             {v4.16b,v5.16b,v6.16b,v7.16b}, [x0], x1
        \type           v18, v0,  v1,  v2,  v3
        \type           v19, v0,  v1,  v2,  v3
        subs            w5,  w5,  #2
        st1             {v16.16b,v17.16b,v18.16b,v19.16b}, [x7], x1
        b.le            0f
        \type           v4, v0,  v1,  v2,  v3
        b               64b
0:
        ret
endfunc

jumptable \type\()_tbl
        .word 640b  - \type\()_tbl
        .word 320b  - \type\()_tbl
        .word 160b  - \type\()_tbl
        .word 80b   - \type\()_tbl
        .word 40b   - \type\()_tbl
endjumptable
.endm

bidir_fn avg
bidir_fn w_avg
bidir_fn mask


.macro w_mask_fn type
function w_mask_\type\()_8bpc_neon, export=1
        ldr             w11, [sp] // sign
        clz             w8,  w4
        movrel          x9,  w_mask_\type\()_tbl
        sub             w8,  w8,  #25
        ldrsw           x8,  [x9,  x8,  lsl #2]
        add             x9,  x9,  x8
        mov             w10, #6903
        dup             v0.8h,   w10
.if \type == 444
        movi            v1.16b,  #64
.elseif \type == 422
        dup             v2.8b,   w11
        movi            v3.8b,   #129
        sub             v3.8b,   v3.8b,   v2.8b
.elseif \type == 420
        dup             v2.8h,   w11
        movi            v3.8h,   #1, lsl #8
        sub             v3.8h,   v3.8h,   v2.8h
.endif
        add             x12,  x0,  x1
        lsl             x1,   x1,  #1
        br              x9
40:
        AARCH64_VALID_JUMP_TARGET
4:
        ld1             {v4.8h,   v5.8h},   [x2],  #32  // tmp1 (four rows at once)
        ld1             {v6.8h,   v7.8h},   [x3],  #32  // tmp2 (four rows at once)
        subs            w5,  w5,  #4
        sub             v16.8h,  v6.8h,   v4.8h
        sub             v17.8h,  v7.8h,   v5.8h
        sabd            v18.8h,  v4.8h,   v6.8h
        sabd            v19.8h,  v5.8h,   v7.8h
        uqsub           v18.8h,  v0.8h,   v18.8h
        uqsub           v19.8h,  v0.8h,   v19.8h
        ushr            v18.8h,  v18.8h,  #8
        ushr            v19.8h,  v19.8h,  #8
        shl             v20.8h,  v18.8h,  #9
        shl             v21.8h,  v19.8h,  #9
        sqdmulh         v20.8h,  v20.8h,  v16.8h
        sqdmulh         v21.8h,  v21.8h,  v17.8h
        add             v20.8h,  v20.8h,  v4.8h
        add             v21.8h,  v21.8h,  v5.8h
        sqrshrun        v22.8b,  v20.8h,  #4
        sqrshrun        v23.8b,  v21.8h,  #4
.if \type == 444
        uzp1            v18.16b,  v18.16b, v19.16b      // Same as xtn, xtn2
        sub             v18.16b,  v1.16b,  v18.16b
        st1             {v18.16b}, [x6],  #16
.elseif \type == 422
        addp            v18.8h,   v18.8h,  v19.8h
        xtn             v18.8b,   v18.8h
        uhsub           v18.8b,   v3.8b,   v18.8b
        st1             {v18.8b},  [x6],  #8
.elseif \type == 420
        trn1            v24.2d,   v18.2d,  v19.2d
        trn2            v25.2d,   v18.2d,  v19.2d
        add             v24.8h,   v24.8h,  v25.8h
        addp            v18.8h,   v24.8h,  v24.8h
        sub             v18.4h,   v3.4h,   v18.4h
        rshrn           v18.8b,   v18.8h,  #2
        str             s18,         [x6],  #4
.endif
        st1             {v22.s}[0],  [x0],  x1
        st1             {v22.s}[1],  [x12], x1
        st1             {v23.s}[0],  [x0],  x1
        st1             {v23.s}[1],  [x12], x1
        b.gt            4b
        ret
80:
        AARCH64_VALID_JUMP_TARGET
8:
        ld1             {v4.8h,   v5.8h},   [x2],  #32
        ld1             {v6.8h,   v7.8h},   [x3],  #32
        subs            w5,  w5,  #2
        sub             v16.8h,  v6.8h,   v4.8h
        sub             v17.8h,  v7.8h,   v5.8h
        sabd            v18.8h,  v4.8h,   v6.8h
        sabd            v19.8h,  v5.8h,   v7.8h
        uqsub           v18.8h,  v0.8h,   v18.8h
        uqsub           v19.8h,  v0.8h,   v19.8h
        ushr            v18.8h,  v18.8h,  #8
        ushr            v19.8h,  v19.8h,  #8
        shl             v20.8h,  v18.8h,  #9
        shl             v21.8h,  v19.8h,  #9
        sqdmulh         v20.8h,  v20.8h,  v16.8h
        sqdmulh         v21.8h,  v21.8h,  v17.8h
        add             v20.8h,  v20.8h,  v4.8h
        add             v21.8h,  v21.8h,  v5.8h
        sqrshrun        v22.8b,  v20.8h,  #4
        sqrshrun        v23.8b,  v21.8h,  #4
.if \type == 444
        uzp1            v18.16b, v18.16b, v19.16b       // Same as xtn, xtn2
        sub             v18.16b, v1.16b,  v18.16b
        st1             {v18.16b}, [x6],  #16
.elseif \type == 422
        addp            v18.8h,  v18.8h,  v19.8h
        xtn             v18.8b,  v18.8h
        uhsub           v18.8b,  v3.8b,   v18.8b
        st1             {v18.8b},  [x6],  #8
.elseif \type == 420
        add             v18.8h,  v18.8h,  v19.8h
        addp            v18.8h,  v18.8h,  v18.8h
        sub             v18.4h,  v3.4h,   v18.4h
        rshrn           v18.8b,  v18.8h,  #2
        str             s18,       [x6],  #4
.endif
        st1             {v22.8b},  [x0],  x1
        st1             {v23.8b},  [x12], x1
        b.gt            8b
        ret
640:
320:
160:
        AARCH64_VALID_JUMP_TARGET
        mov             w11, w4
        sub             x1,  x1,  w4,  uxtw
.if \type == 444
        add             x10, x6,  x7
        lsl             x7,  x7,  #1
        sub             x7,  x7,  w4,  uxtw
.elseif \type == 422
        uxtw            x4,  w4
        add             x10, x6,  x7
        lsl             x7,  x7,  #1
        sub             x7,  x7,  x4,  lsr #1
.elseif \type == 420
        uxtw            x4,  w4
        sub             x7,  x7,  x4,  lsr #1
.endif
        add             x9,  x3,  w4,  uxtw #1
        add             x13, x2,  w4,  uxtw #1
161:
        mov             w8,  w4
16:
        ld1             {v4.8h,   v5.8h},   [x2],  #32
        ld1             {v6.8h,   v7.8h},   [x3],  #32
        ld1             {v16.8h,  v17.8h},  [x13], #32
        ld1             {v18.8h,  v19.8h},  [x9],  #32
        subs            w8,  w8,  #16
        sub             v6.8h,   v6.8h,   v4.8h
        sub             v7.8h,   v7.8h,   v5.8h
        sub             v18.8h,  v18.8h,  v16.8h
        sub             v19.8h,  v19.8h,  v17.8h
        abs             v20.8h,  v6.8h
        abs             v21.8h,  v7.8h
        abs             v22.8h,  v18.8h
        abs             v23.8h,  v19.8h
        uqsub           v20.8h,  v0.8h,   v20.8h
        uqsub           v21.8h,  v0.8h,   v21.8h
        uqsub           v22.8h,  v0.8h,   v22.8h
        uqsub           v23.8h,  v0.8h,   v23.8h
        ushr            v20.8h,  v20.8h,  #8
        ushr            v21.8h,  v21.8h,  #8
        ushr            v22.8h,  v22.8h,  #8
        ushr            v23.8h,  v23.8h,  #8
        shl             v24.8h,  v20.8h,  #9
        shl             v25.8h,  v21.8h,  #9
        shl             v26.8h,  v22.8h,  #9
        shl             v27.8h,  v23.8h,  #9
        sqdmulh         v24.8h,  v24.8h,  v6.8h
        sqdmulh         v25.8h,  v25.8h,  v7.8h
        sqdmulh         v26.8h,  v26.8h,  v18.8h
        sqdmulh         v27.8h,  v27.8h,  v19.8h
        add             v24.8h,  v24.8h,  v4.8h
        add             v25.8h,  v25.8h,  v5.8h
        add             v26.8h,  v26.8h,  v16.8h
        add             v27.8h,  v27.8h,  v17.8h
        sqrshrun        v24.8b,  v24.8h,  #4
        sqrshrun        v25.8b,  v25.8h,  #4
        sqrshrun        v26.8b,  v26.8h,  #4
        sqrshrun        v27.8b,  v27.8h,  #4
.if \type == 444
        uzp1            v20.16b, v20.16b, v21.16b       // Same as xtn, xtn2
        uzp1            v21.16b, v22.16b, v23.16b       // Ditto
        sub             v20.16b, v1.16b,  v20.16b
        sub             v21.16b, v1.16b,  v21.16b
        st1             {v20.16b}, [x6],  #16
        st1             {v21.16b}, [x10], #16
.elseif \type == 422
        addp            v20.8h,  v20.8h,  v21.8h
        addp            v21.8h,  v22.8h,  v23.8h
        xtn             v20.8b,  v20.8h
        xtn             v21.8b,  v21.8h
        uhsub           v20.8b,  v3.8b,   v20.8b
        uhsub           v21.8b,  v3.8b,   v21.8b
        st1             {v20.8b},  [x6],  #8
        st1             {v21.8b},  [x10], #8
.elseif \type == 420
        add             v20.8h,  v20.8h,  v22.8h
        add             v21.8h,  v21.8h,  v23.8h
        addp            v20.8h,  v20.8h,  v21.8h
        sub             v20.8h,  v3.8h,   v20.8h
        rshrn           v20.8b,  v20.8h,  #2
        st1             {v20.8b},  [x6],  #8
.endif
        st1             {v24.8b,  v25.8b},  [x0],  #16
        st1             {v26.8b,  v27.8b},  [x12], #16
        b.gt            16b
        subs            w5,  w5,  #2
        add             x2,  x2,  w4,  uxtw #1
        add             x3,  x3,  w4,  uxtw #1
        add             x13, x13, w4,  uxtw #1
        add             x9,  x9,  w4,  uxtw #1
        add             x6,  x6,  x7
.if \type == 444
        add             x10, x10, x7
.elseif \type == 422
        add             x10, x10, x7
.endif
        add             x0,  x0,  x1
        add             x12, x12, x1
        b.gt            161b
        ret
endfunc

jumptable w_mask_\type\()_tbl
        .word 640b  - w_mask_\type\()_tbl
        .word 320b  - w_mask_\type\()_tbl
        .word 160b  - w_mask_\type\()_tbl
        .word 80b   - w_mask_\type\()_tbl
        .word 40b   - w_mask_\type\()_tbl
endjumptable
.endm

w_mask_fn 444
w_mask_fn 422
w_mask_fn 420


function blend_8bpc_neon, export=1
        movrel          x6,  blend_tbl
        clz             w3,  w3
        sub             w3,  w3,  #25
        ldrsw           x3,  [x6,  x3,  lsl #2]
        add             x6,  x6,  x3
        movi            v4.16b,  #64
        add             x8,  x0,  x1
        lsl             x1,  x1,  #1
        br              x6
40:
        AARCH64_VALID_JUMP_TARGET
4:
        ld1             {v2.8b},  [x5],  #8
        ldr             d1,       [x2],  #8
        ldr             s0,       [x0]
        subs            w4,  w4,  #2
        ld1             {v0.s}[1],   [x8]
        sub             v3.8b,   v4.8b,   v2.8b
        umull           v5.8h,   v1.8b,   v2.8b
        umlal           v5.8h,   v0.8b,   v3.8b
        rshrn           v6.8b,   v5.8h,   #6
        st1             {v6.s}[0],   [x0],  x1
        st1             {v6.s}[1],   [x8],  x1
        b.gt            4b
        ret
80:
        AARCH64_VALID_JUMP_TARGET
8:
        ld1             {v2.16b},  [x5],  #16
        ld1             {v1.16b},  [x2],  #16
        ldr             d0,        [x0]
        ld1             {v0.d}[1], [x8]
        sub             v3.16b,  v4.16b,  v2.16b
        subs            w4,  w4,  #2
        umull           v5.8h,   v1.8b,   v2.8b
        umlal           v5.8h,   v0.8b,   v3.8b
        umull2          v6.8h,   v1.16b,  v2.16b
        umlal2          v6.8h,   v0.16b,  v3.16b
        rshrn           v7.8b,   v5.8h,   #6
        rshrn           v16.8b,  v6.8h,   #6
        st1             {v7.8b},   [x0],  x1
        st1             {v16.8b},  [x8],  x1
        b.gt            8b
        ret
160:
        AARCH64_VALID_JUMP_TARGET
16:
        ld1             {v1.16b,  v2.16b},  [x5],  #32
        ld1             {v5.16b,  v6.16b},  [x2],  #32
        ld1             {v0.16b},  [x0]
        subs            w4,  w4,  #2
        sub             v7.16b,  v4.16b,  v1.16b
        sub             v20.16b, v4.16b,  v2.16b
        ld1             {v3.16b},  [x8]
        umull           v16.8h,  v5.8b,   v1.8b
        umlal           v16.8h,  v0.8b,   v7.8b
        umull2          v17.8h,  v5.16b,  v1.16b
        umlal2          v17.8h,  v0.16b,  v7.16b
        umull           v21.8h,  v6.8b,   v2.8b
        umlal           v21.8h,  v3.8b,   v20.8b
        umull2          v22.8h,  v6.16b,  v2.16b
        umlal2          v22.8h,  v3.16b,  v20.16b
        rshrn           v18.8b,  v16.8h,  #6
        rshrn2          v18.16b, v17.8h,  #6
        rshrn           v19.8b,  v21.8h,  #6
        rshrn2          v19.16b, v22.8h,  #6
        st1             {v18.16b}, [x0],  x1
        st1             {v19.16b}, [x8],  x1
        b.gt            16b
        ret
320:
        AARCH64_VALID_JUMP_TARGET
32:
        ld1             {v0.16b,  v1.16b,  v2.16b,  v3.16b},  [x5],  #64
        ld1             {v16.16b, v17.16b, v18.16b, v19.16b}, [x2],  #64
        ld1             {v20.16b, v21.16b}, [x0]
        subs            w4,  w4,  #2
        ld1             {v22.16b, v23.16b}, [x8]
        sub             v5.16b,  v4.16b,  v0.16b
        sub             v6.16b,  v4.16b,  v1.16b
        sub             v30.16b, v4.16b,  v2.16b
        sub             v31.16b, v4.16b,  v3.16b
        umull           v24.8h,  v16.8b,  v0.8b
        umlal           v24.8h,  v20.8b,  v5.8b
        umull2          v26.8h,  v16.16b, v0.16b
        umlal2          v26.8h,  v20.16b, v5.16b
        umull           v28.8h,  v17.8b,  v1.8b
        umlal           v28.8h,  v21.8b,  v6.8b
        umull2          v7.8h,   v17.16b, v1.16b
        umlal2          v7.8h,   v21.16b, v6.16b
        umull           v27.8h,  v18.8b,  v2.8b
        umlal           v27.8h,  v22.8b,  v30.8b
        umull2          v1.8h,   v18.16b, v2.16b
        umlal2          v1.8h,   v22.16b, v30.16b
        umull           v29.8h,  v19.8b,  v3.8b
        umlal           v29.8h,  v23.8b,  v31.8b
        umull2          v21.8h,  v19.16b, v3.16b
        umlal2          v21.8h,  v23.16b, v31.16b
        rshrn           v24.8b,  v24.8h,  #6
        rshrn2          v24.16b, v26.8h,  #6
        rshrn           v25.8b,  v28.8h,  #6
        rshrn2          v25.16b, v7.8h,   #6
        rshrn           v27.8b,  v27.8h,  #6
        rshrn2          v27.16b, v1.8h,   #6
        rshrn           v28.8b,  v29.8h,  #6
        rshrn2          v28.16b, v21.8h,  #6
        st1             {v24.16b, v25.16b}, [x0],  x1
        st1             {v27.16b, v28.16b}, [x8],  x1
        b.gt            32b
        ret
640:
        AARCH64_VALID_JUMP_TARGET
        // Reset the destination stride, which is doubled in the prologue,
        // for the other cases where we write alternating rows with two
        // pointers.
        lsr             x1,  x1,  #1
64:
        ld1             {v0.16b,  v1.16b,  v2.16b,  v3.16b},  [x5],  #64
        ld1             {v16.16b, v17.16b, v18.16b, v19.16b}, [x2],  #64
        ld1             {v20.16b, v21.16b, v22.16b, v23.16b}, [x0]
        subs            w4,  w4,  #1
        sub             v5.16b,  v4.16b,  v0.16b
        sub             v6.16b,  v4.16b,  v1.16b
        sub             v30.16b, v4.16b,  v2.16b
        sub             v31.16b, v4.16b,  v3.16b
        umull           v24.8h,  v16.8b,  v0.8b
        umlal           v24.8h,  v20.8b,  v5.8b
        umull2          v26.8h,  v16.16b, v0.16b
        umlal2          v26.8h,  v20.16b, v5.16b
        umull           v28.8h,  v17.8b,  v1.8b
        umlal           v28.8h,  v21.8b,  v6.8b
        umull2          v7.8h,   v17.16b, v1.16b
        umlal2          v7.8h,   v21.16b, v6.16b
        umull           v27.8h,  v18.8b,  v2.8b
        umlal           v27.8h,  v22.8b,  v30.8b
        umull2          v1.8h,   v18.16b, v2.16b
        umlal2          v1.8h,   v22.16b, v30.16b
        umull           v29.8h,  v19.8b,  v3.8b
        umlal           v29.8h,  v23.8b,  v31.8b
        umull2          v21.8h,  v19.16b, v3.16b
        umlal2          v21.8h,  v23.16b, v31.16b
        rshrn           v24.8b,  v24.8h,  #6
        rshrn2          v24.16b, v26.8h,  #6
        rshrn           v25.8b,  v28.8h,  #6
        rshrn2          v25.16b, v7.8h,   #6
        rshrn           v26.8b,  v27.8h,  #6
        rshrn2          v26.16b, v1.8h,   #6
        rshrn           v27.8b,  v29.8h,  #6
        rshrn2          v27.16b, v21.8h,  #6
        st1             {v24.16b, v25.16b, v26.16b, v27.16b}, [x0],  x1
        b.gt            64b
        ret
endfunc

jumptable blend_tbl
        .word 640b - blend_tbl
        .word 320b - blend_tbl
        .word 160b - blend_tbl
        .word 80b  - blend_tbl
        .word 40b  - blend_tbl
endjumptable


// This has got the same signature as the put_8tap functions,
// and assumes that x8 is set to (clz(w)-24).
function put_neon, export=1
        movrel          x9,  put_tbl
        ldrsw           x8,  [x9, x8, lsl #2]
        add             x9,  x9,  x8
        br              x9

20:
        AARCH64_VALID_JUMP_TARGET
2:
        ldrh            w9, [x2]
        ldrh            w10, [x2, x3]
        add             x2, x2, x3, lsl #1
        subs            w5, w5, #2
        strh            w9, [x0]
        strh            w10, [x0, x1]
        add             x0, x0, x1, lsl #1
        b.gt            2b
        ret
40:
        AARCH64_VALID_JUMP_TARGET
4:
        ldr             w9, [x2]
        ldr             w10, [x2, x3]
        add             x2, x2, x3, lsl #1
        subs            w5, w5, #2
        str             w9, [x0]
        str             w10, [x0, x1]
        add             x0, x0, x1, lsl #1
        b.gt            4b
        ret
80:
        AARCH64_VALID_JUMP_TARGET
8:
        ldr             x9, [x2]
        ldr             x10, [x2, x3]
        add             x2, x2, x3, lsl #1
        subs            w5, w5, #2
        str             x9, [x0]
        str             x10, [x0, x1]
        add             x0, x0, x1, lsl #1
        b.gt            8b
        ret
160:
        AARCH64_VALID_JUMP_TARGET
16:
        ldr             q0, [x2]
        ldr             q1, [x2, x3]
        add             x2, x2, x3, lsl #1
        subs            w5, w5, #2
        str             q0, [x0]
        str             q1, [x0, x1]
        add             x0, x0, x1, lsl #1
        b.gt            16b
        ret
320:
        AARCH64_VALID_JUMP_TARGET
32:
        ldp             q0, q1, [x2]
        add             x2, x2, x3
        stp             q0, q1, [x0]
        add             x0, x0, x1
        ldp             q2, q3, [x2]
        add             x2, x2, x3
        stp             q2, q3, [x0]
        subs            w5, w5, #2
        add             x0, x0, x1
        b.gt            32b
        ret
640:
        AARCH64_VALID_JUMP_TARGET
64:
        ldp             q0, q1, [x2]
        stp             q0, q1, [x0]
        ldp             q2, q3, [x2, #32]
        add             x2, x2, x3
        stp             q2, q3, [x0, #32]
        subs            w5, w5, #1
        add             x0, x0, x1
        b.gt            64b
        ret
endfunc

jumptable put_tbl
        .word 640b  - put_tbl
        .word 320b  - put_tbl
        .word 160b  - put_tbl
        .word 80b   - put_tbl
        .word 40b   - put_tbl
        .word 20b   - put_tbl
endjumptable


// This has got the same signature as the prep_8tap functions,
// and assumes that x8 is set to (clz(w)-24), and x7 to w*2.
function prep_neon, export=1
        movrel          x9,  prep_tbl
        ldrsw           x8,  [x9, x8, lsl #2]
        movi            v24.16b, #16
        add             x9,  x9,  x8
        br              x9

40:
        AARCH64_VALID_JUMP_TARGET
4:
        ldr             s0, [x2]
        ldr             s2, [x2, x3]
        add             x2, x2, x3, lsl #1
        ldr             s1, [x2]
        ldr             s3, [x2, x3]
        add             x2, x2, x3, lsl #1
        mov             v0.s[1], v2.s[0]
        mov             v1.s[1], v3.s[0]
        ushll           v0.8h, v0.8b, #4
        ushll           v1.8h, v1.8b, #4
        subs            w5, w5, #4
        mov             v2.d[0], v0.d[1]
        mov             v3.d[0], v1.d[1]
        str             d0, [x0]
        str             d2, [x0, x1]
        add             x0, x0, x1, lsl #1
        str             d1, [x0]
        str             d3, [x0, x1]
        add             x0, x0, x1, lsl #1
        b.gt            4b
        ret
80:
        AARCH64_VALID_JUMP_TARGET
8:
        ldr             d0, [x2]
        ldr             d1, [x2, x3]
        add             x2, x2, x3, lsl #1
        ldr             d2, [x2]
        ldr             d3, [x2, x3]
        add             x2, x2, x3, lsl #1
        ushll           v0.8h, v0.8b, #4
        ushll           v1.8h, v1.8b, #4
        umull           v2.8h, v2.8b, v24.8b
        umull           v3.8h, v3.8b, v24.8b
        subs            w5, w5, #4
        str             q0, [x0]
        str             q1, [x0, x1]
        add             x0, x0, x1, lsl #1
        str             q2, [x0]
        str             q3, [x0, x1]
        add             x0, x0, x1, lsl #1
        b.gt            8b
        ret
160:
        AARCH64_VALID_JUMP_TARGET
16:
        ldr             q1, [x2]
        ldr             q3, [x2, x3]
        add             x2, x2, x3, lsl #1
        ushll           v0.8h, v1.8b, #4
        ushll2          v1.8h, v1.16b, #4
        ldr             q5, [x2]
        ldr             q7, [x2, x3]
        add             x2, x2, x3, lsl #1
        umull           v2.8h, v3.8b, v24.8b
        umull2          v3.8h, v3.16b, v24.16b
        ushll           v4.8h, v5.8b, #4
        ushll2          v5.8h, v5.16b, #4
        umull           v6.8h, v7.8b, v24.8b
        umull2          v7.8h, v7.16b, v24.16b
        subs            w5, w5, #4
        stp             q0, q1, [x0]
        add             x0, x0, x1
        stp             q2, q3, [x0]
        add             x0, x0, x1
        stp             q4, q5, [x0]
        add             x0, x0, x1
        stp             q6, q7, [x0]
        add             x0, x0, x1
        b.gt            16b
        ret
320:
        AARCH64_VALID_JUMP_TARGET
32:
        ldp             q4, q5, [x2]
        add             x2, x2, x3
        ldp             q6, q7, [x2]
        add             x2, x2, x3
        ushll           v0.8h, v4.8b, #4
        ushll2          v1.8h, v4.16b, #4
        umull           v2.8h, v5.8b, v24.8b
        umull2          v3.8h, v5.16b, v24.16b
        ushll           v4.8h, v6.8b, #4
        ushll2          v5.8h, v6.16b, #4
        umull           v6.8h, v7.8b, v24.8b
        umull2          v7.8h, v7.16b, v24.16b
        subs            w5, w5, #2
        stp             q0, q1, [x0]
        stp             q2, q3, [x0, #32]
        add             x0, x0, x1
        stp             q4, q5, [x0]
        stp             q6, q7, [x0, #32]
        add             x0, x0, x1
        b.gt            32b
        ret
640:
        AARCH64_VALID_JUMP_TARGET
64:
        ldp             q4, q5, [x2]
        ldp             q6, q7, [x2, #32]
        add             x2, x2, x3
        ushll           v0.8h, v4.8b, #4
        ushll2          v1.8h, v4.16b, #4
        umull           v2.8h, v5.8b, v24.8b
        umull2          v3.8h, v5.16b, v24.16b
        ushll           v4.8h, v6.8b, #4
        ushll2          v5.8h, v6.16b, #4
        umull           v6.8h, v7.8b, v24.8b
        umull2          v7.8h, v7.16b, v24.16b
        subs            w5, w5, #1
        stp             q0, q1, [x0]
        stp             q2, q3, [x0, #32]
        stp             q4, q5, [x0, #64]
        stp             q6, q7, [x0, #96]
        add             x0, x0, x1
        b.gt            64b
        ret
endfunc

jumptable prep_tbl
        .word 640b  - prep_tbl
        .word 320b  - prep_tbl
        .word 160b  - prep_tbl
        .word 80b   - prep_tbl
        .word 40b   - prep_tbl
endjumptable


.macro load_slice s0, s1, strd, wd, d0, d1, d2, d3, d4, d5, d6
        ld1             {\d0\wd}[0], [\s0], \strd
        ld1             {\d1\wd}[0], [\s1], \strd
.ifnb \d2
        ld1             {\d2\wd}[0], [\s0], \strd
        ld1             {\d3\wd}[0], [\s1], \strd
.endif
.ifnb \d4
        ld1             {\d4\wd}[0], [\s0], \strd
.endif
.ifnb \d5
        ld1             {\d5\wd}[0], [\s1], \strd
.endif
.ifnb \d6
        ld1             {\d6\wd}[0], [\s0], \strd
.endif
.endm
.macro load_reg s0, s1, strd, wd, d0, d1, d2, d3, d4, d5, d6
        ld1             {\d0\wd}, [\s0], \strd
        ld1             {\d1\wd}, [\s1], \strd
.ifnb \d2
        ld1             {\d2\wd}, [\s0], \strd
        ld1             {\d3\wd}, [\s1], \strd
.endif
.ifnb \d4
        ld1             {\d4\wd}, [\s0], \strd
.endif
.ifnb \d5
        ld1             {\d5\wd}, [\s1], \strd
.endif
.ifnb \d6
        ld1             {\d6\wd}, [\s0], \strd
.endif
.endm
.macro load_h s0, s1, strd, d0, d1, d2, d3, d4, d5, d6
        load_slice      \s0, \s1, \strd, .h, \d0, \d1, \d2, \d3, \d4, \d5, \d6
.endm
.macro load_s s0, s1, strd, d0, d1, d2, d3, d4, d5, d6
        load_slice      \s0, \s1, \strd, .s, \d0, \d1, \d2, \d3, \d4, \d5, \d6
.endm
.macro load_8b s0, s1, strd, d0, d1, d2, d3, d4, d5, d6
        load_reg        \s0, \s1, \strd, .8b, \d0, \d1, \d2, \d3, \d4, \d5, \d6
.endm
.macro load_16b s0, s1, strd, d0, d1, d2, d3, d4, d5, d6
        load_reg        \s0, \s1, \strd, .16b, \d0, \d1, \d2, \d3, \d4, \d5, \d6
.endm
.macro interleave_1 wd, r0, r1, r2, r3, r4
        trn1            \r0\wd, \r0\wd, \r1\wd
        trn1            \r1\wd, \r1\wd, \r2\wd
.ifnb \r3
        trn1            \r2\wd, \r2\wd, \r3\wd
        trn1            \r3\wd, \r3\wd, \r4\wd
.endif
.endm
.macro interleave_1_h r0, r1, r2, r3, r4
        interleave_1    .4h, \r0, \r1, \r2, \r3, \r4
.endm
.macro interleave_1_s r0, r1, r2, r3, r4
        interleave_1    .2s, \r0, \r1, \r2, \r3, \r4
.endm
.macro interleave_2 wd, r0, r1, r2, r3, r4, r5
        trn1            \r0\wd,  \r0\wd, \r2\wd
        trn1            \r1\wd,  \r1\wd, \r3\wd
        trn1            \r2\wd,  \r2\wd, \r4\wd
        trn1            \r3\wd,  \r3\wd, \r5\wd
.endm
.macro interleave_2_s r0, r1, r2, r3, r4, r5
        interleave_2    .2s, \r0, \r1, \r2, \r3, \r4, \r5
.endm
.macro uxtl_b r0, r1, r2, r3, r4, r5, r6
        uxtl            \r0\().8h, \r0\().8b
        uxtl            \r1\().8h, \r1\().8b
.ifnb \r2
        uxtl            \r2\().8h, \r2\().8b
        uxtl            \r3\().8h, \r3\().8b
.endif
.ifnb \r4
        uxtl            \r4\().8h, \r4\().8b
.endif
.ifnb \r5
        uxtl            \r5\().8h, \r5\().8b
.endif
.ifnb \r6
        uxtl            \r6\().8h, \r6\().8b
.endif
.endm
.macro mul_mla_4tap d, s0, s1, s2, s3, wd
        mul             \d\wd,  \s0\wd,  v0.h[0]
        mla             \d\wd,  \s1\wd,  v0.h[1]
        mla             \d\wd,  \s2\wd,  v0.h[2]
        mla             \d\wd,  \s3\wd,  v0.h[3]
.endm
// Interleaving the mul/mla chains actually hurts performance
// significantly on Cortex A53, thus keeping mul/mla tightly
// chained like this.
.macro mul_mla_6tap_0_4h d0, s0, s1, s2, s3, s4, s5, s6, s7
        mul             \d0\().4h, \s1\().4h, v0.h[1]
        mla             \d0\().4h, \s2\().4h, v0.h[2]
        mla             \d0\().4h, \s3\().4h, v0.h[3]
        mla             \d0\().4h, \s4\().4h, v0.h[4]
        mla             \d0\().4h, \s5\().4h, v0.h[5]
        mla             \d0\().4h, \s6\().4h, v0.h[6]
.endm
.macro mul_mla_6tap_0 d0, s0, s1, s2, s3, s4, s5, s6, s7
        mul             \d0\().8h, \s1\().8h, v0.h[1]
        mla             \d0\().8h, \s2\().8h, v0.h[2]
        mla             \d0\().8h, \s3\().8h, v0.h[3]
        mla             \d0\().8h, \s4\().8h, v0.h[4]
        mla             \d0\().8h, \s5\().8h, v0.h[5]
        mla             \d0\().8h, \s6\().8h, v0.h[6]
.endm
.macro mul_mla_6tap_1 d0, d1, s0, s1, s2, s3, s4, s5, s6, s7, s8
        mul             \d0\().8h, \s1\().8h, v0.h[1]
        mla             \d0\().8h, \s2\().8h, v0.h[2]
        mla             \d0\().8h, \s3\().8h, v0.h[3]
        mla             \d0\().8h, \s4\().8h, v0.h[4]
        mla             \d0\().8h, \s5\().8h, v0.h[5]
        mla             \d0\().8h, \s6\().8h, v0.h[6]
        mul             \d1\().8h, \s2\().8h, v0.h[1]
        mla             \d1\().8h, \s3\().8h, v0.h[2]
        mla             \d1\().8h, \s4\().8h, v0.h[3]
        mla             \d1\().8h, \s5\().8h, v0.h[4]
        mla             \d1\().8h, \s6\().8h, v0.h[5]
        mla             \d1\().8h, \s7\().8h, v0.h[6]
.endm
.macro mul_mla_6tap_2 d0, d1, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9
        mul             \d0\().8h, \s1\().8h, v0.h[1]
        mla             \d0\().8h, \s2\().8h, v0.h[2]
        mla             \d0\().8h, \s3\().8h, v0.h[3]
        mla             \d0\().8h, \s4\().8h, v0.h[4]
        mla             \d0\().8h, \s5\().8h, v0.h[5]
        mla             \d0\().8h, \s6\().8h, v0.h[6]
        mul             \d1\().8h, \s3\().8h, v0.h[1]
        mla             \d1\().8h, \s4\().8h, v0.h[2]
        mla             \d1\().8h, \s5\().8h, v0.h[3]
        mla             \d1\().8h, \s6\().8h, v0.h[4]
        mla             \d1\().8h, \s7\().8h, v0.h[5]
        mla             \d1\().8h, \s8\().8h, v0.h[6]
.endm
.macro mul_mla_8tap_0_4h d0, s0, s1, s2, s3, s4, s5, s6, s7
        mul             \d0\().4h, \s0\().4h, v0.h[0]
        mla             \d0\().4h, \s1\().4h, v0.h[1]
        mla             \d0\().4h, \s2\().4h, v0.h[2]
        mla             \d0\().4h, \s3\().4h, v0.h[3]
        mla             \d0\().4h, \s4\().4h, v0.h[4]
        mla             \d0\().4h, \s5\().4h, v0.h[5]
        mla             \d0\().4h, \s6\().4h, v0.h[6]
        mla             \d0\().4h, \s7\().4h, v0.h[7]
.endm
.macro mul_mla_8tap_0 d0, s0, s1, s2, s3, s4, s5, s6, s7
        mul             \d0\().8h, \s0\().8h, v0.h[0]
        mla             \d0\().8h, \s1\().8h, v0.h[1]
        mla             \d0\().8h, \s2\().8h, v0.h[2]
        mla             \d0\().8h, \s3\().8h, v0.h[3]
        mla             \d0\().8h, \s4\().8h, v0.h[4]
        mla             \d0\().8h, \s5\().8h, v0.h[5]
        mla             \d0\().8h, \s6\().8h, v0.h[6]
        mla             \d0\().8h, \s7\().8h, v0.h[7]
.endm
.macro mul_mla_8tap_1 d0, d1, s0, s1, s2, s3, s4, s5, s6, s7, s8
        mul             \d0\().8h, \s0\().8h, v0.h[0]
        mla             \d0\().8h, \s1\().8h, v0.h[1]
        mla             \d0\().8h, \s2\().8h, v0.h[2]
        mla             \d0\().8h, \s3\().8h, v0.h[3]
        mla             \d0\().8h, \s4\().8h, v0.h[4]
        mla             \d0\().8h, \s5\().8h, v0.h[5]
        mla             \d0\().8h, \s6\().8h, v0.h[6]
        mla             \d0\().8h, \s7\().8h, v0.h[7]
        mul             \d1\().8h, \s1\().8h, v0.h[0]
        mla             \d1\().8h, \s2\().8h, v0.h[1]
        mla             \d1\().8h, \s3\().8h, v0.h[2]
        mla             \d1\().8h, \s4\().8h, v0.h[3]
        mla             \d1\().8h, \s5\().8h, v0.h[4]
        mla             \d1\().8h, \s6\().8h, v0.h[5]
        mla             \d1\().8h, \s7\().8h, v0.h[6]
        mla             \d1\().8h, \s8\().8h, v0.h[7]
.endm
.macro mul_mla_8tap_2 d0, d1, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9
        mul             \d0\().8h, \s0\().8h, v0.h[0]
        mla             \d0\().8h, \s1\().8h, v0.h[1]
        mla             \d0\().8h, \s2\().8h, v0.h[2]
        mla             \d0\().8h, \s3\().8h, v0.h[3]
        mla             \d0\().8h, \s4\().8h, v0.h[4]
        mla             \d0\().8h, \s5\().8h, v0.h[5]
        mla             \d0\().8h, \s6\().8h, v0.h[6]
        mla             \d0\().8h, \s7\().8h, v0.h[7]
        mul             \d1\().8h, \s2\().8h, v0.h[0]
        mla             \d1\().8h, \s3\().8h, v0.h[1]
        mla             \d1\().8h, \s4\().8h, v0.h[2]
        mla             \d1\().8h, \s5\().8h, v0.h[3]
        mla             \d1\().8h, \s6\().8h, v0.h[4]
        mla             \d1\().8h, \s7\().8h, v0.h[5]
        mla             \d1\().8h, \s8\().8h, v0.h[6]
        mla             \d1\().8h, \s9\().8h, v0.h[7]
.endm
.macro sqrshrun_b shift, r0, r1, r2, r3
        sqrshrun        \r0\().8b, \r0\().8h,  #\shift
.ifnb \r1
        sqrshrun        \r1\().8b, \r1\().8h,  #\shift
.endif
.ifnb \r2
        sqrshrun        \r2\().8b, \r2\().8h,  #\shift
        sqrshrun        \r3\().8b, \r3\().8h,  #\shift
.endif
.endm
.macro srshr_h shift, r0, r1, r2, r3
        srshr           \r0\().8h, \r0\().8h,  #\shift
.ifnb \r1
        srshr           \r1\().8h, \r1\().8h,  #\shift
.endif
.ifnb \r2
        srshr           \r2\().8h, \r2\().8h,  #\shift
        srshr           \r3\().8h, \r3\().8h,  #\shift
.endif
.endm
.macro st_h strd, reg, lanes
        st1             {\reg\().h}[0], [x0], \strd
        st1             {\reg\().h}[1], [x8], \strd
.if \lanes > 2
        st1             {\reg\().h}[2], [x0], \strd
        st1             {\reg\().h}[3], [x8], \strd
.endif
.endm
.macro st_s strd, r0, r1
        st1             {\r0\().s}[0], [x0], \strd
        st1             {\r0\().s}[1], [x8], \strd
.ifnb \r1
        st1             {\r1\().s}[0], [x0], \strd
        st1             {\r1\().s}[1], [x8], \strd
.endif
.endm
.macro st_d strd, r0, r1
        st1             {\r0\().8b},   [x0], \strd
        st1             {\r0\().d}[1], [x8], \strd
.ifnb \r1
        st1             {\r1\().8b},   [x0], \strd
        st1             {\r1\().d}[1], [x8], \strd
.endif
.endm
.macro shift_store_4 type, strd, r0, r1
.ifc \type, put
        sqrshrun_b      6,     \r0, \r1
        st_s            \strd, \r0, \r1
.else
        srshr_h         2,     \r0, \r1
        st_d            \strd, \r0, \r1
.endif
.endm
.macro st_reg strd, wd, r0, r1, r2, r3, r4, r5, r6, r7
        st1             {\r0\wd}, [x0], \strd
        st1             {\r1\wd}, [x8], \strd
.ifnb \r2
        st1             {\r2\wd}, [x0], \strd
        st1             {\r3\wd}, [x8], \strd
.endif
.ifnb \r4
        st1             {\r4\wd}, [x0], \strd
        st1             {\r5\wd}, [x8], \strd
        st1             {\r6\wd}, [x0], \strd
        st1             {\r7\wd}, [x8], \strd
.endif
.endm
.macro st_8b strd, r0, r1, r2, r3, r4, r5, r6, r7
        st_reg          \strd, .8b,  \r0, \r1, \r2, \r3, \r4, \r5, \r6, \r7
.endm
.macro st_16b strd, r0, r1, r2, r3, r4, r5, r6, r7
        st_reg          \strd, .16b, \r0, \r1, \r2, \r3, \r4, \r5, \r6, \r7
.endm
.macro shift_store_8 type, strd, r0, r1, r2, r3
.ifc \type, put
        sqrshrun_b      6,     \r0, \r1, \r2, \r3
        st_8b           \strd, \r0, \r1, \r2, \r3
.else
        srshr_h         2,     \r0, \r1, \r2, \r3
        st_16b          \strd, \r0, \r1, \r2, \r3
.endif
.endm
.macro shift_store_16 type, strd, r0, r1, r2, r3
.ifc \type, put
        sqrshrun        \r0\().8b,  \r0\().8h, #6
        sqrshrun2       \r0\().16b, \r1\().8h, #6
        sqrshrun        \r2\().8b,  \r2\().8h, #6
        sqrshrun2       \r2\().16b, \r3\().8h, #6
        st_16b          \strd, \r0, \r2
.else
        srshr_h         2,     \r0, \r1, \r2, \r3
        st1             {\r0\().8h, \r1\().8h}, [x0], \strd
        st1             {\r2\().8h, \r3\().8h}, [x8], \strd
.endif
.endm

.macro make_8tap_fn op, name, type, taps
function \op\()_8tap_\name\()_8bpc_neon, export=1
        mov             x8,  \type
        b               \op\()_\taps\()_neon
endfunc
.endm

// No spaces in these expressions, due to gas-preprocessor.
#define REGULAR ((0*15<<7)|3*15)
#define SMOOTH  ((1*15<<7)|4*15)
#define SHARP   ((2*15<<7)|3*15)

.macro filter_fn type, dst, d_strd, src, s_strd, w, h, mx, xmx, my, xmy, ds2, sr2, shift_hv, taps
function \type\()_\taps\()_neon
        mov             w10,  #0x4081  // (1 << 14) | (1 << 7) | (1 << 0)
        mul             \mx,  \mx, w10
        mul             \my,  \my, w10
        add             \mx,  \mx, w8 // mx, 8tap_h, 4tap_h
        add             \my,  \my, w8 // my, 8tap_v, 4tap_v
.ifc \type, prep
        lsl             \d_strd, \d_strd, #1
.endif

        clz             w8,  \w
        tst             \mx, #(0x7f << 14)
        sub             w8,  w8,  #25
        movrel          x10, X(mc_subpel_filters), -8
        b.ne            L(\type\()_\taps\()_h)
        tst             \my, #(0x7f << 14)
        b.ne            L(\type\()_\taps\()_v)
        b               \type\()_neon

L(\type\()_\taps\()_h):
        cmp             \w,  #4
        ubfx            w9,  \mx, #7, #7
        and             \mx, \mx, #0x7f
        b.le            4f
        mov             \mx,  w9
4:
        tst             \my,  #(0x7f << 14)
        add             \xmx, x10, \mx, uxtw #3
        b.ne            L(\type\()_\taps\()_hv)

        movrel          x9,  \type\()_\taps\()_h_tbl
        ldrsw           x8,  [x9, x8, lsl #2]
        add             x9,  x9,  x8
        br              x9

20:     // 2xN h
        AARCH64_VALID_JUMP_TARGET
.ifc \type, put
        ldur            s0,  [\xmx, #2]
        sub             \src,  \src,  #1
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \d_strd,  \d_strd,  #1
        lsl             \s_strd,  \s_strd,  #1
        sxtl            v0.8h,  v0.8b
2:
        ld1             {v4.8b},  [\src], \s_strd
        ld1             {v6.8b},  [\sr2], \s_strd
        uxtl            v4.8h,  v4.8b
        uxtl            v6.8h,  v6.8b
        ext             v5.16b, v4.16b, v4.16b, #2
        ext             v7.16b, v6.16b, v6.16b, #2
        subs            \h,  \h,  #2
        trn1            v3.2s,  v4.2s,  v6.2s
        trn2            v6.2s,  v4.2s,  v6.2s
        trn1            v4.2s,  v5.2s,  v7.2s
        trn2            v7.2s,  v5.2s,  v7.2s
        mul             v3.4h,  v3.4h,  v0.h[0]
        mla             v3.4h,  v4.4h,  v0.h[1]
        mla             v3.4h,  v6.4h,  v0.h[2]
        mla             v3.4h,  v7.4h,  v0.h[3]
        srshr           v3.4h,  v3.4h,  #2
        sqrshrun        v3.8b,  v3.8h,  #4
        st1             {v3.h}[0], [\dst], \d_strd
        st1             {v3.h}[1], [\ds2], \d_strd
        b.gt            2b
        ret
.endif

40:     // 4xN h
        AARCH64_VALID_JUMP_TARGET
        ldur            s0,  [\xmx, #2]
        sub             \src,  \src,  #1
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \d_strd,  \d_strd,  #1
        lsl             \s_strd,  \s_strd,  #1
        sxtl            v0.8h,  v0.8b
4:
        ld1             {v16.8b}, [\src], \s_strd
        ld1             {v20.8b}, [\sr2], \s_strd
        uxtl            v16.8h,  v16.8b
        uxtl            v20.8h,  v20.8b
        ext             v17.16b, v16.16b, v16.16b, #2
        ext             v18.16b, v16.16b, v16.16b, #4
        ext             v19.16b, v16.16b, v16.16b, #6
        ext             v21.16b, v20.16b, v20.16b, #2
        ext             v22.16b, v20.16b, v20.16b, #4
        ext             v23.16b, v20.16b, v20.16b, #6
        subs            \h,  \h,  #2
        mul             v16.4h,  v16.4h,  v0.h[0]
        mla             v16.4h,  v17.4h,  v0.h[1]
        mla             v16.4h,  v18.4h,  v0.h[2]
        mla             v16.4h,  v19.4h,  v0.h[3]
        mul             v20.4h,  v20.4h,  v0.h[0]
        mla             v20.4h,  v21.4h,  v0.h[1]
        mla             v20.4h,  v22.4h,  v0.h[2]
        mla             v20.4h,  v23.4h,  v0.h[3]
        srshr           v16.4h,  v16.4h,  #2
        srshr           v20.4h,  v20.4h,  #2
.ifc \type, put
        sqrshrun        v16.8b,  v16.8h,  #4
        sqrshrun        v20.8b,  v20.8h,  #4
        str             s16,  [\dst]
        str             s20,  [\ds2]
        add             \dst, \dst, \d_strd
        add             \ds2, \ds2, \d_strd
.else
        st1             {v16.4h}, [\dst], \d_strd
        st1             {v20.4h}, [\ds2], \d_strd
.endif
        b.gt            4b
        ret

80:     // 8xN h
        AARCH64_VALID_JUMP_TARGET
        ld1             {v0.8b}, [\xmx]
.ifc \taps, 6tap
        sub             \src,  \src,  #2
.else
        sub             \src,  \src,  #3
.endif
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \d_strd,  \d_strd,  #1
        lsl             \s_strd,  \s_strd,  #1
        sxtl            v0.8h, v0.8b
8:
        ld1             {v16.8b, v17.8b},  [\src], \s_strd
        ld1             {v20.8b, v21.8b},  [\sr2], \s_strd
        uxtl            v16.8h,  v16.8b
        uxtl            v17.8h,  v17.8b
        uxtl            v20.8h,  v20.8b
        uxtl            v21.8h,  v21.8b

.ifc \taps, 6tap
        mul             v18.8h,  v16.8h,  v0.h[1]
        mul             v22.8h,  v20.8h,  v0.h[1]
    .irpc i, 23456
        ext             v19.16b, v16.16b, v17.16b, #(2*\i-2)
        ext             v23.16b, v20.16b, v21.16b, #(2*\i-2)
        mla             v18.8h,  v19.8h,  v0.h[\i]
        mla             v22.8h,  v23.8h,  v0.h[\i]
    .endr
.else   // 8tap
        mul             v18.8h,  v16.8h,  v0.h[0]
        mul             v22.8h,  v20.8h,  v0.h[0]
    .irpc i, 1234567
        ext             v19.16b, v16.16b, v17.16b, #(2*\i)
        ext             v23.16b, v20.16b, v21.16b, #(2*\i)
        mla             v18.8h,  v19.8h,  v0.h[\i]
        mla             v22.8h,  v23.8h,  v0.h[\i]
    .endr
.endif
        subs            \h,  \h,  #2
        srshr           v18.8h,  v18.8h, #2
        srshr           v22.8h,  v22.8h, #2
.ifc \type, put
        sqrshrun        v18.8b,  v18.8h, #4
        sqrshrun        v22.8b,  v22.8h, #4
        st1             {v18.8b}, [\dst], \d_strd
        st1             {v22.8b}, [\ds2], \d_strd
.else
        st1             {v18.8h}, [\dst], \d_strd
        st1             {v22.8h}, [\ds2], \d_strd
.endif
        b.gt            8b
        ret
160:
320:
640:    // 16xN, 32xN, ... h
        AARCH64_VALID_JUMP_TARGET
        ld1             {v0.8b}, [\xmx]
.ifc \taps, 6tap
        sub             \src,  \src,  #2
.else
        sub             \src,  \src,  #3
.endif
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \s_strd,  \s_strd,  #1
        sxtl            v0.8h, v0.8b

        sub             \s_strd,  \s_strd,  \w, uxtw
        sub             \s_strd,  \s_strd,  #8
        lsl             \d_strd,  \d_strd,  #1
.ifc \type, put
        sub             \d_strd,  \d_strd,  \w, uxtw
.else
        sub             \d_strd,  \d_strd,  \w, uxtw #1
.endif
161:
        ld1             {v16.8b, v17.8b, v18.8b},  [\src], #24
        ld1             {v20.8b, v21.8b, v22.8b},  [\sr2], #24
        mov             \mx, \w
        uxtl            v16.8h,  v16.8b
        uxtl            v17.8h,  v17.8b
        uxtl            v18.8h,  v18.8b
        uxtl            v20.8h,  v20.8b
        uxtl            v21.8h,  v21.8b
        uxtl            v22.8h,  v22.8b

16:
.ifc \taps, 6tap
        mul             v24.8h,  v16.8h,  v0.h[1]
        mul             v25.8h,  v17.8h,  v0.h[1]
        mul             v26.8h,  v20.8h,  v0.h[1]
        mul             v27.8h,  v21.8h,  v0.h[1]
    .irpc i, 23456
        ext             v28.16b, v16.16b, v17.16b, #(2*\i-2)
        ext             v29.16b, v17.16b, v18.16b, #(2*\i-2)
        ext             v30.16b, v20.16b, v21.16b, #(2*\i-2)
        ext             v31.16b, v21.16b, v22.16b, #(2*\i-2)
        mla             v24.8h,  v28.8h,  v0.h[\i]
        mla             v25.8h,  v29.8h,  v0.h[\i]
        mla             v26.8h,  v30.8h,  v0.h[\i]
        mla             v27.8h,  v31.8h,  v0.h[\i]
    .endr
.else   // 8tap
        mul             v24.8h,  v16.8h,  v0.h[0]
        mul             v25.8h,  v17.8h,  v0.h[0]
        mul             v26.8h,  v20.8h,  v0.h[0]
        mul             v27.8h,  v21.8h,  v0.h[0]
    .irpc i, 1234567
        ext             v28.16b, v16.16b, v17.16b, #(2*\i)
        ext             v29.16b, v17.16b, v18.16b, #(2*\i)
        ext             v30.16b, v20.16b, v21.16b, #(2*\i)
        ext             v31.16b, v21.16b, v22.16b, #(2*\i)
        mla             v24.8h,  v28.8h,  v0.h[\i]
        mla             v25.8h,  v29.8h,  v0.h[\i]
        mla             v26.8h,  v30.8h,  v0.h[\i]
        mla             v27.8h,  v31.8h,  v0.h[\i]
    .endr
.endif
        srshr           v24.8h,  v24.8h, #2
        srshr           v25.8h,  v25.8h, #2
        srshr           v26.8h,  v26.8h, #2
        srshr           v27.8h,  v27.8h, #2
        subs            \mx, \mx, #16
.ifc \type, put
        sqrshrun        v24.8b,  v24.8h, #4
        sqrshrun2       v24.16b, v25.8h, #4
        sqrshrun        v26.8b,  v26.8h, #4
        sqrshrun2       v26.16b, v27.8h, #4
        st1             {v24.16b}, [\dst], #16
        st1             {v26.16b}, [\ds2], #16
.else
        st1             {v24.8h, v25.8h}, [\dst], #32
        st1             {v26.8h, v27.8h}, [\ds2], #32
.endif
        b.le            9f

        mov             v16.16b, v18.16b
        mov             v20.16b, v22.16b
        ld1             {v17.8b, v18.8b}, [\src], #16
        ld1             {v21.8b, v22.8b}, [\sr2], #16
        uxtl            v17.8h,  v17.8b
        uxtl            v18.8h,  v18.8b
        uxtl            v21.8h,  v21.8b
        uxtl            v22.8h,  v22.8b
        b               16b

9:
        add             \dst,  \dst,  \d_strd
        add             \ds2,  \ds2,  \d_strd
        add             \src,  \src,  \s_strd
        add             \sr2,  \sr2,  \s_strd

        subs            \h,  \h,  #2
        b.gt            161b
        ret
endfunc

jumptable \type\()_\taps\()_h_tbl
        .word 640b  - \type\()_\taps\()_h_tbl
        .word 320b  - \type\()_\taps\()_h_tbl
        .word 160b  - \type\()_\taps\()_h_tbl
        .word 80b   - \type\()_\taps\()_h_tbl
        .word 40b   - \type\()_\taps\()_h_tbl
        .word 20b   - \type\()_\taps\()_h_tbl
endjumptable

function L(\type\()_\taps\()_v)
        cmp             \h,  #4
        ubfx            w9,  \my, #7, #7
        and             \my, \my, #0x7f
        b.le            4f
        mov             \my, w9
4:
        add             \xmy, x10, \my, uxtw #3

        movrel          x9,  \type\()_\taps\()_v_tbl
        ldrsw           x8,  [x9, x8, lsl #2]
        add             x9,  x9,  x8
        br              x9

20:     // 2xN v
        AARCH64_VALID_JUMP_TARGET
.ifc \type, put
        b.gt            28f

        cmp             \h,  #2
        ldur            s0,  [\xmy, #2]
        sub             \src,  \src,  \s_strd
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \s_strd,  \s_strd,  #1
        lsl             \d_strd,  \d_strd,  #1
        sxtl            v0.8h, v0.8b

        // 2x2 v
        load_h          \src, \sr2, \s_strd, v1, v2, v3, v4, v5
        interleave_1_h  v1, v2, v3, v4, v5
        b.gt            24f
        uxtl_b          v1, v2, v3, v4
        mul_mla_4tap    v6, v1, v2, v3, v4, .4h
        sqrshrun_b      6,  v6
        st_h            \d_strd, v6, 2
        ret

24:     // 2x4 v
        load_h          \sr2, \src, \s_strd, v6, v7
        interleave_1_h  v5, v6, v7
        interleave_2_s  v1, v2, v3, v4, v5, v6
        uxtl_b          v1, v2, v3, v4
        mul_mla_4tap    v6, v1, v2, v3, v4, .8h
        sqrshrun_b      6,  v6
        st_h            \d_strd, v6, 4
        ret

28:     // 2x6, 2x8, 2x12, 2x16 v
        ld1             {v0.8b}, [\xmy]
        sub             \sr2,  \src,  \s_strd, lsl #1
        add             \ds2,  \dst,  \d_strd
        sub             \src,  \sr2,  \s_strd
        lsl             \d_strd,  \d_strd,  #1
        lsl             \s_strd,  \s_strd,  #1
        sxtl            v0.8h, v0.8b

        load_h          \src, \sr2, \s_strd, v1,  v2,  v3,  v4, v5, v6, v7
        interleave_1_h  v1,  v2,  v3,  v4,  v5
        interleave_1_h  v5,  v6,  v7
        interleave_2_s  v1,  v2,  v3,  v4,  v5,  v6
        uxtl_b          v1,  v2,  v3,  v4
216:
        subs            \h,  \h,  #4
        load_h          \sr2, \src, \s_strd, v16, v17, v18, v19
        interleave_1_h  v7,  v16, v17, v18, v19
        interleave_2_s  v5,  v6,  v7,  v16, v17, v18
        uxtl_b          v5,  v6,  v7,  v16
        mul_mla_\taps\()_0 v30, v1, v2, v3, v4, v5, v6, v7, v16
        sqrshrun_b      6,   v30
        st_h            \d_strd, v30, 4
        b.le            0f
        cmp             \h,  #2
        mov             v1.16b,  v5.16b
        mov             v2.16b,  v6.16b
        mov             v3.16b,  v7.16b
        mov             v4.16b,  v16.16b
        mov             v5.16b,  v17.16b
        mov             v6.16b,  v18.16b
        mov             v7.16b,  v19.16b
        b.eq            26f
        b               216b
26:
        load_h          \sr2, \src, \s_strd, v16, v17
        interleave_1_h  v7,  v16, v17
        uxtl_b          v5,  v6,  v7,  v16
        mul_mla_\taps\()_0_4h v30, v1, v2, v3, v4, v5, v6, v7, v16
        sqrshrun_b      6,   v30
        st_h            \d_strd, v30, 2
0:
        ret
.endif

40:
        AARCH64_VALID_JUMP_TARGET
        b.gt            480f

        // 4x2, 4x4 v
        cmp             \h,  #2
        ldur            s0,  [\xmy, #2]
        sub             \src, \src, \s_strd
        add             \ds2, \dst, \d_strd
        add             \sr2, \src, \s_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1
        sxtl            v0.8h, v0.8b

        load_s          \src, \sr2, \s_strd, v1, v2, v3, v4, v5
        interleave_1_s  v1, v2, v3, v4, v5
        uxtl_b          v1, v2, v3, v4
        mul_mla_4tap    v6, v1, v2, v3, v4, .8h
        shift_store_4   \type, \d_strd, v6
        b.le            0f
        load_s          \sr2, \src, \s_strd, v6, v7
        interleave_1_s  v5, v6, v7
        uxtl_b          v5, v6
        mul_mla_4tap    v7, v3, v4, v5, v6, .8h
        shift_store_4   \type, \d_strd, v7
0:
        ret

480:    // 4x6, 4x8, 4x12, 4x16 v
        ld1             {v0.8b}, [\xmy]
        sub             \sr2, \src, \s_strd, lsl #1
        add             \ds2, \dst, \d_strd
        sub             \src, \sr2, \s_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1
        sxtl            v0.8h, v0.8b

        load_s          \src, \sr2, \s_strd, v16, v17, v18, v19, v20, v21, v22
        interleave_1_s  v16, v17, v18
        interleave_1_s  v18, v19, v20, v21, v22
        uxtl_b          v16, v17
        uxtl_b          v18, v19, v20, v21

48:
        subs            \h,  \h,  #4
        load_s          \sr2, \src, \s_strd, v23, v24, v25, v26
        interleave_1_s  v22, v23, v24, v25, v26
        uxtl_b          v22, v23, v24, v25
        mul_mla_\taps\()_2 v1, v2, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25
        shift_store_4   \type, \d_strd, v1, v2
        b.le            0f
        load_s          \sr2,  \src, \s_strd, v27, v16
        subs            \h,  \h,  #2
        interleave_1_s  v26, v27, v16
        uxtl_b          v26, v27
        mul_mla_\taps\()_0 v1, v20, v21, v22, v23, v24, v25, v26, v27
        shift_store_4   \type, \d_strd, v1
        b.le            0f
        load_s          \sr2,  \src, \s_strd, v17, v18
        subs            \h,  \h,  #2
        interleave_1_s  v16, v17, v18
        uxtl_b          v16, v17
        mul_mla_\taps\()_0 v2, v22, v23, v24, v25, v26, v27, v16, v17
        shift_store_4   \type, \d_strd, v2
        b.le            0f
        subs            \h,  \h,  #4
        load_s          \sr2, \src, \s_strd, v19, v20, v21, v22
        interleave_1_s  v18, v19, v20, v21, v22
        uxtl_b          v18, v19, v20, v21
        mul_mla_\taps\()_2 v1, v2, v24, v25, v26, v27, v16, v17, v18, v19, v20, v21
        shift_store_4   \type, \d_strd, v1, v2
        b.gt            48b
0:
        ret

80:
        AARCH64_VALID_JUMP_TARGET
        b.gt            880f

        // 8x2, 8x4 v
        cmp             \h,  #2
        ldur            s0,  [\xmy, #2]
        sub             \src, \src, \s_strd
        add             \ds2, \dst, \d_strd
        add             \sr2, \src, \s_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1
        sxtl            v0.8h, v0.8b

        load_8b         \src, \sr2, \s_strd, v1, v2, v3, v4, v5
        uxtl_b          v1, v2, v3, v4, v5
        mul_mla_4tap    v6, v1, v2, v3, v4, .8h
        mul_mla_4tap    v7, v2, v3, v4, v5, .8h
        shift_store_8   \type, \d_strd, v6, v7
        b.le            0f
        load_8b         \sr2, \src, \s_strd, v6, v7
        uxtl_b          v6, v7
        mul_mla_4tap    v1, v3, v4, v5, v6, .8h
        mul_mla_4tap    v2, v4, v5, v6, v7, .8h
        shift_store_8   \type, \d_strd, v1, v2
0:
        ret

880:    // 8x6, 8x8, 8x16, 8x32 v
1680:   // 16x8, 16x16, ...
320:    // 32x8, 32x16, ...
640:
        AARCH64_VALID_JUMP_TARGET
        ld1             {v0.8b}, [\xmy]
        sub             \src, \src, \s_strd
        sub             \src, \src, \s_strd, lsl #1
        sxtl            v0.8h, v0.8b
        mov             \my,  \h
168:
        add             \ds2, \dst, \d_strd
        add             \sr2, \src, \s_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1

        load_8b         \src, \sr2, \s_strd, v16, v17, v18, v19, v20, v21, v22
        uxtl_b          v16, v17, v18, v19, v20, v21, v22

88:
        subs            \h,  \h,  #2
        load_8b         \sr2, \src, \s_strd, v23, v24
        uxtl_b          v23, v24
        mul_mla_\taps\()_1 v1, v2, v16, v17, v18, v19, v20, v21, v22, v23, v24
        shift_store_8   \type, \d_strd, v1, v2
        b.le            9f
        subs            \h,  \h,  #2
        load_8b         \sr2, \src, \s_strd, v25, v26
        uxtl_b          v25, v26
        mul_mla_\taps\()_1 v3, v4, v18, v19, v20, v21, v22, v23, v24, v25, v26
        shift_store_8   \type, \d_strd, v3, v4
        b.le            9f
        subs            \h,  \h,  #2
        load_8b         \sr2, \src, \s_strd, v27, v16
        uxtl_b          v27, v16
        mul_mla_\taps\()_1 v1, v2, v20, v21, v22, v23, v24, v25, v26, v27, v16
        shift_store_8   \type, \d_strd, v1, v2
        b.le            9f
        subs            \h,  \h,  #2
        load_8b         \sr2, \src, \s_strd, v17, v18
        uxtl_b          v17, v18
        mul_mla_\taps\()_1 v3, v4, v22, v23, v24, v25, v26, v27, v16, v17, v18
        shift_store_8   \type, \d_strd, v3, v4
        b.le            9f
        subs            \h,  \h,  #4
        load_8b         \sr2, \src, \s_strd, v19, v20, v21, v22
        uxtl_b          v19, v20, v21, v22
        mul_mla_\taps\()_1 v1, v2, v24, v25, v26, v27, v16, v17, v18, v19, v20
        mul_mla_\taps\()_1 v3, v4, v26, v27, v16, v17, v18, v19, v20, v21, v22
        shift_store_8   \type, \d_strd, v1, v2, v3, v4
        b.gt            88b
9:
        subs            \w,  \w,  #8
        b.le            0f
        asr             \s_strd, \s_strd, #1
        asr             \d_strd, \d_strd, #1
        msub            \src, \s_strd, \xmy, \src
        msub            \dst, \d_strd, \xmy, \dst
        sub             \src, \src, \s_strd, lsl #3
        mov             \h,  \my
        add             \src, \src, #8
.ifc \type, put
        add             \dst, \dst, #8
.else
        add             \dst, \dst, #16
.endif
        b               168b
0:
        ret

160:
        AARCH64_VALID_JUMP_TARGET
        b.gt            1680b

        // 16x2, 16x4 v
        ldur            s0,  [\xmy, #2]
        sub             \src, \src, \s_strd
        add             \ds2, \dst, \d_strd
        add             \sr2, \src, \s_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1
        sxtl            v0.8h, v0.8b

        cmp             \h,  #2
        load_16b        \src, \sr2, \s_strd, v1,  v2,  v3,  v4,  v5
        uxtl            v16.8h, v1.8b
        uxtl            v17.8h, v2.8b
        uxtl            v18.8h, v3.8b
        uxtl            v19.8h, v4.8b
        uxtl            v20.8h, v5.8b
        uxtl2           v23.8h, v1.16b
        uxtl2           v24.8h, v2.16b
        uxtl2           v25.8h, v3.16b
        uxtl2           v26.8h, v4.16b
        uxtl2           v27.8h, v5.16b
        mul_mla_4tap    v1,  v16, v17, v18, v19, .8h
        mul_mla_4tap    v16, v17, v18, v19, v20, .8h
        mul_mla_4tap    v2,  v23, v24, v25, v26, .8h
        mul_mla_4tap    v17, v24, v25, v26, v27, .8h
        shift_store_16  \type, \d_strd, v1, v2, v16, v17
        b.le            0f
        load_16b        \sr2, \src, \s_strd, v6,  v7
        uxtl            v21.8h, v6.8b
        uxtl            v22.8h, v7.8b
        uxtl2           v28.8h, v6.16b
        uxtl2           v29.8h, v7.16b
        mul_mla_4tap    v1,  v18, v19, v20, v21, .8h
        mul_mla_4tap    v3,  v19, v20, v21, v22, .8h
        mul_mla_4tap    v2,  v25, v26, v27, v28, .8h
        mul_mla_4tap    v4,  v26, v27, v28, v29, .8h
        shift_store_16  \type, \d_strd, v1, v2, v3, v4
0:
        ret
endfunc

jumptable \type\()_\taps\()_v_tbl
        .word 640b  - \type\()_\taps\()_v_tbl
        .word 320b  - \type\()_\taps\()_v_tbl
        .word 160b  - \type\()_\taps\()_v_tbl
        .word 80b   - \type\()_\taps\()_v_tbl
        .word 40b   - \type\()_\taps\()_v_tbl
        .word 20b   - \type\()_\taps\()_v_tbl
endjumptable

function L(\type\()_\taps\()_hv)
        cmp             \h,  #4
        ubfx            w9,  \my, #7, #7
        and             \my, \my, #0x7f
        b.le            4f
        mov             \my,  w9
4:
        add             \xmy,  x10, \my, uxtw #3

        movrel          x9,  \type\()_\taps\()_hv_tbl
        ldrsw           x8,  [x9, x8, lsl #2]
        add             x9,  x9,  x8
        br              x9

20:
        AARCH64_VALID_JUMP_TARGET
.ifc \type, put
        ldur            s0,  [\xmx, #2]
        b.gt            280f
        ldur            s1,  [\xmy, #2]

        // 2x2, 2x4 hv
        sub             \sr2, \src, #1
        sub             \src, \sr2, \s_strd
        add             \ds2, \dst, \d_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1
        sxtl            v0.8h,  v0.8b
        sxtl            v1.8h,  v1.8b
        mov             x15, x30

        ld1             {v28.8b}, [\src], \s_strd
        uxtl            v28.8h,  v28.8b
        ext             v29.16b, v28.16b, v28.16b, #2
        mul             v28.4h,  v28.4h,  v0.4h
        mul             v29.4h,  v29.4h,  v0.4h
        addp            v28.4h,  v28.4h,  v29.4h
        addp            v16.4h,  v28.4h,  v28.4h
        srshr           v16.4h,  v16.4h,  #2
        bl              L(\type\()_\taps\()_filter_2)

        trn1            v16.2s, v16.2s, v28.2s
        mov             v17.8b, v28.8b

2:
        bl              L(\type\()_\taps\()_filter_2)

        ext             v18.8b, v17.8b, v28.8b, #4
        smull           v2.4s,  v16.4h, v1.h[0]
        smlal           v2.4s,  v17.4h, v1.h[1]
        smlal           v2.4s,  v18.4h, v1.h[2]
        smlal           v2.4s,  v28.4h, v1.h[3]

        sqrshrn         v2.4h,  v2.4s,  #\shift_hv
        sqxtun          v2.8b,  v2.8h
        subs            \h,  \h,  #2
        st1             {v2.h}[0], [\dst], \d_strd
        st1             {v2.h}[1], [\ds2], \d_strd
        b.le            0f
        mov             v16.8b, v18.8b
        mov             v17.8b, v28.8b
        b               2b

280:    // 2x8, 2x16, 2x32 hv
        ld1             {v1.8b},  [\xmy]
        sub             \src, \src, #1
        sub             \sr2, \src, \s_strd, lsl #1
        sub             \src, \sr2, \s_strd
        add             \ds2, \dst, \d_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1
        sxtl            v0.8h,  v0.8b
        sxtl            v1.8h,  v1.8b
        mov             x15, x30

        ld1             {v28.8b}, [\src], \s_strd
        uxtl            v28.8h,  v28.8b
        ext             v29.16b, v28.16b, v28.16b, #2
        mul             v28.4h,  v28.4h,  v0.4h
        mul             v29.4h,  v29.4h,  v0.4h
        addp            v28.4h,  v28.4h,  v29.4h
        addp            v16.4h,  v28.4h,  v28.4h
        srshr           v16.4h,  v16.4h,  #2

        bl              L(\type\()_\taps\()_filter_2)
        trn1            v16.2s, v16.2s, v28.2s
        mov             v17.8b, v28.8b
        bl              L(\type\()_\taps\()_filter_2)
        ext             v18.8b, v17.8b, v28.8b, #4
        mov             v19.8b, v28.8b
        bl              L(\type\()_\taps\()_filter_2)
        ext             v20.8b, v19.8b, v28.8b, #4
        mov             v21.8b, v28.8b

28:
        bl              L(\type\()_\taps\()_filter_2)
        ext             v22.8b, v21.8b, v28.8b, #4
.ifc \taps, 6tap
        smull           v2.4s,  v17.4h, v1.h[1]
        smlal           v2.4s,  v18.4h, v1.h[2]
        smlal           v2.4s,  v19.4h, v1.h[3]
        smlal           v2.4s,  v20.4h, v1.h[4]
        smlal           v2.4s,  v21.4h, v1.h[5]
        smlal           v2.4s,  v22.4h, v1.h[6]
.else   // 8tap
        smull           v2.4s,  v16.4h, v1.h[0]
        smlal           v2.4s,  v17.4h, v1.h[1]
        smlal           v2.4s,  v18.4h, v1.h[2]
        smlal           v2.4s,  v19.4h, v1.h[3]
        smlal           v2.4s,  v20.4h, v1.h[4]
        smlal           v2.4s,  v21.4h, v1.h[5]
        smlal           v2.4s,  v22.4h, v1.h[6]
        smlal           v2.4s,  v28.4h, v1.h[7]
.endif

        sqrshrn         v2.4h,  v2.4s,  #\shift_hv
        sqxtun          v2.8b,  v2.8h
        subs            \h,  \h,  #2
        st1             {v2.h}[0], [\dst], \d_strd
        st1             {v2.h}[1], [\ds2], \d_strd
        b.le            0f
        mov             v16.8b, v18.8b
        mov             v17.8b, v19.8b
        mov             v18.8b, v20.8b
        mov             v19.8b, v21.8b
        mov             v20.8b, v22.8b
        mov             v21.8b, v28.8b
        b               28b

0:
        ret             x15

L(\type\()_\taps\()_filter_2):
        ld1             {v28.8b},  [\sr2], \s_strd
        ld1             {v30.8b},  [\src], \s_strd
        uxtl            v28.8h,  v28.8b
        uxtl            v30.8h,  v30.8b
        ext             v29.16b, v28.16b, v28.16b, #2
        ext             v31.16b, v30.16b, v30.16b, #2
        trn1            v27.2s,  v28.2s,  v30.2s
        trn2            v30.2s,  v28.2s,  v30.2s
        trn1            v28.2s,  v29.2s,  v31.2s
        trn2            v31.2s,  v29.2s,  v31.2s
        mul             v27.4h,  v27.4h,  v0.h[0]
        mla             v27.4h,  v28.4h,  v0.h[1]
        mla             v27.4h,  v30.4h,  v0.h[2]
        mla             v27.4h,  v31.4h,  v0.h[3]
        srshr           v28.4h,  v27.4h,  #2
        ret
.endif

40:
        AARCH64_VALID_JUMP_TARGET
        ldur            s0,  [\xmx, #2]
        b.gt            480f
        ldur            s1,  [\xmy, #2]
        sub             \sr2, \src, #1
        sub             \src, \sr2, \s_strd
        add             \ds2, \dst, \d_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1
        sxtl            v0.8h,  v0.8b
        sxtl            v1.8h,  v1.8b
        mov             x15, x30

        // 4x2, 4x4 hv
        ld1             {v26.8b}, [\src], \s_strd
        uxtl            v26.8h,  v26.8b
        ext             v28.16b, v26.16b, v26.16b, #2
        ext             v29.16b, v26.16b, v26.16b, #4
        ext             v30.16b, v26.16b, v26.16b, #6
        mul             v31.4h,  v26.4h,  v0.h[0]
        mla             v31.4h,  v28.4h,  v0.h[1]
        mla             v31.4h,  v29.4h,  v0.h[2]
        mla             v31.4h,  v30.4h,  v0.h[3]
        srshr           v16.4h,  v31.4h,  #2

        bl              L(\type\()_\taps\()_filter_4)
        mov             v17.8b, v28.8b
        mov             v18.8b, v29.8b

4:
        bl              L(\type\()_\taps\()_filter_4)
        // Interleaving the mul/mla chains actually hurts performance
        // significantly on Cortex A53, thus keeping mul/mla tightly
        // chained like this.
        smull           v2.4s,  v16.4h, v1.h[0]
        smlal           v2.4s,  v17.4h, v1.h[1]
        smlal           v2.4s,  v18.4h, v1.h[2]
        smlal           v2.4s,  v28.4h, v1.h[3]
        smull           v3.4s,  v17.4h, v1.h[0]
        smlal           v3.4s,  v18.4h, v1.h[1]
        smlal           v3.4s,  v28.4h, v1.h[2]
        smlal           v3.4s,  v29.4h, v1.h[3]
        sqrshrn         v2.4h,  v2.4s,  #\shift_hv
        sqrshrn         v3.4h,  v3.4s,  #\shift_hv
        subs            \h,  \h,  #2
.ifc \type, put
        sqxtun          v2.8b,  v2.8h
        sqxtun          v3.8b,  v3.8h
        str             s2,  [\dst]
        str             s3,  [\ds2]
        add             \dst, \dst, \d_strd
        add             \ds2, \ds2, \d_strd
.else
        st1             {v2.4h}, [\dst], \d_strd
        st1             {v3.4h}, [\ds2], \d_strd
.endif
        b.le            0f
        mov             v16.8b,  v18.8b
        mov             v17.8b,  v28.8b
        mov             v18.8b,  v29.8b
        b               4b

480:    // 4x8, 4x16, 4x32 hv
        ld1             {v1.8b},  [\xmy]
        sub             \src, \src, #1
.ifc \taps, 6tap
        sub             \sr2, \src, \s_strd
        sub             \src, \src, \s_strd, lsl #1
.else
        sub             \sr2, \src, \s_strd, lsl #1
        sub             \src, \sr2, \s_strd
.endif
        add             \ds2, \dst, \d_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1
        sxtl            v0.8h,  v0.8b
        sxtl            v1.8h,  v1.8b
        mov             x15, x30

        ld1             {v26.8b}, [\src], \s_strd
        uxtl            v26.8h,  v26.8b
        ext             v28.16b, v26.16b, v26.16b, #2
        ext             v29.16b, v26.16b, v26.16b, #4
        ext             v30.16b, v26.16b, v26.16b, #6
        mul             v31.4h,  v26.4h,  v0.h[0]
        mla             v31.4h,  v28.4h,  v0.h[1]
        mla             v31.4h,  v29.4h,  v0.h[2]
        mla             v31.4h,  v30.4h,  v0.h[3]
.ifc \taps, 6tap
        srshr           v18.4h,  v31.4h,  #2
.else
        srshr           v16.4h,  v31.4h,  #2

        bl              L(\type\()_\taps\()_filter_4)
        mov             v17.8b, v28.8b
        mov             v18.8b, v29.8b
.endif
        bl              L(\type\()_\taps\()_filter_4)
        mov             v19.8b, v28.8b
        mov             v20.8b, v29.8b
        bl              L(\type\()_\taps\()_filter_4)
        mov             v21.8b, v28.8b
        mov             v22.8b, v29.8b

48:
        bl              L(\type\()_\taps\()_filter_4)
.ifc \taps, 6tap
        smull           v2.4s,  v18.4h, v1.h[1]
        smlal           v2.4s,  v19.4h, v1.h[2]
        smlal           v2.4s,  v20.4h, v1.h[3]
        smlal           v2.4s,  v21.4h, v1.h[4]
        smlal           v2.4s,  v22.4h, v1.h[5]
        smlal           v2.4s,  v28.4h, v1.h[6]
        smull           v3.4s,  v19.4h, v1.h[1]
        smlal           v3.4s,  v20.4h, v1.h[2]
        smlal           v3.4s,  v21.4h, v1.h[3]
        smlal           v3.4s,  v22.4h, v1.h[4]
        smlal           v3.4s,  v28.4h, v1.h[5]
        smlal           v3.4s,  v29.4h, v1.h[6]
.else   // 8tap
        smull           v2.4s,  v16.4h, v1.h[0]
        smlal           v2.4s,  v17.4h, v1.h[1]
        smlal           v2.4s,  v18.4h, v1.h[2]
        smlal           v2.4s,  v19.4h, v1.h[3]
        smlal           v2.4s,  v20.4h, v1.h[4]
        smlal           v2.4s,  v21.4h, v1.h[5]
        smlal           v2.4s,  v22.4h, v1.h[6]
        smlal           v2.4s,  v28.4h, v1.h[7]
        smull           v3.4s,  v17.4h, v1.h[0]
        smlal           v3.4s,  v18.4h, v1.h[1]
        smlal           v3.4s,  v19.4h, v1.h[2]
        smlal           v3.4s,  v20.4h, v1.h[3]
        smlal           v3.4s,  v21.4h, v1.h[4]
        smlal           v3.4s,  v22.4h, v1.h[5]
        smlal           v3.4s,  v28.4h, v1.h[6]
        smlal           v3.4s,  v29.4h, v1.h[7]
.endif
        sqrshrn         v2.4h,  v2.4s,  #\shift_hv
        sqrshrn         v3.4h,  v3.4s,  #\shift_hv
        subs            \h,  \h,  #2
.ifc \type, put
        sqxtun          v2.8b,  v2.8h
        sqxtun          v3.8b,  v3.8h
        str             s2,  [\dst]
        str             s3,  [\ds2]
        add             \dst, \dst, \d_strd
        add             \ds2, \ds2, \d_strd
.else
        st1             {v2.4h}, [\dst], \d_strd
        st1             {v3.4h}, [\ds2], \d_strd
.endif
        b.le            0f
.ifc \taps, 8tap
        mov             v16.8b,  v18.8b
        mov             v17.8b,  v19.8b
.endif
        mov             v18.8b,  v20.8b
        mov             v19.8b,  v21.8b
        mov             v20.8b,  v22.8b
        mov             v21.8b,  v28.8b
        mov             v22.8b,  v29.8b
        b               48b
0:
        ret             x15

L(\type\()_\taps\()_filter_4):
        ld1             {v26.8b}, [\sr2], \s_strd
        ld1             {v27.8b}, [\src], \s_strd
        uxtl            v26.8h,  v26.8b
        uxtl            v27.8h,  v27.8b
        ext             v28.16b, v26.16b, v26.16b, #2
        ext             v29.16b, v26.16b, v26.16b, #4
        ext             v30.16b, v26.16b, v26.16b, #6
        mul             v31.4h,  v26.4h,  v0.h[0]
        mla             v31.4h,  v28.4h,  v0.h[1]
        mla             v31.4h,  v29.4h,  v0.h[2]
        mla             v31.4h,  v30.4h,  v0.h[3]
        ext             v28.16b, v27.16b, v27.16b, #2
        ext             v29.16b, v27.16b, v27.16b, #4
        ext             v30.16b, v27.16b, v27.16b, #6
        mul             v27.4h,  v27.4h,  v0.h[0]
        mla             v27.4h,  v28.4h,  v0.h[1]
        mla             v27.4h,  v29.4h,  v0.h[2]
        mla             v27.4h,  v30.4h,  v0.h[3]
        srshr           v28.4h,  v31.4h,  #2
        srshr           v29.4h,  v27.4h,  #2
        ret

80:
160:
320:
        AARCH64_VALID_JUMP_TARGET
        b.gt            880f
        ld1             {v0.8b},  [\xmx]
        ldur            s1,  [\xmy, #2]
.ifc \taps, 6tap
        sub             \src,  \src,  #2
.else
        sub             \src,  \src,  #3
.endif
        sub             \src,  \src,  \s_strd
        sxtl            v0.8h,  v0.8b
        sxtl            v1.8h,  v1.8b
        mov             x15, x30
        mov             \my,  \h

164:    // 8x2, 8x4, 16x2, 16x4, 32x2, 32x4 hv
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \d_strd, \d_strd, #1
        lsl             \s_strd, \s_strd, #1

        bl              L(\type\()_\taps\()_filter_8_first)
        bl              L(\type\()_\taps\()_filter_8)
        mov             v17.16b, v24.16b
        mov             v18.16b, v25.16b

8:
        smull           v2.4s,  v16.4h, v1.h[0]
        smull2          v3.4s,  v16.8h, v1.h[0]
        bl              L(\type\()_\taps\()_filter_8)
        smull           v4.4s,  v17.4h, v1.h[0]
        smull2          v5.4s,  v17.8h, v1.h[0]
        smlal           v2.4s,  v17.4h, v1.h[1]
        smlal2          v3.4s,  v17.8h, v1.h[1]
        smlal           v4.4s,  v18.4h, v1.h[1]
        smlal2          v5.4s,  v18.8h, v1.h[1]
        smlal           v2.4s,  v18.4h, v1.h[2]
        smlal2          v3.4s,  v18.8h, v1.h[2]
        smlal           v4.4s,  v24.4h, v1.h[2]
        smlal2          v5.4s,  v24.8h, v1.h[2]
        smlal           v2.4s,  v24.4h, v1.h[3]
        smlal2          v3.4s,  v24.8h, v1.h[3]
        smlal           v4.4s,  v25.4h, v1.h[3]
        smlal2          v5.4s,  v25.8h, v1.h[3]
        sqrshrn         v2.4h,  v2.4s,  #\shift_hv
        sqrshrn2        v2.8h,  v3.4s,  #\shift_hv
        sqrshrn         v4.4h,  v4.4s,  #\shift_hv
        sqrshrn2        v4.8h,  v5.4s,  #\shift_hv
        subs            \h,  \h,  #2
.ifc \type, put
        sqxtun          v2.8b,  v2.8h
        sqxtun          v4.8b,  v4.8h
        st1             {v2.8b}, [\dst], \d_strd
        st1             {v4.8b}, [\ds2], \d_strd
.else
        st1             {v2.8h}, [\dst], \d_strd
        st1             {v4.8h}, [\ds2], \d_strd
.endif
        b.le            9f
        mov             v16.16b, v18.16b
        mov             v17.16b, v24.16b
        mov             v18.16b, v25.16b
        b               8b
9:
        subs            \w,  \w,  #8
        b.le            0f
        asr             \s_strd,  \s_strd,  #1
        asr             \d_strd,  \d_strd,  #1
        msub            \src,  \s_strd,  \xmy,  \src
        msub            \dst,  \d_strd,  \xmy,  \dst
        sub             \src,  \src,  \s_strd,  lsl #2
        mov             \h,  \my
        add             \src,  \src,  #8
.ifc \type, put
        add             \dst,  \dst,  #8
.else
        add             \dst,  \dst,  #16
.endif
        b               164b

880:    // 8x8, 8x16, ..., 16x8, ..., 32x8, ... hv
640:
        AARCH64_VALID_JUMP_TARGET
        ld1             {v0.8b},  [\xmx]
        ld1             {v1.8b},  [\xmy]
.ifc \taps, 6tap
        sub             \src,  \src,  #2
.else
        sub             \src,  \src,  #3
        sub             \src,  \src,  \s_strd
.endif
        sub             \src,  \src,  \s_strd, lsl #1
        sxtl            v0.8h,  v0.8b
        sxtl            v1.8h,  v1.8b
        mov             x15, x30
        mov             \my,  \h

168:
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \d_strd, \d_strd, #1
        lsl             \s_strd, \s_strd, #1

        bl              L(\type\()_\taps\()_filter_8_first)
.ifc \taps, 6tap
        mov             v18.16b, v16.16b
.else
        bl              L(\type\()_\taps\()_filter_8)
        mov             v17.16b, v24.16b
        mov             v18.16b, v25.16b
.endif
        bl              L(\type\()_\taps\()_filter_8)
        mov             v19.16b, v24.16b
        mov             v20.16b, v25.16b
        bl              L(\type\()_\taps\()_filter_8)
        mov             v21.16b, v24.16b
        mov             v22.16b, v25.16b

88:
.ifc \taps, 6tap
        smull           v2.4s,  v18.4h, v1.h[1]
        smull2          v3.4s,  v18.8h, v1.h[1]
        bl              L(\type\()_\taps\()_filter_8)
        smull           v4.4s,  v19.4h, v1.h[1]
        smull2          v5.4s,  v19.8h, v1.h[1]
        smlal           v2.4s,  v19.4h, v1.h[2]
        smlal2          v3.4s,  v19.8h, v1.h[2]
        smlal           v4.4s,  v20.4h, v1.h[2]
        smlal2          v5.4s,  v20.8h, v1.h[2]
        smlal           v2.4s,  v20.4h, v1.h[3]
        smlal2          v3.4s,  v20.8h, v1.h[3]
        smlal           v4.4s,  v21.4h, v1.h[3]
        smlal2          v5.4s,  v21.8h, v1.h[3]
        smlal           v2.4s,  v21.4h, v1.h[4]
        smlal2          v3.4s,  v21.8h, v1.h[4]
        smlal           v4.4s,  v22.4h, v1.h[4]
        smlal2          v5.4s,  v22.8h, v1.h[4]
        smlal           v2.4s,  v22.4h, v1.h[5]
        smlal2          v3.4s,  v22.8h, v1.h[5]
        smlal           v4.4s,  v24.4h, v1.h[5]
        smlal2          v5.4s,  v24.8h, v1.h[5]
        smlal           v2.4s,  v24.4h, v1.h[6]
        smlal2          v3.4s,  v24.8h, v1.h[6]
        smlal           v4.4s,  v25.4h, v1.h[6]
        smlal2          v5.4s,  v25.8h, v1.h[6]
.else   // 8tap
        smull           v2.4s,  v16.4h, v1.h[0]
        smull2          v3.4s,  v16.8h, v1.h[0]
        bl              L(\type\()_\taps\()_filter_8)
        smull           v4.4s,  v17.4h, v1.h[0]
        smull2          v5.4s,  v17.8h, v1.h[0]
        smlal           v2.4s,  v17.4h, v1.h[1]
        smlal2          v3.4s,  v17.8h, v1.h[1]
        smlal           v4.4s,  v18.4h, v1.h[1]
        smlal2          v5.4s,  v18.8h, v1.h[1]
        smlal           v2.4s,  v18.4h, v1.h[2]
        smlal2          v3.4s,  v18.8h, v1.h[2]
        smlal           v4.4s,  v19.4h, v1.h[2]
        smlal2          v5.4s,  v19.8h, v1.h[2]
        smlal           v2.4s,  v19.4h, v1.h[3]
        smlal2          v3.4s,  v19.8h, v1.h[3]
        smlal           v4.4s,  v20.4h, v1.h[3]
        smlal2          v5.4s,  v20.8h, v1.h[3]
        smlal           v2.4s,  v20.4h, v1.h[4]
        smlal2          v3.4s,  v20.8h, v1.h[4]
        smlal           v4.4s,  v21.4h, v1.h[4]
        smlal2          v5.4s,  v21.8h, v1.h[4]
        smlal           v2.4s,  v21.4h, v1.h[5]
        smlal2          v3.4s,  v21.8h, v1.h[5]
        smlal           v4.4s,  v22.4h, v1.h[5]
        smlal2          v5.4s,  v22.8h, v1.h[5]
        smlal           v2.4s,  v22.4h, v1.h[6]
        smlal2          v3.4s,  v22.8h, v1.h[6]
        smlal           v4.4s,  v24.4h, v1.h[6]
        smlal2          v5.4s,  v24.8h, v1.h[6]
        smlal           v2.4s,  v24.4h, v1.h[7]
        smlal2          v3.4s,  v24.8h, v1.h[7]
        smlal           v4.4s,  v25.4h, v1.h[7]
        smlal2          v5.4s,  v25.8h, v1.h[7]
.endif
        sqrshrn         v2.4h,  v2.4s,  #\shift_hv
        sqrshrn2        v2.8h,  v3.4s,  #\shift_hv
        sqrshrn         v4.4h,  v4.4s,  #\shift_hv
        sqrshrn2        v4.8h,  v5.4s,  #\shift_hv
        subs            \h,  \h,  #2
.ifc \type, put
        sqxtun          v2.8b,  v2.8h
        sqxtun          v4.8b,  v4.8h
        st1             {v2.8b}, [\dst], \d_strd
        st1             {v4.8b}, [\ds2], \d_strd
.else
        st1             {v2.8h}, [\dst], \d_strd
        st1             {v4.8h}, [\ds2], \d_strd
.endif
        b.le            9f
.ifc \taps, 8tap
        mov             v16.16b, v18.16b
        mov             v17.16b, v19.16b
.endif
        mov             v18.16b, v20.16b
        mov             v19.16b, v21.16b
        mov             v20.16b, v22.16b
        mov             v21.16b, v24.16b
        mov             v22.16b, v25.16b
        b               88b
9:
        subs            \w,  \w,  #8
        b.le            0f
        asr             \s_strd,  \s_strd,  #1
        asr             \d_strd,  \d_strd,  #1
        msub            \src,  \s_strd,  \xmy,  \src
        msub            \dst,  \d_strd,  \xmy,  \dst
        sub             \src,  \src,  \s_strd,  lsl #3
        mov             \h,  \my
        add             \src,  \src,  #8
.ifc \type, put
        add             \dst,  \dst,  #8
.else
        add             \dst,  \dst,  #16
.endif
.ifc \taps, 6tap
        add             \src,  \src,  \s_strd,  lsl #1
.endif
        b               168b
0:
        ret             x15

L(\type\()_\taps\()_filter_8_first):
        ld1             {v28.8b, v29.8b},  [\src], \s_strd
        uxtl            v28.8h,  v28.8b
        uxtl            v29.8h,  v29.8b
.ifc \taps, 6tap
        mul             v16.8h,  v28.8h,  v0.h[1]
        ext             v25.16b, v28.16b, v29.16b, #(2*1)
        ext             v26.16b, v28.16b, v29.16b, #(2*2)
        ext             v27.16b, v28.16b, v29.16b, #(2*3)
        mla             v16.8h,  v25.8h,  v0.h[2]
        mla             v16.8h,  v26.8h,  v0.h[3]
        mla             v16.8h,  v27.8h,  v0.h[4]
        ext             v24.16b, v28.16b, v29.16b, #(2*4)
        ext             v25.16b, v28.16b, v29.16b, #(2*5)
        mla             v16.8h,  v24.8h,  v0.h[5]
        mla             v16.8h,  v25.8h,  v0.h[6]
.else   // 8tap
        mul             v16.8h,  v28.8h,  v0.h[0]
        ext             v24.16b, v28.16b, v29.16b, #(2*1)
        ext             v25.16b, v28.16b, v29.16b, #(2*2)
        ext             v26.16b, v28.16b, v29.16b, #(2*3)
        ext             v27.16b, v28.16b, v29.16b, #(2*4)
        mla             v16.8h,  v24.8h,  v0.h[1]
        mla             v16.8h,  v25.8h,  v0.h[2]
        mla             v16.8h,  v26.8h,  v0.h[3]
        mla             v16.8h,  v27.8h,  v0.h[4]
        ext             v24.16b, v28.16b, v29.16b, #(2*5)
        ext             v25.16b, v28.16b, v29.16b, #(2*6)
        ext             v26.16b, v28.16b, v29.16b, #(2*7)
        mla             v16.8h,  v24.8h,  v0.h[5]
        mla             v16.8h,  v25.8h,  v0.h[6]
        mla             v16.8h,  v26.8h,  v0.h[7]
.endif
        srshr           v16.8h,  v16.8h,  #2
        ret

L(\type\()_\taps\()_filter_8):
        ld1             {v28.8b, v29.8b},  [\sr2], \s_strd
        ld1             {v30.8b, v31.8b},  [\src], \s_strd
        uxtl            v28.8h,  v28.8b
        uxtl            v29.8h,  v29.8b
        uxtl            v30.8h,  v30.8b
        uxtl            v31.8h,  v31.8b
.ifc \taps, 6tap
        mul             v24.8h,  v28.8h,  v0.h[1]
        mul             v25.8h,  v30.8h,  v0.h[1]
    .irpc i, 23456
        ext             v26.16b, v28.16b, v29.16b, #(2*\i-2)
        ext             v27.16b, v30.16b, v31.16b, #(2*\i-2)
        mla             v24.8h,  v26.8h,  v0.h[\i]
        mla             v25.8h,  v27.8h,  v0.h[\i]
    .endr
.else   // 8tap
        mul             v24.8h,  v28.8h,  v0.h[0]
        mul             v25.8h,  v30.8h,  v0.h[0]
    .irpc i, 1234567
        ext             v26.16b, v28.16b, v29.16b, #(2*\i)
        ext             v27.16b, v30.16b, v31.16b, #(2*\i)
        mla             v24.8h,  v26.8h,  v0.h[\i]
        mla             v25.8h,  v27.8h,  v0.h[\i]
    .endr
.endif
        srshr           v24.8h,  v24.8h, #2
        srshr           v25.8h,  v25.8h, #2
        ret
endfunc

jumptable \type\()_\taps\()_hv_tbl
        .word 640b  - \type\()_\taps\()_hv_tbl
        .word 320b  - \type\()_\taps\()_hv_tbl
        .word 160b  - \type\()_\taps\()_hv_tbl
        .word 80b   - \type\()_\taps\()_hv_tbl
        .word 40b   - \type\()_\taps\()_hv_tbl
        .word 20b   - \type\()_\taps\()_hv_tbl
endjumptable
.endm


.macro filter_bilin_fn type, dst, d_strd, src, s_strd, w, h, mx, xmx, my, xmy, ds2, sr2, shift_hv
function \type\()_bilin_8bpc_neon, export=1
        dup             v1.16b, \mx
        dup             v3.16b, \my
        mov             w9,  #16
        sub             w8, w9, \mx
        sub             w9, w9, \my
        dup             v0.16b, w8
        dup             v2.16b, w9
.ifc \type, prep
        lsl             \d_strd, \d_strd, #1
.endif

        clz             w8,  \w
        sub             w8,  w8,  #25
        cbnz            \mx, L(\type\()_bilin_h)
        cbnz            \my, L(\type\()_bilin_v)
        b               \type\()_neon

L(\type\()_bilin_h):
        cbnz            \my, L(\type\()_bilin_hv)

        movrel          x9,  \type\()_bilin_h_tbl
        ldrsw           x8,  [x9, x8, lsl #2]
        add             x9,  x9,  x8
        br              x9

20:     // 2xN h
        AARCH64_VALID_JUMP_TARGET
.ifc \type, put
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \d_strd,  \d_strd,  #1
        lsl             \s_strd,  \s_strd,  #1
2:
        ld1r            {v4.4s},  [\src], \s_strd
        ld1r            {v6.4s},  [\sr2], \s_strd
        ext             v5.8b,  v4.8b,  v4.8b, #1
        ext             v7.8b,  v6.8b,  v6.8b, #1
        trn1            v4.4h,  v4.4h,  v6.4h
        trn1            v5.4h,  v5.4h,  v7.4h
        subs            \h,  \h,  #2
        umull           v4.8h,  v4.8b,  v0.8b
        umlal           v4.8h,  v5.8b,  v1.8b
        uqrshrn         v4.8b,  v4.8h,  #4
        st1             {v4.h}[0], [\dst], \d_strd
        st1             {v4.h}[1], [\ds2], \d_strd
        b.gt            2b
        ret
.endif

40:     // 4xN h
        AARCH64_VALID_JUMP_TARGET
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \d_strd,  \d_strd,  #1
        lsl             \s_strd,  \s_strd,  #1
4:
        ld1             {v4.8b}, [\src], \s_strd
        ld1             {v6.8b}, [\sr2], \s_strd
        ext             v5.8b,  v4.8b,  v4.8b, #1
        ext             v7.8b,  v6.8b,  v6.8b, #1
        trn1            v4.2s,  v4.2s,  v6.2s
        trn1            v5.2s,  v5.2s,  v7.2s
        subs            \h,  \h,  #2
        umull           v4.8h,  v4.8b,  v0.8b
        umlal           v4.8h,  v5.8b,  v1.8b
.ifc \type, put
        uqrshrn         v4.8b,  v4.8h,  #4
        st1             {v4.s}[0], [\dst], \d_strd
        st1             {v4.s}[1], [\ds2], \d_strd
.else
        st1             {v4.8b},   [\dst], \d_strd
        st1             {v4.d}[1], [\ds2], \d_strd
.endif
        b.gt            4b
        ret

80:     // 8xN h
        AARCH64_VALID_JUMP_TARGET
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \d_strd,  \d_strd,  #1
        lsl             \s_strd,  \s_strd,  #1
8:
        ld1             {v4.16b}, [\src], \s_strd
        ld1             {v6.16b}, [\sr2], \s_strd
        ext             v5.16b, v4.16b, v4.16b, #1
        ext             v7.16b, v6.16b, v6.16b, #1
        subs            \h,  \h,  #2
        umull           v4.8h,  v4.8b,  v0.8b
        umull           v6.8h,  v6.8b,  v0.8b
        umlal           v4.8h,  v5.8b,  v1.8b
        umlal           v6.8h,  v7.8b,  v1.8b
.ifc \type, put
        uqrshrn         v4.8b,  v4.8h,  #4
        uqrshrn         v6.8b,  v6.8h,  #4
        st1             {v4.8b}, [\dst], \d_strd
        st1             {v6.8b}, [\ds2], \d_strd
.else
        st1             {v4.8h}, [\dst], \d_strd
        st1             {v6.8h}, [\ds2], \d_strd
.endif
        b.gt            8b
        ret
160:
320:
640:    // 16xN, 32xN, ... h
        AARCH64_VALID_JUMP_TARGET
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \s_strd,  \s_strd,  #1

        sub             \s_strd,  \s_strd,  \w, uxtw
        sub             \s_strd,  \s_strd,  #8
        lsl             \d_strd,  \d_strd,  #1
.ifc \type, put
        sub             \d_strd,  \d_strd,  \w, uxtw
.else
        sub             \d_strd,  \d_strd,  \w, uxtw #1
.endif
161:
        ld1             {v16.d}[1],  [\src], #8
        ld1             {v20.d}[1],  [\sr2], #8
        mov             \mx, \w

16:
        ld1             {v18.16b},  [\src], #16
        ld1             {v22.16b},  [\sr2], #16
        ext             v17.16b, v16.16b, v18.16b, #8
        ext             v19.16b, v16.16b, v18.16b, #9
        ext             v21.16b, v20.16b, v22.16b, #8
        ext             v23.16b, v20.16b, v22.16b, #9
        umull           v16.8h,  v17.8b,  v0.8b
        umull2          v17.8h,  v17.16b, v0.16b
        umull           v20.8h,  v21.8b,  v0.8b
        umull2          v21.8h,  v21.16b, v0.16b
        umlal           v16.8h,  v19.8b,  v1.8b
        umlal2          v17.8h,  v19.16b, v1.16b
        umlal           v20.8h,  v23.8b,  v1.8b
        umlal2          v21.8h,  v23.16b, v1.16b
        subs            \mx, \mx, #16
.ifc \type, put
        uqrshrn         v16.8b,  v16.8h, #4
        uqrshrn2        v16.16b, v17.8h, #4
        uqrshrn         v20.8b,  v20.8h, #4
        uqrshrn2        v20.16b, v21.8h, #4
        st1             {v16.16b}, [\dst], #16
        st1             {v20.16b}, [\ds2], #16
.else
        st1             {v16.8h, v17.8h}, [\dst], #32
        st1             {v20.8h, v21.8h}, [\ds2], #32
.endif
        b.le            9f

        mov             v16.16b, v18.16b
        mov             v20.16b, v22.16b
        b               16b

9:
        add             \dst,  \dst,  \d_strd
        add             \ds2,  \ds2,  \d_strd
        add             \src,  \src,  \s_strd
        add             \sr2,  \sr2,  \s_strd

        subs            \h,  \h,  #2
        b.gt            161b
        ret
endfunc

jumptable \type\()_bilin_h_tbl
        .word 640b  - \type\()_bilin_h_tbl
        .word 320b  - \type\()_bilin_h_tbl
        .word 160b  - \type\()_bilin_h_tbl
        .word 80b   - \type\()_bilin_h_tbl
        .word 40b   - \type\()_bilin_h_tbl
        .word 20b   - \type\()_bilin_h_tbl
endjumptable


function L(\type\()_bilin_v)
        cmp             \h,  #4
        movrel          x9,  \type\()_bilin_v_tbl
        ldrsw           x8,  [x9, x8, lsl #2]
        add             x9,  x9,  x8
        br              x9

20:     // 2xN v
        AARCH64_VALID_JUMP_TARGET
.ifc \type, put
        cmp             \h,  #2
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \s_strd,  \s_strd,  #1
        lsl             \d_strd,  \d_strd,  #1

        // 2x2 v
        ld1r            {v16.8h}, [\src], \s_strd
        b.gt            24f
22:
        ld1r            {v17.8h}, [\sr2], \s_strd
        ld1r            {v18.8h}, [\src], \s_strd
        trn1            v16.4h, v16.4h, v17.4h
        trn1            v17.4h, v17.4h, v18.4h
        umull           v4.8h,  v16.8b,  v2.8b
        umlal           v4.8h,  v17.8b,  v3.8b
        uqrshrn         v4.8b,  v4.8h,  #4
        str             h4,        [\dst]
        st1             {v4.h}[1], [\ds2]
        ret
24:     // 2x4, 2x6, 2x8, ... v
        ld1r            {v17.8h}, [\sr2], \s_strd
        ld1r            {v18.8h}, [\src], \s_strd
        ld1r            {v19.8h}, [\sr2], \s_strd
        ld1r            {v20.8h}, [\src], \s_strd
        sub             \h,  \h,  #4
        trn1            v16.4h, v16.4h, v17.4h
        trn1            v17.4h, v17.4h, v18.4h
        trn1            v18.4h, v18.4h, v19.4h
        trn1            v19.4h, v19.4h, v20.4h
        trn1            v16.2s, v16.2s, v18.2s
        trn1            v17.2s, v17.2s, v19.2s
        umull           v4.8h,  v16.8b,  v2.8b
        umlal           v4.8h,  v17.8b,  v3.8b
        cmp             \h,  #2
        uqrshrn         v4.8b,  v4.8h,  #4
        st1             {v4.h}[0], [\dst], \d_strd
        st1             {v4.h}[1], [\ds2], \d_strd
        st1             {v4.h}[2], [\dst], \d_strd
        st1             {v4.h}[3], [\ds2], \d_strd
        b.lt            0f
        mov             v16.8b, v20.8b
        b.eq            22b
        b               24b
0:
        ret
.endif

40:     // 4xN v
        AARCH64_VALID_JUMP_TARGET
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \s_strd,  \s_strd,  #1
        lsl             \d_strd,  \d_strd,  #1
        ld1r            {v16.4s}, [\src], \s_strd
4:
        ld1r            {v17.4s}, [\sr2], \s_strd
        ld1r            {v18.4s}, [\src], \s_strd
        trn1            v16.2s, v16.2s, v17.2s
        trn1            v17.2s, v17.2s, v18.2s
        umull           v4.8h,  v16.8b,  v2.8b
        umlal           v4.8h,  v17.8b,  v3.8b
        subs            \h,  \h,  #2
.ifc \type, put
        uqrshrn         v4.8b,  v4.8h,  #4
        st1             {v4.s}[0], [\dst], \d_strd
        st1             {v4.s}[1], [\ds2], \d_strd
.else
        st1             {v4.8b},   [\dst], \d_strd
        st1             {v4.d}[1], [\ds2], \d_strd
.endif
        b.le            0f
        mov             v16.8b, v18.8b
        b               4b
0:
        ret

80:     // 8xN v
        AARCH64_VALID_JUMP_TARGET
        add             \ds2,  \dst,  \d_strd
        add             \sr2,  \src,  \s_strd
        lsl             \s_strd,  \s_strd,  #1
        lsl             \d_strd,  \d_strd,  #1
        ld1             {v16.8b}, [\src], \s_strd
8:
        ld1             {v17.8b}, [\sr2], \s_strd
        ld1             {v18.8b}, [\src], \s_strd
        umull           v4.8h,  v16.8b,  v2.8b
        umull           v5.8h,  v17.8b,  v2.8b
        umlal           v4.8h,  v17.8b,  v3.8b
        umlal           v5.8h,  v18.8b,  v3.8b
        subs            \h,  \h,  #2
.ifc \type, put
        uqrshrn         v4.8b,  v4.8h,  #4
        uqrshrn         v5.8b,  v5.8h,  #4
        st1             {v4.8b}, [\dst], \d_strd
        st1             {v5.8b}, [\ds2], \d_strd
.else
        st1             {v4.8h}, [\dst], \d_strd
        st1             {v5.8h}, [\ds2], \d_strd
.endif
        b.le            0f
        mov             v16.8b, v18.8b
        b               8b
0:
        ret

160:    // 16xN, 32xN, ...
320:
640:
        AARCH64_VALID_JUMP_TARGET
        mov             \my,  \h
1:
        add             \ds2, \dst, \d_strd
        add             \sr2, \src, \s_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1

        ld1             {v16.16b}, [\src], \s_strd
2:
        ld1             {v17.16b}, [\sr2], \s_strd
        ld1             {v18.16b}, [\src], \s_strd
        umull           v4.8h,  v16.8b,  v2.8b
        umull2          v5.8h,  v16.16b, v2.16b
        umull           v6.8h,  v17.8b,  v2.8b
        umull2          v7.8h,  v17.16b, v2.16b
        umlal           v4.8h,  v17.8b,  v3.8b
        umlal2          v5.8h,  v17.16b, v3.16b
        umlal           v6.8h,  v18.8b,  v3.8b
        umlal2          v7.8h,  v18.16b, v3.16b
        subs            \h,  \h,  #2
.ifc \type, put
        uqrshrn         v4.8b,  v4.8h,  #4
        uqrshrn2        v4.16b, v5.8h,  #4
        uqrshrn         v6.8b,  v6.8h,  #4
        uqrshrn2        v6.16b, v7.8h,  #4
        st1             {v4.16b}, [\dst], \d_strd
        st1             {v6.16b}, [\ds2], \d_strd
.else
        st1             {v4.8h, v5.8h}, [\dst], \d_strd
        st1             {v6.8h, v7.8h}, [\ds2], \d_strd
.endif
        b.le            9f
        mov             v16.16b, v18.16b
        b               2b
9:
        subs            \w,  \w,  #16
        b.le            0f
        asr             \s_strd, \s_strd, #1
        asr             \d_strd, \d_strd, #1
        msub            \src, \s_strd, \xmy, \src
        msub            \dst, \d_strd, \xmy, \dst
        sub             \src, \src, \s_strd, lsl #1
        mov             \h,  \my
        add             \src, \src, #16
.ifc \type, put
        add             \dst, \dst, #16
.else
        add             \dst, \dst, #32
.endif
        b               1b
0:
        ret
endfunc

jumptable \type\()_bilin_v_tbl
        .word 640b  - \type\()_bilin_v_tbl
        .word 320b  - \type\()_bilin_v_tbl
        .word 160b  - \type\()_bilin_v_tbl
        .word 80b   - \type\()_bilin_v_tbl
        .word 40b   - \type\()_bilin_v_tbl
        .word 20b   - \type\()_bilin_v_tbl
endjumptable

function L(\type\()_bilin_hv)
        uxtl            v2.8h, v2.8b
        uxtl            v3.8h, v3.8b
        movrel          x9,  \type\()_bilin_hv_tbl
        ldrsw           x8,  [x9, x8, lsl #2]
        add             x9,  x9,  x8
        br              x9

20:     // 2xN hv
        AARCH64_VALID_JUMP_TARGET
.ifc \type, put
        add             \sr2, \src, \s_strd
        add             \ds2, \dst, \d_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1

        ld1r            {v28.4s},  [\src], \s_strd
        ext             v29.8b, v28.8b, v28.8b, #1
        umull           v16.8h, v28.8b, v0.8b
        umlal           v16.8h, v29.8b, v1.8b

2:
        ld1r            {v28.4s},  [\sr2], \s_strd
        ld1r            {v30.4s},  [\src], \s_strd
        ext             v29.8b, v28.8b, v28.8b, #1
        ext             v31.8b, v30.8b, v30.8b, #1
        trn1            v28.4h, v28.4h, v30.4h
        trn1            v29.4h, v29.4h, v31.4h
        umull           v17.8h, v28.8b, v0.8b
        umlal           v17.8h, v29.8b, v1.8b

        trn1            v16.2s, v16.2s, v17.2s

        mul             v4.4h,  v16.4h, v2.4h
        mla             v4.4h,  v17.4h, v3.4h
        uqrshrn         v4.8b,  v4.8h,  #8
        subs            \h,  \h,  #2
        st1             {v4.h}[0], [\dst], \d_strd
        st1             {v4.h}[1], [\ds2], \d_strd
        b.le            0f
        trn2            v16.2s, v17.2s, v17.2s
        b               2b
0:
        ret
.endif

40:     // 4xN hv
        AARCH64_VALID_JUMP_TARGET
        add             \sr2, \src, \s_strd
        add             \ds2, \dst, \d_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1

        ld1             {v28.8b},  [\src], \s_strd
        ext             v29.8b, v28.8b, v28.8b, #1
        umull           v16.8h, v28.8b, v0.8b
        umlal           v16.8h, v29.8b, v1.8b

4:
        ld1             {v28.8b},  [\sr2], \s_strd
        ld1             {v30.8b},  [\src], \s_strd
        ext             v29.8b, v28.8b, v28.8b, #1
        ext             v31.8b, v30.8b, v30.8b, #1
        trn1            v28.2s, v28.2s, v30.2s
        trn1            v29.2s, v29.2s, v31.2s
        umull           v17.8h, v28.8b, v0.8b
        umlal           v17.8h, v29.8b, v1.8b

        trn1            v16.2d, v16.2d, v17.2d

        mul             v4.8h,  v16.8h, v2.8h
        mla             v4.8h,  v17.8h, v3.8h
        subs            \h,  \h,  #2
.ifc \type, put
        uqrshrn         v4.8b,  v4.8h,  #8
        st1             {v4.s}[0], [\dst], \d_strd
        st1             {v4.s}[1], [\ds2], \d_strd
.else
        urshr           v4.8h,  v4.8h,  #4
        st1             {v4.8b},   [\dst], \d_strd
        st1             {v4.d}[1], [\ds2], \d_strd
.endif
        b.le            0f
        trn2            v16.2d, v17.2d, v17.2d
        b               4b
0:
        ret

80:     // 8xN, 16xN, ... hv
160:
320:
640:
        AARCH64_VALID_JUMP_TARGET
        mov             \my,  \h

1:
        add             \sr2, \src, \s_strd
        add             \ds2, \dst, \d_strd
        lsl             \s_strd, \s_strd, #1
        lsl             \d_strd, \d_strd, #1

        ld1             {v28.16b},  [\src], \s_strd
        ext             v29.16b, v28.16b, v28.16b, #1
        umull           v16.8h, v28.8b, v0.8b
        umlal           v16.8h, v29.8b, v1.8b

2:
        ld1             {v28.16b},  [\sr2], \s_strd
        ld1             {v30.16b},  [\src], \s_strd
        ext             v29.16b, v28.16b, v28.16b, #1
        ext             v31.16b, v30.16b, v30.16b, #1
        umull           v17.8h, v28.8b, v0.8b
        umlal           v17.8h, v29.8b, v1.8b
        umull           v18.8h, v30.8b, v0.8b
        umlal           v18.8h, v31.8b, v1.8b

        mul             v4.8h,  v16.8h, v2.8h
        mla             v4.8h,  v17.8h, v3.8h
        mul             v5.8h,  v17.8h, v2.8h
        mla             v5.8h,  v18.8h, v3.8h
        subs            \h,  \h,  #2
.ifc \type, put
        uqrshrn         v4.8b,  v4.8h,  #8
        uqrshrn         v5.8b,  v5.8h,  #8
        st1             {v4.8b}, [\dst], \d_strd
        st1             {v5.8b}, [\ds2], \d_strd
.else
        urshr           v4.8h,  v4.8h,  #4
        urshr           v5.8h,  v5.8h,  #4
        st1             {v4.8h}, [\dst], \d_strd
        st1             {v5.8h}, [\ds2], \d_strd
.endif
        b.le            9f
        mov             v16.16b, v18.16b
        b               2b
9:
        subs            \w,  \w,  #8
        b.le            0f
        asr             \s_strd,  \s_strd,  #1
        asr             \d_strd,  \d_strd,  #1
        msub            \src,  \s_strd,  \xmy,  \src
        msub            \dst,  \d_strd,  \xmy,  \dst
        sub             \src,  \src,  \s_strd,  lsl #1
        mov             \h,  \my
        add             \src,  \src,  #8
.ifc \type, put
        add             \dst,  \dst,  #8
.else
        add             \dst,  \dst,  #16
.endif
        b               1b
0:
        ret
endfunc

jumptable \type\()_bilin_hv_tbl
        .word 640b  - \type\()_bilin_hv_tbl
        .word 320b  - \type\()_bilin_hv_tbl
        .word 160b  - \type\()_bilin_hv_tbl
        .word 80b   - \type\()_bilin_hv_tbl
        .word 40b   - \type\()_bilin_hv_tbl
        .word 20b   - \type\()_bilin_hv_tbl
endjumptable
.endm

make_8tap_fn    put,  sharp,   SHARP,   8tap
filter_fn       put,  x0, x1, x2, x3, w4, w5, w6, x6, w7, x7, x8, x9, 10, 8tap

make_8tap_fn    put,  regular, REGULAR, 6tap
make_8tap_fn    put,  smooth,  SMOOTH,  6tap
filter_fn       put,  x0, x1, x2, x3, w4, w5, w6, x6, w7, x7, x8, x9, 10, 6tap
filter_bilin_fn put,  x0, x1, x2, x3, w4, w5, w6, x6, w7, x7, x8, x9, 10

make_8tap_fn    prep, sharp,   SHARP,   8tap
filter_fn       prep, x0, x1, x2, x3, w4, w5, w6, x6, w7, x7, x8, x9, 6,  8tap

make_8tap_fn    prep, regular, REGULAR, 6tap
make_8tap_fn    prep, smooth,  SMOOTH,  6tap
filter_fn       prep, x0, x1, x2, x3, w4, w5, w6, x6, w7, x7, x8, x9, 6,  6tap
filter_bilin_fn prep, x0, x1, x2, x3, w4, w5, w6, x6, w7, x7, x8, x9, 6


.macro load_filter_row dst, src, inc
        asr             w13, \src, #10
        add             \src, \src, \inc
        ldr             \dst, [x11, w13, sxtw #3]
.endm

function warp_filter_horz_neon
        add             w12, w5,  #512

        ld1             {v16.8b, v17.8b}, [x2], x3

        load_filter_row d0, w12, w7
        load_filter_row d1, w12, w7
        load_filter_row d2, w12, w7
        load_filter_row d3, w12, w7
        load_filter_row d4, w12, w7
        load_filter_row d5, w12, w7
        load_filter_row d6, w12, w7
        // subtract by 128 to allow using smull
        eor             v16.8b,  v16.8b,  v22.8b
        eor             v17.8b,  v17.8b,  v22.8b
        load_filter_row d7, w12, w7

        ext             v18.8b,  v16.8b,  v17.8b,  #1
        ext             v19.8b,  v16.8b,  v17.8b,  #2
        smull           v0.8h,   v0.8b,   v16.8b
        smull           v1.8h,   v1.8b,   v18.8b
        ext             v18.8b,  v16.8b,  v17.8b,  #3
        ext             v20.8b,  v16.8b,  v17.8b,  #4
        smull           v2.8h,   v2.8b,   v19.8b
        smull           v3.8h,   v3.8b,   v18.8b
        ext             v18.8b,  v16.8b,  v17.8b,  #5
        ext             v19.8b,  v16.8b,  v17.8b,  #6
        smull           v4.8h,   v4.8b,   v20.8b
        smull           v5.8h,   v5.8b,   v18.8b
        ext             v18.8b,  v16.8b,  v17.8b,  #7
        smull           v6.8h,   v6.8b,   v19.8b
        smull           v7.8h,   v7.8b,   v18.8b

        addp            v0.8h,   v0.8h,   v1.8h
        addp            v2.8h,   v2.8h,   v3.8h
        addp            v4.8h,   v4.8h,   v5.8h
        addp            v6.8h,   v6.8h,   v7.8h

        addp            v0.8h,   v0.8h,   v2.8h
        addp            v4.8h,   v4.8h,   v6.8h

        addp            v0.8h,   v0.8h,   v4.8h

        add             w5,  w5,  w8

        ret
endfunc

// void dav2d_warp_affine_8x8_8bpc_neon(
//         pixel *dst, const ptrdiff_t dst_stride,
//         const pixel *src, const ptrdiff_t src_stride,
//         const int16_t *const abcd, int mx, int my)
.macro warp t, shift
function warp_affine_8x8\t\()_8bpc_neon, export=1
        ldr             x4,  [x4]
        sbfx            x7,  x4, #0,  #16
        sbfx            x8,  x4, #16, #16
        sbfx            x9,  x4, #32, #16
        sbfx            x4,  x4, #48, #16
        mov             w10, #8
        sub             x2,  x2,  x3, lsl #1
        sub             x2,  x2,  x3
        sub             x2,  x2,  #3
        movrel          x11, X(mc_warp_filter), 3*64*8
        mov             x15, x30
.ifnb \t
        lsl             x1,  x1,  #1
.endif

        movi            v22.8b,  #128
.ifb \t
        movi            v23.8h,  #128
.else
        movi            v23.8h,  #8, lsl #8
.endif

        bl              warp_filter_horz_neon
        srshr           v24.8h,  v0.8h,  #3
        bl              warp_filter_horz_neon
        srshr           v25.8h,  v0.8h,  #3
        bl              warp_filter_horz_neon
        srshr           v26.8h,  v0.8h,  #3
        bl              warp_filter_horz_neon
        srshr           v27.8h,  v0.8h,  #3
        bl              warp_filter_horz_neon
        srshr           v28.8h,  v0.8h,  #3
        bl              warp_filter_horz_neon
        srshr           v29.8h,  v0.8h,  #3
        bl              warp_filter_horz_neon
        srshr           v30.8h,  v0.8h,  #3

1:
        add             w14, w6,  #512
        bl              warp_filter_horz_neon
        srshr           v31.8h,  v0.8h,  #3

        load_filter_row d0, w14, w9
        load_filter_row d1, w14, w9
        load_filter_row d2, w14, w9
        load_filter_row d3, w14, w9
        load_filter_row d4, w14, w9
        load_filter_row d5, w14, w9
        load_filter_row d6, w14, w9
        load_filter_row d7, w14, w9
        transpose_8x8b_xtl v0, v1, v2, v3, v4, v5, v6, v7, sxtl

        // This ordering of smull/smlal/smull2/smlal2 is highly
        // beneficial for Cortex A53 here.
        smull           v16.4s,  v24.4h,  v0.4h
        smlal           v16.4s,  v25.4h,  v1.4h
        smlal           v16.4s,  v26.4h,  v2.4h
        smlal           v16.4s,  v27.4h,  v3.4h
        smlal           v16.4s,  v28.4h,  v4.4h
        smlal           v16.4s,  v29.4h,  v5.4h
        smlal           v16.4s,  v30.4h,  v6.4h
        smlal           v16.4s,  v31.4h,  v7.4h
        smull2          v17.4s,  v24.8h,  v0.8h
        smlal2          v17.4s,  v25.8h,  v1.8h
        smlal2          v17.4s,  v26.8h,  v2.8h
        smlal2          v17.4s,  v27.8h,  v3.8h
        smlal2          v17.4s,  v28.8h,  v4.8h
        smlal2          v17.4s,  v29.8h,  v5.8h
        smlal2          v17.4s,  v30.8h,  v6.8h
        smlal2          v17.4s,  v31.8h,  v7.8h

        mov             v24.16b, v25.16b
        mov             v25.16b, v26.16b
        sqrshrn         v16.4h,  v16.4s,  #\shift
        mov             v26.16b, v27.16b
        sqrshrn2        v16.8h,  v17.4s,  #\shift
        mov             v27.16b, v28.16b
        mov             v28.16b, v29.16b
        add             v16.8h,  v16.8h,  v23.8h
.ifb \t
        sqxtun          v16.8b,  v16.8h
.endif
        mov             v29.16b, v30.16b
        mov             v30.16b, v31.16b
        subs            w10, w10, #1
.ifnb \t
        st1             {v16.8h}, [x0], x1
.else
        st1             {v16.8b}, [x0], x1
.endif

        add             w6,  w6,  w4
        b.gt            1b

        ret             x15
endfunc
.endm

warp  , 11
warp t, 7

// void dav2d_emu_edge_8bpc_neon(
//         const intptr_t bw, const intptr_t bh,
//         const intptr_t iw, const intptr_t ih,
//         const intptr_t x, const intptr_t y,
//         pixel *dst, const ptrdiff_t dst_stride,
//         const pixel *ref, const ptrdiff_t ref_stride)
function emu_edge_8bpc_neon, export=1
        ldp             x8,  x9,  [sp]

        // ref += iclip(y, 0, ih - 1) * PXSTRIDE(ref_stride)
        // ref += iclip(x, 0, iw - 1)
        sub             x12, x3,  #1           // ih - 1
        cmp             x5,  x3
        sub             x13, x2,  #1           // iw - 1
        csel            x12, x12, x5,  ge      // min(y, ih - 1)
        cmp             x4,  x2
        bic             x12, x12, x12, asr #63 // max(min(y, ih - 1), 0)
        csel            x13, x13, x4,  ge      // min(x, iw - 1)
        bic             x13, x13, x13, asr #63 // max(min(x, iw - 1), 0)
        madd            x8,  x12, x9,  x8      // ref += iclip() * stride
        add             x8,  x8,  x13          // ref += iclip()

        // bottom_ext = iclip(y + bh - ih, 0, bh - 1)
        // top_ext = iclip(-y, 0, bh - 1)
        add             x10, x5,  x1           // y + bh
        neg             x5,  x5                // -y
        sub             x10, x10, x3           // y + bh - ih
        sub             x12, x1,  #1           // bh - 1
        cmp             x10, x1
        bic             x5,  x5,  x5,  asr #63 // max(-y, 0)
        csel            x10, x10, x12, lt      // min(y + bh - ih, bh-1)
        cmp             x5,  x1
        bic             x10, x10, x10, asr #63 // max(min(y + bh - ih, bh-1), 0)
        csel            x5,  x5,  x12, lt      // min(max(-y, 0), bh-1)

        // right_ext = iclip(x + bw - iw, 0, bw - 1)
        // left_ext = iclip(-x, 0, bw - 1)
        add             x11, x4,  x0           // x + bw
        neg             x4,  x4                // -x
        sub             x11, x11, x2           // x + bw - iw
        sub             x13, x0,  #1           // bw - 1
        cmp             x11, x0
        bic             x4,  x4,  x4,  asr #63 // max(-x, 0)
        csel            x11, x11, x13, lt      // min(x + bw - iw, bw-1)
        cmp             x4,  x0
        bic             x11, x11, x11, asr #63 // max(min(x + bw - iw, bw-1), 0)
        csel            x4,  x4,  x13, lt      // min(max(-x, 0), bw - 1)

        // center_h = bh - top_ext - bottom_ext
        // dst += top_ext * PXSTRIDE(dst_stride)
        // center_w = bw - left_ext - right_ext
        sub             x1,  x1,  x5           // bh - top_ext
        madd            x6,  x5,  x7,  x6
        sub             x2,  x0,  x4           // bw - left_ext
        sub             x1,  x1,  x10          // center_h = bh - top_ext - bottom_ext
        sub             x2,  x2,  x11          // center_w = bw - left_ext - right_ext

        mov             x14, x6                // backup of dst

.macro v_loop need_left, need_right
0:
.if \need_left
        ld1r            {v0.16b}, [x8]
        mov             x12, x6                // out = dst
        mov             x3,  x4
1:
        subs            x3,  x3,  #16
        st1             {v0.16b}, [x12], #16
        b.gt            1b
.endif
        mov             x13, x8
        add             x12, x6,  x4           // out = dst + left_ext
        mov             x3,  x2
1:
        ld1             {v0.16b, v1.16b}, [x13], #32
        subs            x3,  x3,  #32
        st1             {v0.16b, v1.16b}, [x12], #32
        b.gt            1b
.if \need_right
        add             x3,  x8,  x2           // in + center_w
        sub             x3,  x3,  #1           // in + center_w - 1
        add             x12, x6,  x4           // dst + left_ext
        ld1r            {v0.16b}, [x3]
        add             x12, x12, x2           // out = dst + left_ext + center_w
        mov             x3,  x11
1:
        subs            x3,  x3,  #16
        st1             {v0.16b}, [x12], #16
        b.gt            1b
.endif

        subs            x1,  x1,  #1           // center_h--
        add             x6,  x6,  x7
        add             x8,  x8,  x9
        b.gt            0b
.endm

        cbz             x4,  2f
        // need_left
        cbz             x11, 3f
        // need_left + need_right
        v_loop          1,   1
        b               5f

2:
        // !need_left
        cbz             x11, 4f
        // !need_left + need_right
        v_loop          0,   1
        b               5f

3:
        // need_left + !need_right
        v_loop          1,   0
        b               5f

4:
        // !need_left + !need_right
        v_loop          0,   0

5:

        cbz             x10, 3f
        // need_bottom
        sub             x8,  x6,  x7           // ref = dst - stride
        mov             x4,  x0
1:
        ld1             {v0.16b, v1.16b}, [x8], #32
        mov             x3,  x10
2:
        subs            x3,  x3,  #1
        st1             {v0.16b, v1.16b}, [x6], x7
        b.gt            2b
        msub            x6,  x7,  x10,  x6     // dst -= bottom_ext * stride
        subs            x4,  x4,  #32          // bw -= 32
        add             x6,  x6,  #32          // dst += 32
        b.gt            1b

3:
        cbz             x5,  3f
        // need_top
        msub            x6,  x7,  x5,  x14     // dst = stored_dst - top_ext * stride
1:
        ld1             {v0.16b, v1.16b}, [x14], #32
        mov             x3,  x5
2:
        subs            x3,  x3,  #1
        st1             {v0.16b, v1.16b}, [x6], x7
        b.gt            2b
        msub            x6,  x7,  x5,  x6      // dst -= top_ext * stride
        subs            x0,  x0,  #32          // bw -= 32
        add             x6,  x6,  #32          // dst += 32
        b.gt            1b

3:
        ret
endfunc

.macro SAD8xN_8bpc
        ldr             q0,  [x12]              // overread by 4
        ldr             q1,  [x13]
        ldr             q5,  [x12, x1]          // overread by 4
        ldr             q6,  [x13, x3]
        add             x16, x12, x1, lsl #1    // PXSTRIDE(p0_stride) * 4
        add             x17, x13, x3, lsl #1    // PXSTRIDE(p1_stride) * 4
        uabd            v0.16b, v0.16b, v1.16b
        uabd            v5.16b, v5.16b, v6.16b
        sub             w9,  w5,  #4
        uaddlp          v2.8h,  v0.16b
        uaddlp          v4.8h,  v5.16b
7:
        ldr             q0,  [x16]              // overread by 4
        ldr             q1,  [x17]
        ldr             q5,  [x16, x1]          // overread by 4
        ldr             q6,  [x17, x3]
        add             x16, x16, x1, lsl #1    // PXSTRIDE(p0_stride) * 2
        add             x17, x17, x3, lsl #1    // PXSTRIDE(p1_stride) * 2
        uabd            v0.16b, v0.16b, v1.16b
        uabd            v5.16b, v5.16b, v6.16b
        sub             w9,  w9,  #4
        uadalp          v2.8h,  v0.16b
        uadalp          v4.8h,  v5.16b
        cbnz            w9,  7b

        add             v2.8h,  v2.8h,  v4.8h
        and             v2.16b, v2.16b, v3.16b  // mask the last 4
        uaddlv          s5,  v2.8h
        fmov            w15, s5
.endm

.macro SAD16xN_8bpc
        add             x14, x12, #16
        add             x15, x13, #16
        ldr             q0,  [x12]
        ldr             q1,  [x13]
        sub             w9,  w5,  #4
        ldr             s4,  [x12, #16]
        ldr             s5,  [x13, #16]
        add             x14, x14, x1
        add             x15, x15, x3
        ldr             q16, [x12, x1]
        ldr             q17, [x13, x3]
        ld1             {v4.s}[1], [x14]
        ld1             {v5.s}[1], [x15]
        uabd            v0.16b,  v0.16b,  v1.16b
        uabd            v16.16b, v16.16b, v17.16b
        add             x16, x12, x1, lsl #1 // PXSTRIDE(p0_stride) * 4
        add             x17, x13, x3, lsl #1 // PXSTRIDE(p1_stride) * 4
        uaddlp          v2.8h, v0.16b
        uaddlp          v3.8h, v16.16b
        uabdl           v6.8h, v4.8b, v5.8b
7:
        ldr             q0,  [x16]
        ldr             q1,  [x17]
        ldr             s4,  [x16, #16]
        ldr             s5,  [x17, #16]
        add             x14, x14, x1, lsl #1
        add             x15, x15, x3, lsl #1
        ldr             q16, [x16, x1]
        ldr             q17, [x17, x3]
        ld1             {v4.s}[1], [x14]
        ld1             {v5.s}[1], [x15]
        uabd            v0.16b,  v0.16b,  v1.16b
        uabd            v16.16b, v16.16b, v17.16b
        sub             w9,  w9,  #4
        add             x16, x16, x1, lsl #1 // PXSTRIDE(p0_stride) * 2
        add             x17, x17, x3, lsl #1 // PXSTRIDE(p1_stride) * 2
        uadalp          v2.8h, v0.16b
        uadalp          v3.8h, v16.16b
        uabal           v6.8h, v4.8b, v5.8b
        cbnz            w9,  7b

        add             v2.8h, v2.8h, v3.8h
        add             v2.8h, v2.8h, v6.8h
        uaddlv          s2,  v2.8h
        fmov            w15, s2
.endm

.macro sad_refine_mv_8bpc_w width
        add             w4,  w4,  #4          // w + 4
        add             w5,  w5,  #4          // h + 4
        mul             w8,  w4,  w5
        lsl             w8,  w8,  #1          // sad_thr
        mov             x10, #0               // best_dx
        mov             x11, #0               // best_dy
        mov             x15, #0xffffffff      // best_sad
.if \width==8
        movi            v3.16b, #0xff
        ins             v3.s[3],  wzr
.endif
        cbz             w6,  2f               // is_implicit
        // is_implicit = 1
        add             x12, x0,  x1
        add             x13, x2,  x3
        add             x12, x12, #2          // 2 * p0_stride + 2
        add             x13, x13, #2          // 2 * p1_stride + 2
        SAD\width\()xN_8bpc
        sub             x15, x15, x15, lsr #3 // (best_sad * 7 + 7) >> 3
        cmp             x15, x8               // if best_sad < sad_thr
        b.lt            6f
2:      // is_implicit = 0
        add             x2,  x2,  x3,  lsl #1 // p1_start = p1 + 4 * p1_stride
        mov             x6,  x15              // best_sad
        add             x2,  x2,  #4          // p1_start = p1 + 4 * p1_stride + 4
        mov             x4,  #-2              // y_off
3:
        mov             x12, x0               // p0_ptr = p0_start
        mov             x13, x2               // p1_ptr = p1_start
        mov             x8,  #-2              // x_off
4:
        orr             x9,  x4,  x8
        cbz             x9,  5f               // skip if both are 0
        SAD\width\()xN_8bpc
        cmp             x6,  x15              // if (sad >= best_sad)
        b.ls            5f

        mov             x6,  x15              // best_sad = sad
        mov             x10, x8               // best_dx = x_off
        mov             x11, x4               // best_dy = y_off
5:
        add             x8,  x8,  #1          // x_off++
        add             x12, x12, #1          // p0++
        sub             x13, x13, #1          // p1--
        cmp             x8,  #2
        b.le            4b

        add             x4,  x4,  #1          // y_off++
        add             x0,  x0,  x1, lsr #1  // p0 += p0_stride
        sub             x2,  x2,  x3, lsr #1  // p1 -= p1_stride
        cmp             x4,  #2
        b.le            3b
6:
        strb            w11, [x7]             // o->y
        strb            w10, [x7, #1]         // o->x
        ret
.endm

// void sad_refine_mv_8bpc_neon(const pixel *const p0,
//                              const ptrdiff_t p0_stride,
//                              const pixel *const p1,
//                              const ptrdiff_t p1_stride,
//                              const int w, const int h,
//                              const int is_implicit,
//                              struct OpflOffset *const o)
function sad_refine_mv_8bpc_neon, export=1
        cmp             w4,  #8
        lsl             x1,  x1,  #1
        lsl             x3,  x3,  #1
        b.eq            1f
        sad_refine_mv_8bpc_w 16
1:      sad_refine_mv_8bpc_w 8
endfunc

// unsigned sad8x8_8bpc_neon(const pixel *p0, const ptrdiff_t p0_stride,
//                           const pixel *p1, const ptrdiff_t p1_stride)
function sad8x8_8bpc_neon, export=1
        ldr             d0,  [x0]
        ldr             d1,  [x2]
        ldr             d2,  [x0, x1]
        ldr             d3,  [x2, x3]
        add             x0,  x0,  x1, lsl #1
        add             x2,  x2,  x3, lsl #1
        uabdl           v4.8h,   v0.8b,   v1.8b
        uabdl           v5.8h,   v2.8b,   v3.8b

.rept 2
        ldr             d0,  [x0]
        ldr             d1,  [x2]
        ldr             d2,  [x0, x1]
        ldr             d3,  [x2, x3]
        add             x0,  x0,  x1, lsl #1
        add             x2,  x2,  x3, lsl #1
        uabal           v4.8h,   v0.8b,   v1.8b
        uabal           v5.8h,   v2.8b,   v3.8b
.endr

        ldr             d0,  [x0]
        ldr             d1,  [x2]
        ldr             d2,  [x0, x1]
        ldr             d3,  [x2, x3]
        uabal           v4.8h,   v0.8b,   v1.8b
        uabal           v5.8h,   v2.8b,   v3.8b

        add             v4.8h,   v4.8h,   v5.8h
        uaddlv          s5,  v4.8h
        fmov            w0,  s5
        ret
endfunc

// LUT to combine shifting a 32-bit value by 8 and narrowing it to 16 bits.
const narrow_shift_tbl, align=4
        .byte 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30
endconst

// void morph_8bpc_neon(pixel *dst, const ptrdiff_t dst_stride,
//                      const int alpha, const int beta, const int w,
//                      const int h)
function morph_8bpc_neon, export=1
        dup             v18.4s, w3
        dup             v0.8h,  w2                        // alpha
        cmp             w4,  #8
        b.lt            L(morph_4xh)
        ldrel           q17, x8,  narrow_shift_tbl
        b.eq            L(morph_8xh)

1:      // w >= 16
        mov             w6,  w4                           // width
        mov             x7,  x0                           // dst
2:
        ldr             q3,  [x7]
        mov             v4.16b, v18.16b
        mov             v6.16b, v18.16b
        uxtl            v2.8h,  v3.8b
        uxtl2           v3.8h,  v3.16b
        mov             v5.16b, v18.16b
        mov             v7.16b, v18.16b
        smlal           v4.4s,  v2.4h,  v0.4h             // alpha * dst[x] + beta
        smlal           v6.4s,  v3.4h,  v0.4h
        smlal2          v5.4s,  v2.8h,  v0.8h
        smlal2          v7.4s,  v3.8h,  v0.8h
        sub             w6,  w6,  #16
        tbl             v4.16b, {v4.16b, v5.16b}, v17.16b // narrow and >> 8
        tbl             v6.16b, {v6.16b, v7.16b}, v17.16b
        sqxtun          v4.8b,  v4.8h
        sqxtun          v5.8b,  v6.8h
        stp             d4,  d5,  [x7], #16
        cbnz            w6,  2b

        sub             w5,  w5,  #1                      // height--
        add             x0,  x0,  x1                      // dst += dst_stride
        cbnz            w5,  1b
        ret

L(morph_8xh):
        ldr             d2,  [x0]
        ldr             d3,  [x0, x1]
        mov             v4.16b, v18.16b
        mov             v6.16b, v18.16b
        uxtl            v2.8h,  v2.8b
        uxtl            v3.8h,  v3.8b
        mov             v5.16b, v18.16b
        mov             v7.16b, v18.16b
        smlal           v4.4s,  v2.4h,  v0.4h             // alpha * dst[x] + beta
        smlal           v6.4s,  v3.4h,  v0.4h
        smlal2          v5.4s,  v2.8h,  v0.8h
        smlal2          v7.4s,  v3.8h,  v0.8h
        sub             w5,  w5,  #2
        tbl             v4.16b, {v4.16b, v5.16b}, v17.16b // narrow and >> 8
        tbl             v6.16b, {v6.16b, v7.16b}, v17.16b
        sqxtun          v4.8b,  v4.8h
        sqxtun          v6.8b,  v6.8h
        str             d4,  [x0]
        str             d6,  [x0, x1]
        add             x0,  x0,  x1,  lsl #1
        cbnz            w5,  L(morph_8xh)
        ret

L(morph_4xh):
        ldr             s2,  [x0]
        ldr             s3,  [x0, x1]
        mov             v4.16b, v18.16b
        mov             v5.16b, v18.16b
        uxtl            v2.8h,  v2.8b
        uxtl            v3.8h,  v3.8b
        smlal           v4.4s,  v2.4h, v0.4h              // alpha * dst[x] + beta
        smlal           v5.4s,  v3.4h, v0.4h
        sub             w5,  w5,  #2
        shrn            v4.4h,  v4.4s, #8
        shrn            v5.4h,  v5.4s, #8
        sqxtun          v4.8b,  v4.8h
        sqxtun          v5.8b,  v5.8h
        str             s4,  [x0]
        str             s5,  [x0, x1]
        add             x0,  x0,  x1,  lsl #1
        cbnz            w5,  L(morph_4xh)
        ret
endfunc

// Replace the shift right by 7 with a shift right by 6 when e1 = 1, allowing
// to skip the multiplication by (e1 + 1).
const shift_large
        .short -6, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -6
endconst

// Process a 16-pixel-wide stripe as two adjacent 8x8 subblocks. Compute and
// accumulate two lines at a time until both subblocks are complete, then move
// down to the next pair of subblocks. At the bottom of the block, move 16
// pixels to the right and restart from the top.
.macro opfl_derive_mv_8bpc_large_neon
        uxtw            x5,  w5
        movrel          x8,  shift_large
        sub             sp,  sp,  #224                   // 10 * 128-bit vectors for out accs + 4 spill vectors

        mov             x9,  #0                          // bx = 0
        sub             w16, w6,  #2                     // h - 2
        mul             w14, w7,  w7
2:
        mov             x10, x1                          // p0_ptr
        mov             x11, x3                          // p1_ptr
        mov             x15, x0
        // This decides how much to shift y0 when computing the vertical
        // gradient. It's a shift right by 7 in most cases, unless e2 is 1 in
        // which case a shift right by 6 allows to skip the multiplication by
        // (1 + e2). This is true for y = 0, so initialize for this special
        // case, then later overwrite it for the general case.
        mov             w19, #-6

        // Compute tmp0/tmp1 for the first two lines.
        // in:  line 0: [v4, v6]   (p0, p1)
        //      line 1: [v10, v11] (p0, p1)
        // out: line 0: tmp0 [v1, v2],  tmp1 [v3, v4]
        //      line 1: tmp0 [v7, v8],  tmp1 [v9, v10]
        ldr             q4, [x10]
        ldr             q6, [x11]
        ldr             q10, [x10, x2]
        ldr             q12, [x11, x4]
        mov             x12, #0                          // y = 0
        add             x10, x10, x2, lsl #1
        add             x11, x11, x4, lsl #1
        uxtl            v3.8h,   v4.8b
        uxtl2           v4.8h,   v4.16b
        uxtl            v5.8h,   v6.8b
        uxtl2           v6.8h,   v6.16b
        mul             v1.8h,   v3.8h,   v0.h[0]        // d.i8[0] * p0p (lo line 0)
        mul             v2.8h,   v4.8h,   v0.h[0]        // d.i8[0] * p0p (hi line 0)
        uxtl            v9.8h,   v10.8b
        uxtl2           v10.8h,  v10.16b
        uxtl            v11.8h,  v12.8b
        uxtl2           v12.8h,  v12.16b
        mla             v1.8h,   v5.8h,   v0.h[1]        // d.i8[0] * p0p + d.i8[1] * p1p (lo line 0)
        mla             v2.8h,   v6.8h,   v0.h[1]        // d.i8[0] * p0p + d.i8[1] * p1p (hi line 0)
        sub             v3.8h,   v3.8h,   v5.8h          // tmp1 (lo line 0)
        sub             v4.8h,   v4.8h,   v6.8h          // tmp1 (hi line 0)
        mul             v7.8h,   v9.8h,   v0.h[0]        // d.i8[0] * p0p (lo line 1)
        mul             v8.8h,   v10.8h,  v0.h[0]        // d.i8[0] * p0p (hi line 1)
        mov             v24.16b, v1.16b
        mov             v25.16b, v1.16b
        mov             v26.16b, v1.16b
        sub             v9.8h,   v9.8h,   v11.8h         // tmp1 (lo line 1)
        sub             v10.8h,  v10.8h,  v12.8h         // tmp1 (hi line 1)
        mla             v7.8h,   v11.8h,  v0.h[1]        // d.i8[0] * p0p + d.i8[1] * p1p (lo line 1)
        mla             v8.8h,   v12.8h,  v0.h[1]        // d.i8[0] * p0p + d.i8[1] * p1p (hi line 1)

        mov             v28.16b, v2.16b
        mov             v29.16b, v2.16b
        mov             v30.16b, v2.16b
        mov             v27.16b, v7.16b
        mov             v31.16b, v8.16b

3:
        movi            v11.4s,  #0

        stp             q11, q11, [sp]                  // su2_0, su2_1
        stp             q11, q11, [sp, #32]             // suv_0, suv_1
        stp             q11, q11, [sp, #64]             // sv2_0, sv2_1
        mov             w13, w12                        // py = y
        stp             q11, q11, [sp, #96]             // suw_0, suw_1
        stp             q11, q11, [sp, #128]            // svw_0, svw_1

4:
        // Compute the horizontal gradient for the two current lines.
        // in: line 0: [v27, v31], line 1: [v26, v30]
        // out: line 0: [v20, v19], line 1: [v22, v14]
        movi            v16.8h,  #42
        movi            v11.8h,  #5
        dup             v22.8h,  v27.h[0]               // min_x (line 0)
        dup             v23.8h,  v31.h[7]               // max_x (line 0)
        ldp             q12, q13, [x8]                  // shift_lo/hi
        dup             v20.8h,  v26.h[0]               // min_x (line 1)
        dup             v21.8h,  v30.h[7]               // max_x (line 1)
        ext             v17.16b, v22.16b, v27.16b, #12  // p0_lo (line 0)
        ext             v18.16b, v22.16b, v27.16b, #14  // p1_lo (line 0)
        ext             v1.16b,  v20.16b, v26.16b, #12  // p0_lo (line 1)
        ext             v2.16b,  v20.16b, v26.16b, #14  // p1_lo (line 1)
        ext             v22.16b, v27.16b, v31.16b, #2   // p2_lo (line 0)
        ext             v5.16b,  v27.16b, v31.16b, #4   // p3_lo (line 0)

        sub             v22.8h,  v22.8h,  v18.8h        // p2 - p1 (lo line 0)
        sub             v17.8h,  v5.8h,   v17.8h        // p3 - p0 (lo line 0)
        ext             v20.16b, v26.16b, v30.16b, #2   // p2_lo (line 1)
        ext             v5.16b,  v26.16b, v30.16b, #4   // p3_lo (line 1)
        ext             v14.16b, v31.16b, v23.16b, #2   // p2_hi (line 0)
        ext             v15.16b, v31.16b, v23.16b, #4   // p3_hi (line 0)
        sub             v20.8h,  v20.8h,  v2.8h         // p2 - p1 (lo line 1)
        sub             v5.8h,   v5.8h,   v1.8h         // p3 - p0 (lo line 1)
        ext             v7.16b,  v27.16b, v31.16b, #12  // p0_hi (line 0)
        ext             v23.16b, v27.16b, v31.16b, #14  // p1_hi (line 0)

        ext             v19.16b, v30.16b, v21.16b, #2   // p2_hi (line 1)
        mul             v22.8h,  v22.8h,  v16.8h        // (p2 - p1) * 42 (lo line 0)
        ext             v8.16b,  v26.16b, v30.16b, #14  // p1_hi (line 1)
        ext             v6.16b,  v30.16b, v21.16b, #4   // p3_hi (line 1)
        ext             v21.16b, v26.16b, v30.16b, #12  // p0_hi (line 1)
        mul             v20.8h,  v20.8h,  v16.8h        // (p2 - p1) * 42 (lo line 1)
        mls             v22.8h,  v17.8h,  v11.8h        // y0 (lo line 0)
        sub             v14.8h,  v14.8h,  v23.8h        // p2 - p1 (hi line 0)
        sub             v19.8h,  v19.8h,  v8.8h         // p2 - p1 (hi line 1)
        sub             v6.8h,   v6.8h,   v21.8h        // p3 - p0 (hi line 1)
        mls             v20.8h,  v5.8h,   v11.8h        // y0 (lo line 1)
        sub             v15.8h,  v15.8h,  v7.8h         // p3 - p0 (hi line 0)
        mul             v14.8h,  v14.8h,  v16.8h        // (p2 - p1) * 42 (hi line 0)
        mul             v19.8h,  v19.8h,  v16.8h        // (p2 - p1) * 42 (hi line 1)
        ssra            v22.8h,  v22.8h,  #15           // Adjust for SRSHL rounding up (lo line 0)
        ssra            v20.8h,  v20.8h,  #15           // Adjust for SRSHL rounding up (lo line 1)
        mls             v14.8h,  v15.8h,  v11.8h        // y0 (hi line 0)
        mls             v19.8h,  v6.8h,   v11.8h        // y0 (hi line 1)
        srshl           v22.8h,  v22.8h,  v12.8h        // x0 (lo line 0)
        srshl           v20.8h,  v20.8h,  v12.8h        // x0 (lo line 1)
        ssra            v14.8h,  v14.8h,  #15           // Adjust for SRSHL rounding up (hi line 0)
        ssra            v19.8h,  v19.8h,  #15           // Adjust for SRSHL rounding up (hi line 1)
        ldp             q11, q16, [sp]                  // su2_0, su2_1
        ldp             q17, q18, [sp, #96]             // suw_0, suw_1
        srshl           v14.8h,  v14.8h,  v13.8h        // x0 (hi line 0)
        srshl           v19.8h,  v19.8h,  v13.8h        // x0 (hi line 1)

        // Start accumulating the results that don't need the vertical gradient
        // (su2 and suw).
        smlal           v11.4s,  v20.4h,  v20.4h        // su2_0
        smlal           v16.4s,  v19.4h,  v19.4h        // su2_1
        smlal           v17.4s,  v20.4h,  v3.4h         // suw_0
        smlal           v18.4s,  v19.4h,  v4.4h         // suw_1
        smlal2          v11.4s,  v20.8h,  v20.8h        // su2_0
        smlal2          v16.4s,  v19.8h,  v19.8h        // su2_1
        smlal2          v17.4s,  v20.8h,  v3.8h         // suw_0
        smlal2          v18.4s,  v14.8h,  v10.8h        // suw_1
        smlal           v11.4s,  v22.4h,  v22.4h        // su2_0
        smlal           v16.4s,  v14.4h,  v14.4h        // su2_1
        smlal           v17.4s,  v22.4h,  v9.4h         // suw_0
        smlal           v18.4s,  v14.4h,  v10.4h        // suw_1
        smlal2          v11.4s,  v22.8h,  v22.8h        // su2_0
        smlal2          v16.4s,  v14.8h,  v14.8h        // su2_1
        smlal2          v17.4s,  v22.8h,  v9.8h         // suw_0
        smlal2          v18.4s,  v19.8h,  v4.8h         // suw_1
        cmp             w13, w16                        // py < h - 2
        stp             q11, q16, [sp]                  // su2_0, su2_1
        stp             q17, q18, [sp, #96]             // suw_0, suw_1
        stp             q22, q14, [sp, #192]
        stp             q20, q19, [sp, #160]

        // If needed, compute tmp0/tmp1 for the next two lines (otherwise
        // reuse the last line), then compute the vertical gradients for the
        // current two lines.
        movi            v16.8h,  #42
        movi            v11.8h,  #5
        b.lt            5f

        // We reached the end of the block, compute the vertical gradient
        // reusing the last line as needed. The result for the very last line
        // is right-shifted by 6 to account for e2 = 1.

        // Compute the vertical gradient for the low halves of the current two lines.
        // in:  line 0: [v24, v28], line 1: [v25, v29], line 2: [v26, v30],
        //      line 3: [v27, v31], line 4: [v27, v31], line 5: [v27, v31]
        // out: line 0: [v5,  -],  line 1: [v19, -]
        sub             v5.8h,   v27.8h,  v25.8h         // q2 - q1 (lo line 0)
        sub             v24.8h,  v27.8h,  v24.8h         // q3 - q0 (lo line 0)
        sub             v19.8h,  v27.8h,  v26.8h         // q2 - q1 (lo line 1)
        sub             v25.8h,  v27.8h,  v25.8h         // q3 - q0 (lo line 1)
        mul             v5.8h,   v5.8h,   v16.8h         // (q2 - q1) * 42 (lo line 0)
        mul             v19.8h,  v19.8h,  v16.8h         // (q2 - q1) * 42 (lo line 1)
        mls             v5.8h,   v24.8h,  v11.8h         // y0 (lo line 0)
        mls             v19.8h,  v25.8h,  v11.8h         // y0 (lo line 1)
        ssra            v5.8h,   v5.8h,   #15            // Adjust for SRSHR rounding up (lo line 0)
        ssra            v19.8h,  v19.8h,  #15            // Adjust for SRSHR rounding up (lo line 1)
        srshr           v5.8h,   v5.8h,   #7             // y0 (lo line 0)
        srshr           v19.8h,  v19.8h,  #6             // y0 (lo line 1)
        // Compute the vertical gradient for the high halves of the current two lines.
        // in:  line 0: [-, v28], line 1: [-, v29], line 2: [-, v30],
        //      line 3: [-, v31], line 4: [-, v31], line 5: [-, v31]
        // out: line 0: [-,  v14], line 1: [-,  v15]
        sub             v14.8h,  v31.8h,  v29.8h         // q2 - q1 (hi line 0)
        sub             v28.8h,  v31.8h,  v28.8h         // q3 - q0 (hi line 0)
        sub             v15.8h,  v31.8h,  v30.8h         // q2 - q1 (hi line 1)
        sub             v29.8h,  v31.8h,  v29.8h         // q3 - q0 (hi line 1)
        mul             v14.8h,  v14.8h,  v16.8h         // (q2 - q1) * 42 (hi line 0)
        mul             v15.8h,  v15.8h,  v16.8h         // (q2 - q1) * 42 (hi line 1)
        mls             v14.8h,  v28.8h,  v11.8h         // y0 (hi line 0)
        mls             v15.8h,  v29.8h,  v11.8h         // y0 (hi line 1)
        ssra            v14.8h,  v14.8h,  #15            // Adjust for SRSHR rounding up (hi line 0)
        ssra            v15.8h,  v15.8h,  #15            // Adjust for SRSHR rounding up (hi line 1)
        srshr           v14.8h,  v14.8h,  #7             // y0 (hi line 0)
        srshr           v15.8h,  v15.8h,  #6             // y0 (hi line 1)
        b               6f

5:
        // Compute tmp0 and tmp1 for the next two lines.
        // in:  line 2: [v21, v7]   (p0, p1)
        //      line 3: [v13, v14]  (p0, p1)
        // out: line 2: tmp0 [v1,  v2],  tmp1 [v8,  v21]
        //      line 3: tmp0 [v20, v22], tmp1 [v23, v13]
        ldr             q21, [x10]
        ldr             q7, [x11]
        ldr             q13, [x10, x2]
        ldr             q14, [x11, x4]
        add             x10, x10, x2, lsl #1
        add             x11, x11, x4, lsl #1
        uxtl            v8.8h,   v21.8b
        uxtl2           v21.8h,  v21.16b
        uxtl            v23.8h,  v13.8b
        uxtl2           v13.8h,  v13.16b
        mul             v1.8h,   v8.8h,   v0.h[0]        // d.i8[0] * p0p (lo line 2)
        mul             v2.8h,   v21.8h,  v0.h[0]        // d.i8[0] * p0p (hi line 2)
        mul             v20.8h,  v23.8h,  v0.h[0]        // d.i8[0] * p0p (lo line 3)
        mul             v22.8h,  v13.8h,  v0.h[0]        // d.i8[0] * p0p (hi line 3)
        uxtl            v5.8h,   v7.8b
        uxtl2           v6.8h,   v7.16b
        uxtl            v7.8h,   v14.8b
        uxtl2           v14.8h,  v14.16b
        mla             v1.8h,   v5.8h,   v0.h[1]        // d.i8[0] * p0p + d.i8[1] * p1p (lo line 2)
        mla             v2.8h,   v6.8h,   v0.h[1]        // d.i8[0] * p0p + d.i8[1] * p1p (hi line 2)
        mla             v20.8h,  v7.8h,   v0.h[1]        // d.i8[0] * p0p + d.i8[1] * p1p (lo line 3)
        mla             v22.8h,  v14.8h,  v0.h[1]        // d.i8[0] * p0p + d.i8[1] * p1p (hi line 3)
        sub             v8.8h,   v8.8h,   v5.8h          // tmp1 (lo line 2)
        sub             v21.8h,  v21.8h,  v6.8h          // tmp1 (hi line 2)
        sub             v23.8h,  v23.8h,  v7.8h          // tmp1 (lo line 3)
        sub             v13.8h,  v13.8h,  v14.8h         // tmp1 (hi line 3)
        dup             v17.8h,  w19

        // Compute the vertical gradient for the low halves of the current two lines.
        // in:  line 0: [v24, -], line 1: [v25, -], line 2: [v26, -],
        //      line 3: [v27, -], line 4: [v1,  -], line 5: [v20, -]
        // out: line 0: [v5,  -],  line 1: [v19, -]
        sub             v5.8h,   v27.8h,  v25.8h         // q2 - q1 (lo line 0)
        sub             v24.8h,  v1.8h,   v24.8h         // q3 - q0 (lo line 0)
        sub             v19.8h,  v1.8h,   v26.8h         // q2 - q1 (lo line 1)
        sub             v25.8h,  v20.8h,  v25.8h         // q3 - q0 (lo line 1)
        mul             v5.8h,   v5.8h,   v16.8h         // (q2 - q1) * 42 (lo line 0)
        mul             v19.8h,  v19.8h,  v16.8h         // (q2 - q1) * 42 (lo line 1)
        mls             v5.8h,   v24.8h,  v11.8h         // y0 (lo line 0)
        mls             v19.8h,  v25.8h,  v11.8h         // y0 (lo line 1)
        ssra            v5.8h,   v5.8h,   #15            // Adjust for SRSHL rounding up (lo line 0)
        ssra            v19.8h,  v19.8h,  #15            // Adjust for SRSHR rounding up (lo line 1)
        mov             v24.16b, v26.16b
        mov             v25.16b, v27.16b
        srshl           v5.8h,   v5.8h,   v17.8h         // y0 (lo line 0)
        // For all lines after y = 0 use the regular shift value.
        srshr           v19.8h,  v19.8h,  #7             // y0 (lo line 1)
        mov             v26.16b, v1.16b
        // Compute the vertical gradient for the high halves of the current two lines.
        // in:  line 0: [-, v28], line 1: [-, v29], line 2: [-, v30],
        //      line 3: [-, v31], line 4: [-,  v2], line 5: [-, v22]
        // out: line 0: [-,  v14], line 1: [-,  v15]
        sub             v14.8h,  v31.8h,  v29.8h         // q2 - q1 (hi line 0)
        sub             v28.8h,  v2.8h,   v28.8h         // q3 - q0 (hi line 0)
        mov             v27.16b, v20.16b
        sub             v15.8h,  v2.8h,   v30.8h         // q2 - q1 (hi line 1)
        sub             v29.8h,  v22.8h,  v29.8h         // q3 - q0 (hi line 1)
        mul             v14.8h,  v14.8h,  v16.8h         // (q2 - q1) * 42 (hi line 0)
        mul             v15.8h,  v15.8h,  v16.8h         // (q2 - q1) * 42 (hi line 1)
        mls             v14.8h,  v28.8h,  v11.8h         // y0 (hi line 0)
        mls             v15.8h,  v29.8h,  v11.8h         // y0 (hi line 1)
        mov             v28.16b, v30.16b
        ssra            v14.8h,  v14.8h,  #15            // Adjust for SRSHL rounding up (hi line 0)
        ssra            v15.8h,  v15.8h,  #15            // Adjust for SRSHR rounding up (hi line 1)
        mov             v29.16b, v31.16b
        srshl           v14.8h,  v14.8h,  v17.8h         // y0 (hi line 0)
        srshr           v15.8h,  v15.8h,  #7             // y0 (hi line 1)

        mov             w19, #-7
        mov             v30.16b, v2.16b
        mov             v31.16b, v22.16b
6:

        // v0_lo = v5, v0_hi = v14, v1_lo = v19, v1_hi = v15
        // u0_lo = v1, u0_hi = v2,  u1_lo = v6,  u1_hi = v7
        // w0_lo = v3, w0_hi = v4,  w1_lo = v9,  w1_hi = v10
        // Do the rest of the accumulating (sv2, suv, svw).
        ldp             q1,  q2,  [sp, #160]            // horiz_gradient_low
        ldp             q6,  q7,  [sp, #192]            // horiz_gradient_high
        add             w13, w13, #2                    // py += 2
        add             w17, w12, w7                    // py_end = y + bs

        ldp             q12, q22, [sp, #32]             // suv_0, suv_1
        ldp             q16, q17, [sp, #64]             // sv2_0, sv2_1
        ldp             q18, q20, [sp, #128]            // svw_0, svw_1

        smlal           v12.4s,  v1.4h,   v5.4h         // suv_0
        smlal           v22.4s,  v2.4h,   v14.4h        // suv_1
        smlal           v16.4s,  v5.4h,   v5.4h         // sv2_0
        smlal           v17.4s,  v14.4h,  v14.4h        // sv2_1
        smlal           v18.4s,  v5.4h,   v3.4h         // svw_0
        smlal           v20.4s,  v14.4h,  v4.4h         // svw_1

        smlal2          v12.4s,  v1.8h,   v5.8h         // suv_0
        smlal2          v22.4s,  v2.8h,   v14.8h        // suv_1
        smlal2          v16.4s,  v5.8h,   v5.8h         // sv2_0
        smlal2          v17.4s,  v14.8h,  v14.8h        // sv2_1
        smlal2          v18.4s,  v5.8h,   v3.8h         // svw_0
        smlal2          v20.4s,  v14.8h,  v4.8h         // svw_1

        smlal           v12.4s,  v6.4h,   v19.4h        // suv_0
        smlal           v22.4s,  v7.4h,   v15.4h        // suv_1
        smlal           v16.4s,  v19.4h,  v19.4h        // sv2_0
        smlal           v17.4s,  v15.4h,  v15.4h        // sv2_1
        smlal           v18.4s,  v19.4h,  v9.4h         // svw_0
        smlal           v20.4s,  v15.4h,  v10.4h        // svw_1

        smlal2          v12.4s,  v6.8h,   v19.8h        // suv_0
        smlal2          v22.4s,  v7.8h,   v15.8h        // suv_1
        smlal2          v16.4s,  v19.8h,  v19.8h        // sv2_0
        smlal2          v17.4s,  v15.8h,  v15.8h        // sv2_1
        smlal2          v18.4s,  v19.8h,  v9.8h         // svw_0
        smlal2          v20.4s,  v15.8h,  v10.8h        // svw_1

        cmp             w13, w17                        // py < py_end

        stp             q12, q22, [sp, #32]
        stp             q16, q17, [sp, #64]
        stp             q18, q20, [sp, #128]

        mov             v3.16b,  v8.16b
        mov             v4.16b,  v21.16b
        mov             v9.16b,  v23.16b
        mov             v10.16b, v13.16b

        b.ne            4b

        // Accumulate and store to `out`.
        dup             v20.2d,  x14
        add             w12, w12, w7                    // y += bs
        ldp             q21, q11, [sp]                  // su2_0, su2_1
        ldp             q12, q13, [sp, #32]             // suv_0, suv_1
        ldp             q14, q15, [sp, #64]             // sv2_0, sv2_1
        ldp             q16, q17, [sp, #96]             // suw_0, suw_1
        ldp             q18, q19, [sp, #128]            // svw_0, svw_1
        cmp             w12, w6                         // y < h

        addv            s18,     v18.4s                 // svw_0
        addv            s19,     v19.4s                 // svw_1
        addp            v21.4s,  v21.4s,  v12.4s        // su2_0, suv_0
        addp            v14.4s,  v14.4s,  v16.4s        // sv2_0, suw_0
        addp            v11.4s,  v11.4s,  v13.4s        // su2_1, suv_1
        addp            v15.4s,  v15.4s,  v17.4s        // sv2_1, suw_1
        str             s18, [x15, #16]                 // out[0].svw
        addp            v21.4s,  v21.4s,  v14.4s        // su2_0, suv_0, sv2_0, suw_0
        addp            v11.4s,  v11.4s,  v15.4s        // su2_1, suv_1, sv2_1, suw_1
        str             s19, [x15, #36]                 // out[1].svw
        add             v21.4s,  v21.4s,  v20.4s        // add bs * bs to su2_0 and sv2_0
        add             v11.4s,  v11.4s,  v20.4s        // add bs * bs to su2_1 and sv2_1
        mov             x13, #20
        lsr             x17, x5,  #3
        str             q21, [x15]                      // out[0].{su2, suv, sv2, suw}
        str             q11, [x15, #20]                 // out[1].{su2, suv, sv2, suw}

        madd            x15, x17, x13, x15              // out += width >> 3 (20 bytes per struct)
        b.ne            3b

        add             w9,  w9,  #16                   // bx += 16
        add             x1,  x1,  #16
        cmp             w9,  w5                         // bx < w
        add             x0,  x0,  #40                   // 2 * out
        add             x3,  x3,  #16
        b.ne            2b

        add             sp,  sp,  #224
.endm

// Replace the shift right by 7 with a shift right by 6 when e1 = 1, allowing
// to skip the multiplication by (e1 + 1).
const shift_small
        .short -6, -7, -7, -7, -7, -7, -7, -6
endconst

// Process the 8-pixel-wide block two lines at a time. For bs == 4, the low
// and high halves form two adjacent 4x4 subblocks; for bs == 8, they form one
// 8x8 subblock. After completing the current subblock row, move down to the
// next one until the bottom of the block is reached.
.macro opfl_derive_mv_8bpc_small_neon
        // Compute tmp0/tmp1 for the first two lines.
        // in:  line 0: [v2, v3]  (p0, p1)
        //      line 1: [v8, v9]  (p0, p1)
        // out: line 0: tmp0 [v4], tmp1 [v5]
        //      line 1: tmp0 [v6], tmp1 [v7]
        ldr             d2, [x1]
        ldr             d3, [x3]
        ldr             d8, [x1, x2]
        ldr             d9, [x3, x4]
        uxtl            v2.8h,   v2.8b
        uxtl            v3.8h,   v3.8b
        uxtl            v8.8h,   v8.8b
        uxtl            v9.8h,   v9.8b
        sub             w11, w6,  #2                 // h - 2
        mul             v4.8h,   v2.8h,   v0.h[0]    // d.i8[0] * p0p (line 0)
        mul             v6.8h,   v8.8h,   v0.h[0]    // d.i8[0] * p0p (line 1)
        sub             v5.8h,   v2.8h,   v3.8h      // tmp1 (line 0)
        sub             v7.8h,   v8.8h,   v9.8h      // tmp1 (line 1)
        add             x1,  x1,  x2, lsl #1
        add             x3,  x3,  x4, lsl #1
        mla             v4.8h,   v3.8h,   v0.h[1]    // d.i8[0] * p0p + d.i8[1] * p1p (line 0)
        mla             v6.8h,   v9.8h,   v0.h[1]    // d.i8[0] * p0p + d.i8[1] * p1p (line 1)
        mov             w9,  #0                      // y = 0
        ldrel           q8,  x8,  shift_small
        mul             w16, w7,  w7                 // bs * bs

        mov             v28.16b, v4.16b
        mov             v29.16b, v4.16b
        mov             v30.16b, v4.16b
        mov             v31.16b, v6.16b
        movi            v9.8h,   #42
        movi            v27.8h,  #5
        // This decides how much to shift y0 when computing the vertical
        // gradient. It's a shift right by 7 in most cases, unless e2 is 1 in
        // which case a shift right by 6 allows to skip the multiplication by
        // (1 + e2). This is true for y = 0, so initialize for this special
        // case, then later overwrite it for the general case.
        mov             w12, #-6
2:
        mov             w10, w9                      // py = y
        movi            v10.4s,  #0                  // su2_0
        movi            v11.4s,  #0                  // suv_0
        movi            v12.4s,  #0                  // sv2_0
        movi            v13.4s,  #0                  // suw_0
        movi            v14.4s,  #0                  // svw_0
        add             w17, w9,  w7                 // py_end = y + bs
        movi            v15.4s,  #0                  // su2_1
        movi            v16.4s,  #0                  // suv_1
        movi            v17.4s,  #0                  // sv2_1
        movi            v18.4s,  #0                  // suw_1
        movi            v19.4s,  #0                  // svw_1
3:
        // Compute the horizontal gradient for the current two lines.
        // in:  line 0: [v30], line 1: [v31]
        // out: line 0: [v2], line 1: [v25]
        dup             v20.8h,  v30.h[0]                // min_x (line 0)
        dup             v21.8h,  v30.h[7]                // max_x (line 0)
        ext             v2.16b,  v30.16b, v21.16b, #2    // p2 (line 0)
        ext             v1.16b,  v20.16b, v30.16b, #14   // p1 (line 0)
        ext             v22.16b, v20.16b, v30.16b, #12   // p0 (line 0)
        ext             v3.16b,  v30.16b, v21.16b, #4    // p3 (line 0)
        dup             v20.8h,  v31.h[0]                // min_x (line 1)
        dup             v21.8h,  v31.h[7]                // max_x (line 1)

        sub             v2.8h,   v2.8h,   v1.8h          // p2 - p1 (line 0)
        sub             v3.8h,   v3.8h,   v22.8h         // p3 - p0 (line 0)
        ext             v23.16b, v20.16b, v31.16b, #12   // p0 (line 1)
        ext             v26.16b, v31.16b, v21.16b, #4    // p3 (line 1)
        mul             v2.8h,   v2.8h,   v9.8h          // (p2 - p1) * 42 (line 0)
        ext             v24.16b, v20.16b, v31.16b, #14   // p1 (line 1)
        ext             v25.16b, v31.16b, v21.16b, #2    // p2 (line 1)
        mls             v2.8h,   v3.8h,   v27.8h         // y0 (line 0)
        sub             v25.8h,  v25.8h,  v24.8h         // p2 - p1 (line 1)
        sub             v26.8h,  v26.8h,  v23.8h         // p3 - p0 (line 1)
        mul             v25.8h,  v25.8h,  v9.8h          // (p2 - p1) * 42 (line 1)
        mls             v25.8h,  v26.8h,  v27.8h         // y0 (line 1)
        ssra            v25.8h,  v25.8h,  #15            // Adjust for SRSHL rounding up (line 1)
        ssra            v2.8h,   v2.8h,   #15            // Adjust for SRSHL rounding up (line 0)
        srshl           v25.8h,  v25.8h,  v8.8h          // x0 (line 1)
        srshl           v2.8h,   v2.8h,   v8.8h          // x0 (line 0)
        cmp             w10, w11                         // py < h - 2
        // Start accumulating the results that don't need the vertical gradient
        // (su2 and suw).
        smlal           v10.4s,  v25.4h,  v25.4h         // su2_0
        smlal2          v15.4s,  v25.8h,  v25.8h         // su2_1
        smlal           v13.4s,  v25.4h,  v7.4h          // suw_0
        smlal2          v18.4s,  v25.8h,  v7.8h          // suw_1
        smlal           v10.4s,  v2.4h,   v2.4h          // su2_0
        smlal2          v15.4s,  v2.8h,   v2.8h          // su2_1
        smlal           v13.4s,  v2.4h,   v5.4h          // suw_0
        smlal2          v18.4s,  v2.8h,   v5.8h          // suw_1
        b.lt            4f

        // We reached the end of the block, compute the vertical gradient
        // reusing the last line as needed. The result for the very last line
        // is right-shifted by 6 to account for e2 = 1.

        // Compute the vertical gradient for both rows.
        // in:  line 0: [v28], line 1: [v29], line 2: [v30]
        //      line 3: [v31], line 4: [v31], line 5: [v31]
        // out: line 0: [v24], line 1: [v26]
        sub             v24.8h,  v31.8h,  v29.8h         // q2 - q1 (line 0)
        sub             v1.8h,   v31.8h,  v28.8h         // q3 - q0 (line 0)
        sub             v22.8h,  v31.8h,  v30.8h         // q2 - q1 (line 1)
        sub             v23.8h,  v31.8h,  v29.8h         // q3 - q0 (line 1)
        mul             v24.8h,  v24.8h,  v9.8h          // (q2 - q1) * 42 (line 0)
        mul             v22.8h,  v22.8h,  v9.8h          // (q2 - q1) * 42 (line 1)
        mls             v24.8h,  v1.8h,   v27.8h         // y0 (line 0)
        mls             v22.8h,  v23.8h,  v27.8h         // y0 (line 1)
        ssra            v24.8h,  v24.8h,  #15            // Adjust for SRSHR rounding up (line 0)
        ssra            v22.8h,  v22.8h,  #15            // Adjust for SRSHR rounding up (line 1)
        srshr           v24.8h,  v24.8h,  #7             // y0 (line 0)
        srshr           v26.8h,  v22.8h,  #6             // y0 (line 1)
        // No need to move anything, it's the last iteration.
        b               5f

4:      // Compute tmp0/tmp1 for the next two lines.
        // in:  line 2: [v22, v23] (p0, p1)
        //      line 3: [v2,  v3]  (p0, p1)
        // out: line 2: tmp0 [v4], tmp1 [v20]
        //      line 3: tmp0 [v6], tmp1 [v21]
        ldr             d22, [x1]
        ldr             d23, [x3]
        ldr             d21, [x1, x2]
        ldr             d3, [x3, x4]
        uxtl            v22.8h,  v22.8b
        uxtl            v23.8h,  v23.8b
        uxtl            v21.8h,  v21.8b
        uxtl            v3.8h,   v3.8b
        mul             v4.8h,   v22.8h,  v0.h[0]       // d.i8[0] * p0p (line 2)
        mul             v6.8h,   v21.8h,  v0.h[0]       // d.i8[0] * p0p (line 3)
        sub             v20.8h,  v22.8h,  v23.8h        // tmp1 (line 2)
        sub             v21.8h,  v21.8h,  v3.8h         // tmp1 (line 3)
        mla             v4.8h,   v23.8h,  v0.h[1]       // d.i8[0] * p0p + d.i8[1] * p1p (line 2)
        mla             v6.8h,   v3.8h,   v0.h[1]       // d.i8[0] * p0p + d.i8[1] * p1p (line 3)
        // Compute vertical gradient for the current two lines.
        // in:  line 0: [v28], line 1: [v29], line 2: [v30],
        //      line 3: [v31], line 4: [v4],  line 5: [v6]
        // out: line 0: [v24], line 1: [v26]
        dup             v3.8h,   w12
        add             x1,  x1,  x2, lsl #1
        add             x3,  x3,  x4, lsl #1
        sub             v1.8h,   v4.8h,   v28.8h        // q3 - q0 (line 0)
        sub             v24.8h,  v31.8h,  v29.8h        // q2 - q1 (line 0)
        sub             v22.8h,  v4.8h,   v30.8h        // q2 - q1 (line 1)
        sub             v23.8h,  v6.8h,   v29.8h        // q3 - q0 (line 1)
        // For all lines after y = 0 use the regular shift value.
        mov             w12, #-7
        mul             v24.8h,  v24.8h,  v9.8h         // (q2 - q1) * 42 (line 0)
        mul             v22.8h,  v22.8h,  v9.8h         // (q2 - q1) * 42 (line 1)
        mls             v24.8h,  v1.8h,   v27.8h        // y0 (line 0)
        mls             v22.8h,  v23.8h,  v27.8h        // y0 (line 1)
        ssra            v24.8h,  v24.8h,  #15           // Adjust for SRSHL rounding up (line 0)
        ssra            v22.8h,  v22.8h,  #15           // Adjust for SRSHR rounding up (line 1)
        srshl           v24.8h,  v24.8h,  v3.8h         // y0 (line 0)
        srshr           v26.8h,  v22.8h,  #7            // y0 (line 1)
        mov             v28.16b, v30.16b
        mov             v29.16b, v31.16b
        mov             v30.16b, v4.16b
        mov             v31.16b, v6.16b
5:
        // v0 = v24, v1 = v25, u0 = v22, u1 = v23, w0 = v5, w1 = v7
        // Do the rest of the accumulating (sv2, suv, svw).
        smlal           v14.4s,  v24.4h,  v5.4h         // svw_0
        smlal2          v19.4s,  v24.8h,  v5.8h         // svw_1
        add             w10, w10, #2                    // py += 2
        smlal           v14.4s,  v26.4h,  v7.4h         // svw_0
        smlal2          v19.4s,  v26.8h,  v7.8h         // svw_1
        smlal           v11.4s,  v2.4h,   v24.4h        // suv_0
        smlal2          v16.4s,  v2.8h,   v24.8h        // suv_1
        smlal           v12.4s,  v24.4h,  v24.4h        // sv2_0
        smlal2          v17.4s,  v24.8h,  v24.8h        // sv2_1
        mov             v5.16b,  v20.16b
        mov             v7.16b,  v21.16b
        smlal           v11.4s,  v25.4h,  v26.4h        // suv_0
        smlal2          v16.4s,  v25.8h,  v26.8h        // suv_1
        smlal           v12.4s,  v26.4h,  v26.4h        // sv2_0
        smlal2          v17.4s,  v26.8h,  v26.8h        // sv2_1

        cmp             w10, w17                        // py < py_end
        b.ne            3b

        // Accumulate and store to `out`.
        cmp             w7,  #4                         // bs
        dup             v20.2d,  x16                    // [0, bs * bs, 0, bs * bs]
        b.ne            6f

        // bs == 4
        addv            s14,     v14.4s                 // svw_0
        addv            s19,     v19.4s                 // svw_1
        addp            v10.4s,  v10.4s,  v11.4s        // su2_0, suv_0
        addp            v12.4s,  v12.4s,  v13.4s        // sv2_0, suw_0
        addp            v15.4s,  v15.4s,  v16.4s        // su2_1, suv_1
        addp            v17.4s,  v17.4s,  v18.4s        // sv2_1, suw_1
        addp            v10.4s,  v10.4s,  v12.4s        // su2_0, suv_0, sv2_0, suw_0
        addp            v15.4s,  v15.4s,  v17.4s        // su2_1, suv_1, sv2_1, suw_1
        str             s14, [x0, #16]                  // out[0].svw
        str             s19, [x0, #36]                  // out[1].svw
        add             v10.4s,  v10.4s,  v20.4s        // add bs * bs to su2_0 and sv2_0
        add             v15.4s,  v15.4s,  v20.4s        // add bs * bs to su2_1 and sv2_1
        str             q10, [x0]                       // out[0].{su2, suv, sv2, suw}
        str             q15, [x0, #20]                  // out[1].{su2, suv, sv2, suw}
        add             x0,  x0,  #40                   // out += 2
        b               7f
6:      // bs == 8
        add             v10.4s,  v10.4s,  v15.4s        // su2 = su2_0 + su2_1
        add             v11.4s,  v11.4s,  v16.4s        // suv = suv_0 + suv_1
        add             v12.4s,  v12.4s,  v17.4s        // sv2 = sv2_0 + sv2_1
        add             v13.4s,  v13.4s,  v18.4s        // suw = suw_0 + suw_1
        add             v14.4s,  v14.4s,  v19.4s        // svw = svw_0 + svw_1
        addv            s14,     v14.4s                 // svw
        addp            v10.4s,  v10.4s,  v11.4s        // su2, suv
        addp            v12.4s,  v12.4s,  v13.4s        // sv2, suw
        addp            v10.4s,  v10.4s,  v12.4s        // su2, suv, sv2, suw
        add             v10.4s,  v10.4s,  v20.4s        // add bs * bs to su2 and sv2
        str             s14, [x0, #16]                  // out->svw
        str             q10, [x0]                       // out->{su2, suv, sv2, suw}
        add             x0,  x0,  #20                   // out++
7:
        add             w9,  w9,  w7                    // y += bs
        cmp             w9,  w6                         // y < h
        b.ne            2b
.endm

// void dav2d_opfl_derive_mv_8bpc_neon(struct OpflRegressionData *out,
//                                     const pixel *p0, const ptrdiff_t p0_stride,
//                                     const pixel *p1, const ptrdiff_t p1_stride,
//                                     const int w, const int h, const int bs,
//                                     const union aliasi16 d HIGHBD_DECL_SUFFIX)
function opfl_derive_mv_8bpc_neon, export=1
        ldr             h0, [sp]
        stp             d8,  d9,  [sp, #-80]!
        stp             d10, d11, [sp, #16]
        stp             d12, d13, [sp, #32]
        stp             d14, d15, [sp, #48]
        str             x19,      [sp, #64]
        cmp             w5,  #8
        sxtl            v0.8h,   v0.8b
        b.ne            1f

        opfl_derive_mv_8bpc_small_neon
        b               L(end)

1:
        opfl_derive_mv_8bpc_large_neon
L(end):
        ldr             x19,      [sp, #64]
        ldp             d10, d11, [sp, #16]
        ldp             d12, d13, [sp, #32]
        ldp             d14, d15, [sp, #48]
        ldp             d8,  d9,  [sp], #80
        ret
endfunc
