/******************************************************************************
 * Copyright © 2018, VideoLAN and dav2d authors
 * Copyright © 2015 Martin Storsjo
 * Copyright © 2015 Janne Grunau
 * Copyright © 2026 Arm Limited
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
 *****************************************************************************/

#ifndef DAV2D_SRC_ARM_64_UTIL_S
#define DAV2D_SRC_ARM_64_UTIL_S

#include "config.h"
#include "src/arm/asm.S"

#ifndef __has_feature
#define __has_feature(x) 0
#endif

.macro  ldrel rd, raddr, val, offset=0
#if defined(__APPLE__)
        adrp            \raddr, \val+(\offset)@PAGE
        ldr             \rd, [\raddr, \val+(\offset)@PAGEOFF]
#elif defined(_MSC_VER)
        movrel          \raddr, \val, \offset
        ldr             \rd, [\raddr]
#elif __has_feature(hwaddress_sanitizer)
        adrp            \raddr, :pg_hi21_nc:\val+(\offset)
        movk            \raddr, #:prel_g3:\val+0x100000000
        ldr             \rd, [\raddr, :lo12:\val+(\offset)]
#else
        adrp            \raddr, \val+(\offset)
        ldr             \rd, [\raddr, :lo12:\val+(\offset)]
#endif
.endm

.macro  movrel rd, val, offset=0
#if defined(__APPLE__)
  .if \offset < 0
        adrp            \rd, \val@PAGE
        add             \rd, \rd, \val@PAGEOFF
        sub             \rd, \rd, -(\offset)
  .else
        adrp            \rd, \val+(\offset)@PAGE
        add             \rd, \rd, \val+(\offset)@PAGEOFF
  .endif
#elif defined(_WIN32)
  .if \offset < 0
        adrp            \rd, \val
        add             \rd, \rd, :lo12:\val
        sub             \rd, \rd, -(\offset)
  .else
        adrp            \rd, \val+(\offset)
        add             \rd, \rd, :lo12:\val+(\offset)
  .endif
#elif __has_feature(hwaddress_sanitizer)
        adrp            \rd, :pg_hi21_nc:\val+(\offset)
        movk            \rd, #:prel_g3:\val+0x100000000
        add             \rd, \rd, :lo12:\val+(\offset)
#else
        adrp            \rd, \val+(\offset)
        add             \rd, \rd, :lo12:\val+(\offset)
#endif
.endm

.macro sub_sp space
#ifdef _WIN32
.if \space > 8192
        // Here, we'd need to touch two (or more) pages while decrementing
        // the stack pointer.
        .error          "sub_sp_align doesn't support values over 8K at the moment"
.elseif \space > 4096
        sub             x16, sp,  #4096
        ldr             xzr, [x16]
        sub             sp,  x16, #(\space - 4096)
.else
        sub             sp,  sp,  #\space
.endif
#else
.if \space >= 4096
        sub             sp,  sp,  #(\space)/4096*4096
.endif
.if (\space % 4096) != 0
        sub             sp,  sp,  #(\space)%4096
.endif
#endif
.endm

.macro transpose_8x8b_xtl r0, r1, r2, r3, r4, r5, r6, r7, xtl
        // a0 b0 a1 b1 a2 b2 a3 b3 a4 b4 a5 b5 a6 b6 a7 b7
        zip1            \r0\().16b, \r0\().16b, \r1\().16b
        // c0 d0 c1 d1 c2 d2 d3 d3 c4 d4 c5 d5 c6 d6 d7 d7
        zip1            \r2\().16b, \r2\().16b, \r3\().16b
        // e0 f0 e1 f1 e2 f2 e3 f3 e4 f4 e5 f5 e6 f6 e7 f7
        zip1            \r4\().16b, \r4\().16b, \r5\().16b
        // g0 h0 g1 h1 g2 h2 h3 h3 g4 h4 g5 h5 g6 h6 h7 h7
        zip1            \r6\().16b, \r6\().16b, \r7\().16b

        // a0 b0 c0 d0 a2 b2 c2 d2 a4 b4 c4 d4 a6 b6 c6 d6
        trn1            \r1\().8h,  \r0\().8h,  \r2\().8h
        // a1 b1 c1 d1 a3 b3 c3 d3 a5 b5 c5 d5 a7 b7 c7 d7
        trn2            \r3\().8h,  \r0\().8h,  \r2\().8h
        // e0 f0 g0 h0 e2 f2 g2 h2 e4 f4 g4 h4 e6 f6 g6 h6
        trn1            \r5\().8h,  \r4\().8h,  \r6\().8h
        // e1 f1 g1 h1 e3 f3 g3 h3 e5 f5 g5 h5 e7 f7 g7 h7
        trn2            \r7\().8h,  \r4\().8h,  \r6\().8h

        // a0 b0 c0 d0 e0 f0 g0 h0 a4 b4 c4 d4 e4 f4 g4 h4
        trn1            \r0\().4s,  \r1\().4s,  \r5\().4s
        // a2 b2 c2 d2 e2 f2 g2 h2 a6 b6 c6 d6 e6 f6 g6 h6
        trn2            \r2\().4s,  \r1\().4s,  \r5\().4s
        // a1 b1 c1 d1 e1 f1 g1 h1 a5 b5 c5 d5 e5 f5 g5 h5
        trn1            \r1\().4s,  \r3\().4s,  \r7\().4s
        // a3 b3 c3 d3 e3 f3 g3 h3 a7 b7 c7 d7 e7 f7 g7 h7
        trn2            \r3\().4s,  \r3\().4s,  \r7\().4s

        \xtl\()2        \r4\().8h,  \r0\().16b
        \xtl            \r0\().8h,  \r0\().8b
        \xtl\()2        \r6\().8h,  \r2\().16b
        \xtl            \r2\().8h,  \r2\().8b
        \xtl\()2        \r5\().8h,  \r1\().16b
        \xtl            \r1\().8h,  \r1\().8b
        \xtl\()2        \r7\().8h,  \r3\().16b
        \xtl            \r3\().8h,  \r3\().8b
.endm

.macro transpose_8x8h r0, r1, r2, r3, r4, r5, r6, r7, t8, t9
        trn1            \t8\().8h,  \r0\().8h,  \r1\().8h
        trn2            \t9\().8h,  \r0\().8h,  \r1\().8h
        trn1            \r1\().8h,  \r2\().8h,  \r3\().8h
        trn2            \r3\().8h,  \r2\().8h,  \r3\().8h
        trn1            \r0\().8h,  \r4\().8h,  \r5\().8h
        trn2            \r5\().8h,  \r4\().8h,  \r5\().8h
        trn1            \r2\().8h,  \r6\().8h,  \r7\().8h
        trn2            \r7\().8h,  \r6\().8h,  \r7\().8h

        trn1            \r4\().4s,  \r0\().4s,  \r2\().4s
        trn2            \r2\().4s,  \r0\().4s,  \r2\().4s
        trn1            \r6\().4s,  \r5\().4s,  \r7\().4s
        trn2            \r7\().4s,  \r5\().4s,  \r7\().4s
        trn1            \r5\().4s,  \t9\().4s,  \r3\().4s
        trn2            \t9\().4s,  \t9\().4s,  \r3\().4s
        trn1            \r3\().4s,  \t8\().4s,  \r1\().4s
        trn2            \t8\().4s,  \t8\().4s,  \r1\().4s

        trn1            \r0\().2d,  \r3\().2d,  \r4\().2d
        trn2            \r4\().2d,  \r3\().2d,  \r4\().2d
        trn1            \r1\().2d,  \r5\().2d,  \r6\().2d
        trn2            \r5\().2d,  \r5\().2d,  \r6\().2d
        trn2            \r6\().2d,  \t8\().2d,  \r2\().2d
        trn1            \r2\().2d,  \t8\().2d,  \r2\().2d
        trn1            \r3\().2d,  \t9\().2d,  \r7\().2d
        trn2            \r7\().2d,  \t9\().2d,  \r7\().2d
.endm

.macro transpose_8x8h_mov r0, r1, r2, r3, r4, r5, r6, r7, t8, t9, o0, o1, o2, o3, o4, o5, o6, o7
        trn1            \t8\().8h,  \r0\().8h,  \r1\().8h
        trn2            \t9\().8h,  \r0\().8h,  \r1\().8h
        trn1            \r1\().8h,  \r2\().8h,  \r3\().8h
        trn2            \r3\().8h,  \r2\().8h,  \r3\().8h
        trn1            \r0\().8h,  \r4\().8h,  \r5\().8h
        trn2            \r5\().8h,  \r4\().8h,  \r5\().8h
        trn1            \r2\().8h,  \r6\().8h,  \r7\().8h
        trn2            \r7\().8h,  \r6\().8h,  \r7\().8h

        trn1            \r4\().4s,  \r0\().4s,  \r2\().4s
        trn2            \r2\().4s,  \r0\().4s,  \r2\().4s
        trn1            \r6\().4s,  \r5\().4s,  \r7\().4s
        trn2            \r7\().4s,  \r5\().4s,  \r7\().4s
        trn1            \r5\().4s,  \t9\().4s,  \r3\().4s
        trn2            \t9\().4s,  \t9\().4s,  \r3\().4s
        trn1            \r3\().4s,  \t8\().4s,  \r1\().4s
        trn2            \t8\().4s,  \t8\().4s,  \r1\().4s

        trn1            \o0\().2d,  \r3\().2d,  \r4\().2d
        trn2            \o4\().2d,  \r3\().2d,  \r4\().2d
        trn1            \o1\().2d,  \r5\().2d,  \r6\().2d
        trn2            \o5\().2d,  \r5\().2d,  \r6\().2d
        trn2            \o6\().2d,  \t8\().2d,  \r2\().2d
        trn1            \o2\().2d,  \t8\().2d,  \r2\().2d
        trn1            \o3\().2d,  \t9\().2d,  \r7\().2d
        trn2            \o7\().2d,  \t9\().2d,  \r7\().2d
.endm

.macro transpose_8x16b r0, r1, r2, r3, r4, r5, r6, r7, t8, t9
        trn1            \t8\().16b, \r0\().16b, \r1\().16b
        trn2            \t9\().16b, \r0\().16b, \r1\().16b
        trn1            \r1\().16b, \r2\().16b, \r3\().16b
        trn2            \r3\().16b, \r2\().16b, \r3\().16b
        trn1            \r0\().16b, \r4\().16b, \r5\().16b
        trn2            \r5\().16b, \r4\().16b, \r5\().16b
        trn1            \r2\().16b, \r6\().16b, \r7\().16b
        trn2            \r7\().16b, \r6\().16b, \r7\().16b

        trn1            \r4\().8h,  \r0\().8h,  \r2\().8h
        trn2            \r2\().8h,  \r0\().8h,  \r2\().8h
        trn1            \r6\().8h,  \r5\().8h,  \r7\().8h
        trn2            \r7\().8h,  \r5\().8h,  \r7\().8h
        trn1            \r5\().8h,  \t9\().8h,  \r3\().8h
        trn2            \t9\().8h,  \t9\().8h,  \r3\().8h
        trn1            \r3\().8h,  \t8\().8h,  \r1\().8h
        trn2            \t8\().8h,  \t8\().8h,  \r1\().8h

        trn1            \r0\().4s,  \r3\().4s,  \r4\().4s
        trn2            \r4\().4s,  \r3\().4s,  \r4\().4s
        trn1            \r1\().4s,  \r5\().4s,  \r6\().4s
        trn2            \r5\().4s,  \r5\().4s,  \r6\().4s
        trn2            \r6\().4s,  \t8\().4s,  \r2\().4s
        trn1            \r2\().4s,  \t8\().4s,  \r2\().4s
        trn1            \r3\().4s,  \t9\().4s,  \r7\().4s
        trn2            \r7\().4s,  \t9\().4s,  \r7\().4s
.endm

.macro  transpose_4x16b r0, r1, r2, r3, t4, t5, t6, t7
        trn1            \t4\().16b, \r0\().16b, \r1\().16b
        trn2            \t5\().16b, \r0\().16b, \r1\().16b
        trn1            \t6\().16b, \r2\().16b, \r3\().16b
        trn2            \t7\().16b, \r2\().16b, \r3\().16b

        trn1            \r0\().8h,  \t4\().8h,  \t6\().8h
        trn2            \r2\().8h,  \t4\().8h,  \t6\().8h
        trn1            \r1\().8h,  \t5\().8h,  \t7\().8h
        trn2            \r3\().8h,  \t5\().8h,  \t7\().8h
.endm

.macro  transpose_4x4h  r0, r1, r2, r3, t4, t5, t6, t7
        trn1            \t4\().4h,  \r0\().4h,  \r1\().4h
        trn2            \t5\().4h,  \r0\().4h,  \r1\().4h
        trn1            \t6\().4h,  \r2\().4h,  \r3\().4h
        trn2            \t7\().4h,  \r2\().4h,  \r3\().4h

        trn1            \r0\().2s,  \t4\().2s,  \t6\().2s
        trn2            \r2\().2s,  \t4\().2s,  \t6\().2s
        trn1            \r1\().2s,  \t5\().2s,  \t7\().2s
        trn2            \r3\().2s,  \t5\().2s,  \t7\().2s
.endm

.macro  transpose_4x4s  r0, r1, r2, r3, t4, t5, t6, t7
        trn1            \t4\().4s,  \r0\().4s,  \r1\().4s
        trn2            \t5\().4s,  \r0\().4s,  \r1\().4s
        trn1            \t6\().4s,  \r2\().4s,  \r3\().4s
        trn2            \t7\().4s,  \r2\().4s,  \r3\().4s

        trn1            \r0\().2d,  \t4\().2d,  \t6\().2d
        trn2            \r2\().2d,  \t4\().2d,  \t6\().2d
        trn1            \r1\().2d,  \t5\().2d,  \t7\().2d
        trn2            \r3\().2d,  \t5\().2d,  \t7\().2d
.endm

.macro  transpose_4x8h  r0, r1, r2, r3, t4, t5, t6, t7
        trn1            \t4\().8h,  \r0\().8h,  \r1\().8h
        trn2            \t5\().8h,  \r0\().8h,  \r1\().8h
        trn1            \t6\().8h,  \r2\().8h,  \r3\().8h
        trn2            \t7\().8h,  \r2\().8h,  \r3\().8h

        trn1            \r0\().4s,  \t4\().4s,  \t6\().4s
        trn2            \r2\().4s,  \t4\().4s,  \t6\().4s
        trn1            \r1\().4s,  \t5\().4s,  \t7\().4s
        trn2            \r3\().4s,  \t5\().4s,  \t7\().4s
.endm

.macro  transpose_4x8h_mov r0, r1, r2, r3, t4, t5, t6, t7, o0, o1, o2, o3
        trn1            \t4\().8h,  \r0\().8h,  \r1\().8h
        trn2            \t5\().8h,  \r0\().8h,  \r1\().8h
        trn1            \t6\().8h,  \r2\().8h,  \r3\().8h
        trn2            \t7\().8h,  \r2\().8h,  \r3\().8h

        trn1            \o0\().4s,  \t4\().4s,  \t6\().4s
        trn2            \o2\().4s,  \t4\().4s,  \t6\().4s
        trn1            \o1\().4s,  \t5\().4s,  \t7\().4s
        trn2            \o3\().4s,  \t5\().4s,  \t7\().4s
.endm

.macro load_2_strided dst0, dst1, ptr, out_ptr, stride, inc=0
        ldr             \dst0, [\ptr]
        ldr             \dst1, [\ptr, \stride]
  .if \inc
        add             \out_ptr, \ptr, \stride, lsl #1
  .endif
.endm

.macro load_4_strided dst0, dst1, dst2, dst3, ptr, out_ptr, stride, inc=0
        ldr             \dst0, [\ptr]
        ldr             \dst1, [\ptr, \stride]
        add             \out_ptr, \ptr, \stride, lsl #1
        ldr             \dst2, [\out_ptr]
        ldr             \dst3, [\out_ptr, \stride]
  .if \inc
        add             \out_ptr, \out_ptr, \stride, lsl #1
  .endif
.endm

.macro load_8_strided dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7, ptr, out_ptr, stride, inc=0
        load_4_strided  \dst0, \dst1, \dst2, \dst3, \ptr, \out_ptr, \stride, 1
        load_4_strided  \dst4, \dst5, \dst6, \dst7, \out_ptr, \out_ptr, \stride, \inc
.endm

.macro store_2_strided dst0, dst1, ptr, out_ptr, stride, inc=0
        str             \dst0, [\ptr]
        str             \dst1, [\ptr, \stride]
  .if \inc
        add             \out_ptr, \ptr, \stride, lsl #1
  .endif
.endm

.macro store_4_strided dst0, dst1, dst2, dst3, ptr, out_ptr, stride, inc=0
        str             \dst0, [\ptr]
        str             \dst1, [\ptr, \stride]
        add             \out_ptr, \ptr, \stride, lsl #1
        str             \dst2, [\out_ptr]
        str             \dst3, [\out_ptr, \stride]
  .if \inc
        add             \out_ptr, \out_ptr, \stride, lsl #1
  .endif
.endm

.macro store_8_strided dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7, ptr, out_ptr, stride, inc=0
        store_4_strided \dst0, \dst1, \dst2, \dst3, \ptr, \out_ptr, \stride, 1
        store_4_strided \dst4, \dst5, \dst6, \dst7, \out_ptr, \out_ptr, \stride, \inc
.endm

.macro shift_4x4s dst0, dst1, dst2, dst3, src0, src1, src2, src3, vshift
        srshl           \dst0\().4s, \src0\().4s, \vshift\().4s
        srshl           \dst1\().4s, \src1\().4s, \vshift\().4s
        srshl           \dst2\().4s, \src2\().4s, \vshift\().4s
        srshl           \dst3\().4s, \src3\().4s, \vshift\().4s
.endm

.macro shift_and_narrow_4x4s dst0, dst1, dst2, dst3, src0, src1, src2, src3, vshift
        shift_4x4s      \dst0, \dst1, \dst2, \dst3, \src0, \src1, \src2, \src3, \vshift
        sqxtn           \dst0\().4h, \dst0\().4s
        sqxtn           \dst1\().4h, \dst1\().4s
        sqxtn           \dst2\().4h, \dst2\().4s
        sqxtn           \dst3\().4h, \dst3\().4s
.endm

#endif /* DAV2D_SRC_ARM_64_UTIL_S */
