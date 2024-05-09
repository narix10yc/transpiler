	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0
	.globl	_f32_s3_sep_u3_k1_33330333      ; -- Begin function f32_s3_sep_u3_k1_33330333
	.p2align	2
_f32_s3_sep_u3_k1_33330333:             ; @f32_s3_sep_u3_k1_33330333
; %bb.0:                                ; %entry
	subs	x8, x3, x2
	b.le	LBB0_3
; %bb.1:                                ; %loopBody.preheader
	ldp	q1, q0, [x4]
	dup.4s	v2, v1[1]
	dup.4s	v3, v1[3]
	dup.4s	v4, v0[1]
	dup.4s	v5, v0[2]
	dup.4s	v6, v0[3]
	lsl	x10, x2, #6
	add	x9, x1, x10
	add	x10, x0, x10
LBB0_2:                                 ; %loopBody
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q7, q16, [x10]
	zip2.2d	v17, v7, v16
	mov.d	v7[1], v16[0]
	ldp	q16, q18, [x10, #32]
	zip2.2d	v19, v16, v18
	mov.d	v16[1], v18[0]
	ldp	q18, q20, [x9]
	zip2.2d	v21, v18, v20
	mov.d	v18[1], v20[0]
	ldp	q20, q22, [x9, #32]
	zip2.2d	v23, v20, v22
	mov.d	v20[1], v22[0]
	fmul.4s	v22, v16, v1[0]
	fmul.4s	v24, v7, v1[0]
	fmla.4s	v24, v17, v2
	fmla.4s	v22, v19, v2
	fmul.4s	v25, v21, v0[1]
	fmul.4s	v26, v23, v0[1]
	fsub.4s	v22, v22, v26
	fsub.4s	v24, v24, v25
	fmul.4s	v25, v20, v1[0]
	fmul.4s	v26, v18, v1[0]
	fmla.4s	v26, v21, v2
	fmla.4s	v25, v23, v2
	fmla.4s	v25, v19, v4
	fmla.4s	v26, v17, v4
	fmul.4s	v27, v7, v1[2]
	fmul.4s	v28, v16, v1[2]
	fmla.4s	v28, v19, v3
	fmla.4s	v27, v17, v3
	fmul.4s	v29, v20, v0[2]
	fmul.4s	v30, v18, v0[2]
	fsub.4s	v27, v27, v30
	fsub.4s	v28, v28, v29
	fmul.4s	v29, v21, v0[3]
	fmul.4s	v30, v23, v0[3]
	fsub.4s	v28, v28, v30
	fsub.4s	v27, v27, v29
	fmul.4s	v18, v18, v1[2]
	fmul.4s	v20, v20, v1[2]
	fmla.4s	v20, v16, v5
	fmla.4s	v18, v7, v5
	fmla.4s	v18, v17, v6
	fmla.4s	v20, v19, v6
	fmla.4s	v20, v23, v3
	zip2.2d	v7, v24, v27
	mov.d	v24[1], v27[0]
	fmla.4s	v18, v21, v3
	zip2.2d	v16, v22, v28
	mov.d	v22[1], v28[0]
	zip2.2d	v17, v26, v18
	mov.d	v26[1], v18[0]
	mov.16b	v18, v25
	mov.d	v18[1], v20[0]
	zip2.2d	v19, v25, v20
	stp	q22, q16, [x10, #32]
	stp	q24, q7, [x10], #64
	stp	q18, q19, [x9, #32]
	stp	q26, q17, [x9], #64
	subs	x8, x8, #1
	b.ne	LBB0_2
LBB0_3:                                 ; %ret
	ret
                                        ; -- End function
	.globl	_f32_s3_sep_u3_k0_33330333      ; -- Begin function f32_s3_sep_u3_k0_33330333
	.p2align	2
_f32_s3_sep_u3_k0_33330333:             ; @f32_s3_sep_u3_k0_33330333
; %bb.0:                                ; %entry
	subs	x8, x3, x2
	b.le	LBB1_3
; %bb.1:                                ; %loopBody.preheader
	ldp	q1, q0, [x4]
	dup.4s	v2, v1[1]
	dup.4s	v3, v1[3]
	dup.4s	v4, v0[1]
	dup.4s	v5, v0[2]
	dup.4s	v6, v0[3]
	lsl	x10, x2, #6
	add	x9, x1, x10
	add	x10, x0, x10
LBB1_2:                                 ; %loopBody
                                        ; =>This Inner Loop Header: Depth=1
	mov	x11, x10
	ld2.4s	{ v16, v17 }, [x11], #32
	ld2.4s	{ v18, v19 }, [x11]
	mov	x12, x9
	ld2.4s	{ v20, v21 }, [x12], #32
	ld2.4s	{ v22, v23 }, [x12]
	fmul.4s	v7, v18, v1[0]
	fmul.4s	v24, v16, v1[0]
	fmla.4s	v24, v17, v2
	fmla.4s	v7, v19, v2
	fmul.4s	v25, v21, v0[1]
	fmul.4s	v26, v23, v0[1]
	fsub.4s	v26, v7, v26
	fmul.4s	v7, v18, v1[2]
	fmla.4s	v7, v19, v3
	fmul.4s	v28, v22, v0[2]
	fsub.4s	v7, v7, v28
	fmul.4s	v28, v23, v0[3]
	fsub.4s	v27, v7, v28
	fsub.4s	v24, v24, v25
	fmul.4s	v7, v16, v1[2]
	fmla.4s	v7, v17, v3
	fmul.4s	v28, v20, v0[2]
	fsub.4s	v7, v7, v28
	fmul.4s	v28, v21, v0[3]
	fsub.4s	v25, v7, v28
	fmul.4s	v28, v22, v1[0]
	fmul.4s	v30, v20, v1[0]
	fmla.4s	v30, v21, v2
	fmla.4s	v28, v23, v2
	fmla.4s	v28, v19, v4
	fmul.4s	v29, v22, v1[2]
	fmla.4s	v29, v18, v5
	fmla.4s	v30, v17, v4
	st2.4s	{ v24, v25 }, [x10]
	fmul.4s	v31, v20, v1[2]
	fmla.4s	v31, v16, v5
	st2.4s	{ v26, v27 }, [x11]
	fmla.4s	v31, v17, v6
	fmla.4s	v31, v21, v3
	st2.4s	{ v30, v31 }, [x9]
	fmla.4s	v29, v19, v6
	fmla.4s	v29, v23, v3
	st2.4s	{ v28, v29 }, [x12]
	add	x9, x9, #64
	add	x10, x10, #64
	subs	x8, x8, #1
	b.ne	LBB1_2
LBB1_3:                                 ; %ret
	ret
                                        ; -- End function
.subsections_via_symbols
