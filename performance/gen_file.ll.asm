	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 14, 0
	.globl	_f64_s1_sep_u2q_k2l1            ; -- Begin function f64_s1_sep_u2q_k2l1
	.p2align	2
_f64_s1_sep_u2q_k2l1:                   ; @f64_s1_sep_u2q_k2l1
; %bb.0:                                ; %entry
	subs	x8, x3, x2
	b.le	LBB0_4
; %bb.1:                                ; %loopBody.preheader
	stp	d15, d14, [sp, #-80]!           ; 16-byte Folded Spill
	stp	d13, d12, [sp, #16]             ; 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #64]             ; 16-byte Folded Spill
	sub	sp, sp, #624
	ldp	q16, q0, [x4, #224]
	ldp	q17, q1, [x4, #192]
	ldp	q18, q2, [x4, #160]
	ldp	q19, q3, [x4, #128]
	ldp	q20, q4, [x4, #96]
	ldp	q21, q5, [x4, #64]
	ldp	q22, q6, [x4, #32]
	ldp	q23, q7, [x4]
	stp	q23, q22, [sp, #352]            ; 32-byte Folded Spill
	dup.2d	v24, v23[1]
	dup.2d	v23, v7[0]
	dup.2d	v7, v7[1]
	stp	q7, q23, [sp, #592]             ; 32-byte Folded Spill
	dup.2d	v7, v22[1]
	stp	q7, q24, [sp, #320]             ; 32-byte Folded Spill
	dup.2d	v7, v6[0]
	dup.2d	v6, v6[1]
	stp	q6, q7, [sp, #288]              ; 32-byte Folded Spill
	stp	q21, q20, [sp, #384]            ; 32-byte Folded Spill
	dup.2d	v7, v21[1]
	dup.2d	v6, v5[0]
	stp	q6, q7, [sp, #256]              ; 32-byte Folded Spill
	dup.2d	v6, v5[1]
	dup.2d	v5, v20[1]
	stp	q5, q6, [sp, #224]              ; 32-byte Folded Spill
	dup.2d	v5, v4[0]
	dup.2d	v4, v4[1]
	stp	q4, q5, [sp, #192]              ; 32-byte Folded Spill
	dup.2d	v5, v19[0]
	stp	q19, q18, [sp, #416]            ; 32-byte Folded Spill
	dup.2d	v6, v19[1]
	dup.2d	v4, v3[0]
	stp	q4, q6, [sp, #560]              ; 32-byte Folded Spill
	dup.2d	v4, v3[1]
	dup.2d	v3, v18[0]
	stp	q3, q5, [sp, #160]              ; 32-byte Folded Spill
	dup.2d	v5, v18[1]
	dup.2d	v3, v2[0]
	stp	q3, q5, [sp, #128]              ; 32-byte Folded Spill
	dup.2d	v2, v2[1]
	stp	q2, q4, [sp, #528]              ; 32-byte Folded Spill
	dup.2d	v2, v1[0]
	dup.2d	v1, v1[1]
	stp	q1, q2, [sp, #96]               ; 32-byte Folded Spill
	dup.2d	v1, v0[0]
	dup.2d	v0, v0[1]
	stp	q0, q1, [sp, #64]               ; 32-byte Folded Spill
	mov	w9, #32                         ; =0x20
	orr	x10, x9, x2, lsl #6
	add	x9, x0, x10
	add	x10, x1, x10
	dup.2d	v1, v17[0]
	stp	q17, q16, [sp, #448]            ; 32-byte Folded Spill
	dup.2d	v2, v17[1]
	dup.2d	v0, v16[0]
	stp	q0, q1, [sp, #32]               ; 32-byte Folded Spill
	dup.2d	v0, v16[1]
	stp	q0, q2, [sp]                    ; 32-byte Folded Spill
	ldp	q25, q24, [sp, #336]            ; 32-byte Folded Reload
	ldp	q30, q29, [sp, #304]            ; 32-byte Folded Reload
	ldp	q9, q31, [sp, #272]             ; 32-byte Folded Reload
	ldp	q15, q27, [sp, #128]            ; 32-byte Folded Reload
	ldp	q11, q10, [sp, #240]            ; 32-byte Folded Reload
	ldp	q3, q26, [sp, #80]              ; 32-byte Folded Reload
	ldr	q23, [sp, #400]                 ; 16-byte Folded Reload
	ldp	q13, q12, [sp, #208]            ; 32-byte Folded Reload
	ldr	q14, [sp, #192]                 ; 16-byte Folded Reload
LBB0_2:                                 ; %loopBody
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q2, q0, [x9, #-32]
	ldp	q8, q28, [x9]
	ldp	q6, q16, [x10, #-32]
	ldp	q5, q4, [x10]
	fmul.2d	v7, v2, v24[0]
	fmla.2d	v7, v0, v25
	ldr	q1, [sp, #608]                  ; 16-byte Folded Reload
	fmla.2d	v7, v8, v1
	ldr	q1, [sp, #592]                  ; 16-byte Folded Reload
	fmla.2d	v7, v28, v1
	ldp	q17, q18, [sp, #416]            ; 32-byte Folded Reload
	fmul.2d	v17, v6, v17[0]
	ldr	q1, [sp, #576]                  ; 16-byte Folded Reload
	fmla.2d	v17, v16, v1
	ldr	q1, [sp, #560]                  ; 16-byte Folded Reload
	fmla.2d	v17, v5, v1
	ldr	q1, [sp, #544]                  ; 16-byte Folded Reload
	fmla.2d	v17, v4, v1
	fsub.2d	v1, v7, v17
	str	q1, [sp, #512]                  ; 16-byte Folded Spill
	ldp	q21, q22, [sp, #368]            ; 32-byte Folded Reload
	fmul.2d	v7, v2, v21[0]
	fmla.2d	v7, v0, v29
	fmla.2d	v7, v8, v30
	fmla.2d	v7, v28, v31
	fmul.2d	v18, v6, v18[0]
	fmla.2d	v18, v16, v27
	fmla.2d	v18, v5, v15
	ldr	q1, [sp, #528]                  ; 16-byte Folded Reload
	fmla.2d	v18, v4, v1
	fsub.2d	v1, v7, v18
	str	q1, [sp, #496]                  ; 16-byte Folded Spill
	fmul.2d	v7, v2, v22[0]
	fmla.2d	v7, v0, v9
	fmla.2d	v7, v8, v10
	fmla.2d	v7, v28, v11
	ldp	q19, q20, [sp, #448]            ; 32-byte Folded Reload
	fmul.2d	v19, v6, v19[0]
	ldr	q17, [sp, #16]                  ; 16-byte Folded Reload
	fmla.2d	v19, v16, v17
	ldr	q18, [sp, #112]                 ; 16-byte Folded Reload
	fmla.2d	v19, v5, v18
	fmla.2d	v19, v4, v26
	fsub.2d	v1, v7, v19
	str	q1, [sp, #480]                  ; 16-byte Folded Spill
	fmul.2d	v19, v2, v23[0]
	fmla.2d	v19, v0, v12
	fmla.2d	v19, v8, v13
	fmla.2d	v19, v28, v14
	fmul.2d	v20, v6, v20[0]
	ldr	q1, [sp]                        ; 16-byte Folded Reload
	fmla.2d	v20, v16, v1
	fmla.2d	v20, v5, v3
	ldr	q7, [sp, #64]                   ; 16-byte Folded Reload
	fmla.2d	v20, v4, v7
	fsub.2d	v19, v19, v20
	fmul.2d	v20, v6, v24[0]
	fmla.2d	v20, v16, v25
	fmul.2d	v21, v6, v21[0]
	fmla.2d	v21, v16, v29
	fmul.2d	v22, v6, v22[0]
	fmla.2d	v22, v16, v9
	fmul.2d	v6, v6, v23[0]
	fmla.2d	v6, v16, v12
	ldr	q16, [sp, #608]                 ; 16-byte Folded Reload
	fmla.2d	v20, v5, v16
	fmla.2d	v21, v5, v30
	fmla.2d	v22, v5, v10
	fmla.2d	v6, v5, v13
	ldr	q5, [sp, #592]                  ; 16-byte Folded Reload
	fmla.2d	v20, v4, v5
	fmla.2d	v21, v4, v31
	fmla.2d	v22, v4, v11
	fmla.2d	v6, v4, v14
	ldr	q4, [sp, #176]                  ; 16-byte Folded Reload
	fmla.2d	v20, v2, v4
	ldr	q4, [sp, #160]                  ; 16-byte Folded Reload
	fmla.2d	v21, v2, v4
	ldr	q4, [sp, #48]                   ; 16-byte Folded Reload
	fmla.2d	v22, v2, v4
	ldr	q4, [sp, #32]                   ; 16-byte Folded Reload
	fmla.2d	v6, v2, v4
	ldr	q2, [sp, #576]                  ; 16-byte Folded Reload
	fmla.2d	v20, v0, v2
	fmla.2d	v21, v0, v27
	fmla.2d	v22, v0, v17
	fmla.2d	v6, v0, v1
	ldr	q0, [sp, #560]                  ; 16-byte Folded Reload
	fmla.2d	v20, v8, v0
	fmla.2d	v21, v8, v15
	fmla.2d	v22, v8, v18
	fmla.2d	v6, v8, v3
	ldp	q0, q1, [sp, #496]              ; 32-byte Folded Reload
	stp	q1, q0, [x9, #-32]
	ldr	q0, [sp, #480]                  ; 16-byte Folded Reload
	stp	q0, q19, [x9], #64
	ldr	q0, [sp, #544]                  ; 16-byte Folded Reload
	fmla.2d	v20, v28, v0
	ldr	q0, [sp, #528]                  ; 16-byte Folded Reload
	fmla.2d	v21, v28, v0
	stp	q20, q21, [x10, #-32]
	fmla.2d	v22, v28, v26
	fmla.2d	v6, v28, v7
	stp	q22, q6, [x10], #64
	subs	x8, x8, #1
	b.ne	LBB0_2
; %bb.3:
	add	sp, sp, #624
	ldp	x28, x27, [sp, #64]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             ; 16-byte Folded Reload
	ldp	d15, d14, [sp], #80             ; 16-byte Folded Reload
LBB0_4:                                 ; %ret
	ret
                                        ; -- End function
	.globl	_f64_s1_sep_u2q_k2l1_batched    ; -- Begin function f64_s1_sep_u2q_k2l1_batched
	.p2align	2
_f64_s1_sep_u2q_k2l1_batched:           ; @f64_s1_sep_u2q_k2l1_batched
; %bb.0:                                ; %global_entry
	subs	x8, x5, x4
	b.le	LBB1_3
; %bb.1:                                ; %loopBody_batch_0.preheader
	ldp	q0, q16, [x6, #128]
	ldp	q1, q4, [x6]
	dup.2d	v2, v1[1]
	dup.2d	v3, v4[0]
	dup.2d	v4, v4[1]
	dup.2d	v5, v0[0]
	dup.2d	v6, v0[1]
	dup.2d	v7, v16[0]
	lsl	x10, x4, #6
	add	x9, x3, x10
	add	x10, x2, x10
	mov	w12, #32                        ; =0x20
	bfi	x12, x4, #6, #58
	add	x11, x1, x12
	add	x12, x0, x12
	dup.2d	v16, v16[1]
LBB1_2:                                 ; %loopBody_batch_0
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q17, q18, [x12, #-32]
	ldp	q19, q20, [x12], #64
	ldp	q21, q22, [x11, #-32]
	ldp	q23, q24, [x11], #64
	fmul.2d	v25, v17, v1[0]
	fmla.2d	v25, v18, v2
	fmla.2d	v25, v19, v3
	fmul.2d	v26, v21, v0[0]
	fmla.2d	v26, v22, v6
	fmla.2d	v25, v20, v4
	fmla.2d	v26, v23, v7
	fmla.2d	v26, v24, v16
	fmul.2d	v21, v21, v1[0]
	fmla.2d	v21, v22, v2
	fmla.2d	v21, v23, v3
	fsub.2d	v22, v25, v26
	fmla.2d	v21, v24, v4
	fmla.2d	v21, v17, v5
	fmla.2d	v21, v18, v6
	fmla.2d	v21, v19, v7
	str	q22, [x10], #64
	fmla.2d	v21, v20, v16
	str	q21, [x9], #64
	subs	x8, x8, #1
	b.ne	LBB1_2
LBB1_3:                                 ; %entry_batch_1.loopexit
	ldp	q0, q6, [x6, #224]
	ldp	q2, q7, [x6, #192]
	ldp	q4, q18, [x6, #160]
	ldp	q1, q16, [x6, #96]
	ldp	q3, q17, [x6, #64]
	ldp	q5, q19, [x6, #32]
	subs	x8, x5, x4
	b.gt	LBB1_6
; %bb.4:                                ; %entry_batch_2
	subs	x8, x5, x4
	b.gt	LBB1_9
LBB1_5:                                 ; %entry_batch_3
	subs	x8, x5, x4
	b.gt	LBB1_12
	b	LBB1_14
LBB1_6:                                 ; %loopBody_batch_1.preheader
	dup.2d	v0, v5[1]
	dup.2d	v1, v19[0]
	dup.2d	v2, v19[1]
	dup.2d	v3, v4[0]
	dup.2d	v6, v4[1]
	dup.2d	v7, v18[0]
	mov	w9, #16                         ; =0x10
	orr	x10, x9, x4, lsl #6
	dup.2d	v16, v18[1]
	add	x9, x3, x10
	add	x10, x2, x10
	mov	w12, #32                        ; =0x20
	bfi	x12, x4, #6, #58
	add	x11, x0, x12
	add	x12, x1, x12
LBB1_7:                                 ; %loopBody_batch_1
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q17, q18, [x11, #-32]
	ldp	q19, q20, [x11], #64
	ldp	q21, q22, [x12, #-32]
	ldp	q23, q24, [x12], #64
	fmul.2d	v25, v17, v5[0]
	fmla.2d	v25, v18, v0
	fmla.2d	v25, v19, v1
	fmul.2d	v26, v21, v4[0]
	fmla.2d	v26, v22, v6
	fmla.2d	v25, v20, v2
	fmla.2d	v26, v23, v7
	fmla.2d	v26, v24, v16
	fmul.2d	v21, v21, v5[0]
	fmla.2d	v21, v22, v0
	fmla.2d	v21, v23, v1
	fsub.2d	v22, v25, v26
	fmla.2d	v21, v24, v2
	fmla.2d	v21, v17, v3
	fmla.2d	v21, v18, v6
	fmla.2d	v21, v19, v7
	str	q22, [x10], #64
	fmla.2d	v21, v20, v16
	str	q21, [x9], #64
	subs	x8, x8, #1
	b.ne	LBB1_7
; %bb.8:                                ; %entry_batch_2.loopexit
	ldp	q0, q6, [x6, #224]
	ldp	q2, q7, [x6, #192]
	ldp	q1, q16, [x6, #96]
	ldp	q3, q17, [x6, #64]
	subs	x8, x5, x4
	b.le	LBB1_5
LBB1_9:                                 ; %loopBody_batch_2.preheader
	dup.2d	v0, v3[1]
	dup.2d	v1, v17[0]
	dup.2d	v4, v17[1]
	dup.2d	v5, v2[0]
	dup.2d	v6, v2[1]
	dup.2d	v16, v7[0]
	mov	w9, #32                         ; =0x20
	orr	x12, x9, x4, lsl #6
	dup.2d	v7, v7[1]
	add	x9, x0, x12
	add	x10, x1, x12
	add	x11, x3, x12
	add	x12, x2, x12
LBB1_10:                                ; %loopBody_batch_2
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q17, q18, [x9, #-32]
	ldp	q19, q20, [x9], #64
	ldp	q21, q22, [x10, #-32]
	ldp	q23, q24, [x10], #64
	fmul.2d	v25, v17, v3[0]
	fmla.2d	v25, v18, v0
	fmla.2d	v25, v19, v1
	fmul.2d	v26, v21, v2[0]
	fmla.2d	v26, v22, v6
	fmla.2d	v25, v20, v4
	fmla.2d	v26, v23, v16
	fmla.2d	v26, v24, v7
	fmul.2d	v21, v21, v3[0]
	fmla.2d	v21, v22, v0
	fmla.2d	v21, v23, v1
	fsub.2d	v22, v25, v26
	fmla.2d	v21, v24, v4
	fmla.2d	v21, v17, v5
	fmla.2d	v21, v18, v6
	fmla.2d	v21, v19, v16
	str	q22, [x12], #64
	fmla.2d	v21, v20, v7
	str	q21, [x11], #64
	subs	x8, x8, #1
	b.ne	LBB1_10
; %bb.11:                               ; %entry_batch_3.loopexit
	ldp	q0, q6, [x6, #224]
	ldp	q1, q16, [x6, #96]
	subs	x8, x5, x4
	b.le	LBB1_14
LBB1_12:                                ; %loopBody_batch_3.preheader
	dup.2d	v2, v1[1]
	dup.2d	v3, v16[0]
	dup.2d	v4, v16[1]
	dup.2d	v5, v0[0]
	dup.2d	v7, v0[1]
	dup.2d	v16, v6[0]
	mov	w9, #48                         ; =0x30
	orr	x10, x9, x4, lsl #6
	dup.2d	v6, v6[1]
	add	x9, x2, x10
	add	x10, x3, x10
	mov	w12, #32                        ; =0x20
	bfi	x12, x4, #6, #58
	add	x11, x1, x12
	add	x12, x0, x12
LBB1_13:                                ; %loopBody_batch_3
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q17, q18, [x12, #-32]
	ldp	q19, q20, [x12], #64
	ldp	q21, q22, [x11, #-32]
	ldp	q23, q24, [x11], #64
	fmul.2d	v25, v17, v1[0]
	fmla.2d	v25, v18, v2
	fmla.2d	v25, v19, v3
	fmul.2d	v26, v21, v0[0]
	fmla.2d	v26, v22, v7
	fmla.2d	v25, v20, v4
	fmla.2d	v26, v23, v16
	fmla.2d	v26, v24, v6
	fmul.2d	v21, v21, v1[0]
	fmla.2d	v21, v22, v2
	fmla.2d	v21, v23, v3
	fsub.2d	v22, v25, v26
	fmla.2d	v21, v24, v4
	fmla.2d	v21, v17, v5
	fmla.2d	v21, v18, v7
	fmla.2d	v21, v19, v16
	str	q22, [x9], #64
	fmla.2d	v21, v20, v6
	str	q21, [x10], #64
	subs	x8, x8, #1
	b.ne	LBB1_13
LBB1_14:                                ; %ret
	ret
                                        ; -- End function
	.globl	_f32_s2_sep_u2q_k4l3            ; -- Begin function f32_s2_sep_u2q_k4l3
	.p2align	2
_f32_s2_sep_u2q_k4l3:                   ; @f32_s2_sep_u2q_k4l3
; %bb.0:                                ; %entry
	subs	x8, x3, x2
	b.le	LBB2_4
; %bb.1:                                ; %loopBody.preheader
	stp	d15, d14, [sp, #-80]!           ; 16-byte Folded Spill
	stp	d13, d12, [sp, #16]             ; 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #64]             ; 16-byte Folded Spill
	sub	sp, sp, #608
	ldp	q0, q1, [x4, #96]
	ldp	q3, q2, [x4, #64]
	ldp	q5, q4, [x4, #32]
	ldp	q7, q6, [x4]
	dup.4s	v16, v7[1]
	stp	q16, q7, [sp, #288]             ; 32-byte Folded Spill
	dup.4s	v16, v7[2]
	dup.4s	v7, v7[3]
	stp	q7, q16, [sp, #576]             ; 32-byte Folded Spill
	dup.4s	v7, v6[1]
	str	q7, [sp, #272]                  ; 16-byte Folded Spill
	dup.4s	v7, v6[2]
	str	q7, [sp, #560]                  ; 16-byte Folded Spill
	stp	q6, q5, [sp, #320]              ; 32-byte Folded Spill
	dup.4s	v7, v6[3]
	dup.4s	v6, v5[1]
	stp	q6, q7, [sp, #240]              ; 32-byte Folded Spill
	dup.4s	v6, v5[2]
	dup.4s	v5, v5[3]
	stp	q5, q6, [sp, #208]              ; 32-byte Folded Spill
	dup.4s	v6, v4[1]
	dup.4s	v5, v4[2]
	stp	q5, q6, [sp, #176]              ; 32-byte Folded Spill
	stp	q4, q3, [sp, #352]              ; 32-byte Folded Spill
	dup.4s	v5, v4[3]
	dup.4s	v4, v3[0]
	stp	q4, q5, [sp, #144]              ; 32-byte Folded Spill
	dup.4s	v5, v3[1]
	dup.4s	v4, v3[2]
	stp	q4, q5, [sp, #528]              ; 32-byte Folded Spill
	dup.4s	v4, v3[3]
	dup.4s	v5, v2[0]
	dup.4s	v3, v2[1]
	stp	q3, q4, [sp, #496]              ; 32-byte Folded Spill
	dup.4s	v3, v2[2]
	stp	q2, q0, [sp, #384]              ; 32-byte Folded Spill
	dup.4s	v2, v2[3]
	stp	q2, q3, [sp, #464]              ; 32-byte Folded Spill
	dup.4s	v2, v0[0]
	stp	q2, q5, [sp, #112]              ; 32-byte Folded Spill
	dup.4s	v3, v0[1]
	dup.4s	v2, v0[2]
	stp	q2, q3, [sp, #80]               ; 32-byte Folded Spill
	dup.4s	v2, v0[3]
	lsl	x9, x2, #4
	lsl	x10, x2, #2
	dup.4s	v0, v1[0]
	stp	q0, q2, [sp, #48]               ; 32-byte Folded Spill
	dup.4s	v2, v1[1]
	dup.4s	v0, v1[2]
	stp	q0, q2, [sp, #16]               ; 32-byte Folded Spill
	str	q1, [sp, #416]                  ; 16-byte Folded Spill
	dup.4s	v0, v1[3]
	str	q0, [sp]                        ; 16-byte Folded Spill
	ldp	q27, q26, [sp, #288]            ; 32-byte Folded Reload
	ldr	q25, [sp, #320]                 ; 16-byte Folded Reload
	ldp	q10, q31, [sp, #256]            ; 32-byte Folded Reload
	ldp	q13, q11, [sp, #224]            ; 32-byte Folded Reload
	ldp	q15, q14, [sp, #192]            ; 32-byte Folded Reload
	ldp	q30, q29, [sp, #80]             ; 32-byte Folded Reload
	ldr	q9, [sp, #64]                   ; 16-byte Folded Reload
	ldr	q24, [sp, #352]                 ; 16-byte Folded Reload
	ldr	q8, [sp, #176]                  ; 16-byte Folded Reload
	ldr	q1, [sp, #32]                   ; 16-byte Folded Reload
LBB2_2:                                 ; %loopBody
                                        ; =>This Inner Loop Header: Depth=1
	and	x11, x9, #0x3fffffffffffffe0
	and	x12, x10, #0x4
	orr	x11, x11, x12
	lsl	x11, x11, #2
	orr	x12, x11, #0x20
	orr	x13, x11, #0x40
	orr	x14, x11, #0x60
	ldr	q28, [x0, x11]
	ldr	q12, [x0, x12]
	ldr	q3, [x0, x13]
	ldr	q2, [x0, x14]
	ldr	q19, [x1, x11]
	ldr	q18, [x1, x12]
	ldr	q17, [x1, x13]
	ldr	q16, [x1, x14]
	fmul.4s	v4, v28, v26[0]
	fmla.4s	v4, v12, v27
	ldr	q0, [sp, #592]                  ; 16-byte Folded Reload
	fmla.4s	v4, v3, v0
	ldr	q0, [sp, #576]                  ; 16-byte Folded Reload
	fmla.4s	v4, v2, v0
	ldr	q5, [sp, #368]                  ; 16-byte Folded Reload
	fmul.4s	v20, v19, v5[0]
	ldr	q0, [sp, #544]                  ; 16-byte Folded Reload
	fmla.4s	v20, v18, v0
	ldr	q0, [sp, #528]                  ; 16-byte Folded Reload
	fmla.4s	v20, v17, v0
	fmul.4s	v21, v28, v25[0]
	ldr	q0, [sp, #512]                  ; 16-byte Folded Reload
	fmla.4s	v20, v16, v0
	fmla.4s	v21, v12, v31
	ldr	q0, [sp, #560]                  ; 16-byte Folded Reload
	fmla.4s	v21, v3, v0
	ldr	q5, [sp, #384]                  ; 16-byte Folded Reload
	fmul.4s	v22, v19, v5[0]
	ldr	q0, [sp, #496]                  ; 16-byte Folded Reload
	fmla.4s	v22, v18, v0
	ldr	q0, [sp, #480]                  ; 16-byte Folded Reload
	fmla.4s	v22, v17, v0
	fmla.4s	v21, v2, v10
	ldr	q0, [sp, #464]                  ; 16-byte Folded Reload
	fmla.4s	v22, v16, v0
	ldr	q7, [sp, #336]                  ; 16-byte Folded Reload
	fmul.4s	v5, v28, v7[0]
	fmla.4s	v5, v12, v11
	fmla.4s	v5, v3, v13
	fmla.4s	v5, v2, v14
	fsub.4s	v6, v4, v20
	ldr	q4, [sp, #400]                  ; 16-byte Folded Reload
	fmul.4s	v4, v19, v4[0]
	fmla.4s	v4, v18, v29
	fmla.4s	v4, v17, v30
	fmla.4s	v4, v16, v9
	fmul.4s	v23, v28, v24[0]
	fsub.4s	v0, v21, v22
	stp	q0, q6, [sp, #432]              ; 32-byte Folded Spill
	fmla.4s	v23, v12, v15
	fmla.4s	v23, v3, v8
	ldr	q0, [sp, #160]                  ; 16-byte Folded Reload
	fmla.4s	v23, v2, v0
	ldr	q6, [sp, #416]                  ; 16-byte Folded Reload
	fmul.4s	v6, v19, v6[0]
	fmla.4s	v6, v18, v1
	fsub.4s	v21, v5, v4
	ldp	q20, q4, [sp]                   ; 32-byte Folded Reload
	fmla.4s	v6, v17, v4
	fmla.4s	v6, v16, v20
	fmul.4s	v22, v19, v26[0]
	fmla.4s	v22, v18, v27
	fmul.4s	v5, v19, v25[0]
	fsub.4s	v6, v23, v6
	fmla.4s	v5, v18, v31
	fmul.4s	v23, v19, v7[0]
	fmla.4s	v23, v18, v11
	fmul.4s	v19, v19, v24[0]
	fmla.4s	v19, v18, v15
	ldr	q7, [sp, #592]                  ; 16-byte Folded Reload
	fmla.4s	v22, v17, v7
	ldr	q7, [sp, #560]                  ; 16-byte Folded Reload
	fmla.4s	v5, v17, v7
	fmla.4s	v23, v17, v13
	fmla.4s	v19, v17, v8
	ldr	q7, [sp, #576]                  ; 16-byte Folded Reload
	fmla.4s	v22, v16, v7
	fmla.4s	v5, v16, v10
	fmla.4s	v23, v16, v14
	fmla.4s	v19, v16, v0
	ldr	q0, [sp, #448]                  ; 16-byte Folded Reload
	str	q0, [x0, x11]
	ldr	q0, [sp, #144]                  ; 16-byte Folded Reload
	fmla.4s	v22, v28, v0
	ldr	q0, [sp, #128]                  ; 16-byte Folded Reload
	fmla.4s	v5, v28, v0
	ldr	q0, [sp, #112]                  ; 16-byte Folded Reload
	fmla.4s	v23, v28, v0
	ldr	q0, [sp, #48]                   ; 16-byte Folded Reload
	fmla.4s	v19, v28, v0
	ldr	q0, [sp, #432]                  ; 16-byte Folded Reload
	str	q0, [x0, x12]
	ldr	q0, [sp, #544]                  ; 16-byte Folded Reload
	fmla.4s	v22, v12, v0
	ldr	q0, [sp, #496]                  ; 16-byte Folded Reload
	fmla.4s	v5, v12, v0
	fmla.4s	v23, v12, v29
	fmla.4s	v19, v12, v1
	str	q21, [x0, x13]
	ldr	q0, [sp, #528]                  ; 16-byte Folded Reload
	fmla.4s	v22, v3, v0
	ldr	q0, [sp, #480]                  ; 16-byte Folded Reload
	fmla.4s	v5, v3, v0
	fmla.4s	v23, v3, v30
	fmla.4s	v19, v3, v4
	str	q6, [x0, x14]
	ldr	q0, [sp, #512]                  ; 16-byte Folded Reload
	fmla.4s	v22, v2, v0
	str	q22, [x1, x11]
	ldr	q0, [sp, #464]                  ; 16-byte Folded Reload
	fmla.4s	v5, v2, v0
	str	q5, [x1, x12]
	fmla.4s	v23, v2, v9
	str	q23, [x1, x13]
	fmla.4s	v19, v2, v20
	str	q19, [x1, x14]
	add	x9, x9, #16
	add	x10, x10, #4
	subs	x8, x8, #1
	b.ne	LBB2_2
; %bb.3:
	add	sp, sp, #608
	ldp	x28, x27, [sp, #64]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             ; 16-byte Folded Reload
	ldp	d15, d14, [sp], #80             ; 16-byte Folded Reload
LBB2_4:                                 ; %ret
	ret
                                        ; -- End function
	.globl	_f64_s1_sep_u2q_k2l0            ; -- Begin function f64_s1_sep_u2q_k2l0
	.p2align	2
_f64_s1_sep_u2q_k2l0:                   ; @f64_s1_sep_u2q_k2l0
; %bb.0:                                ; %entry
	subs	x8, x3, x2
	b.le	LBB3_4
; %bb.1:                                ; %loopBody.preheader
	stp	d15, d14, [sp, #-80]!           ; 16-byte Folded Spill
	stp	d13, d12, [sp, #16]             ; 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	stp	x28, x27, [sp, #64]             ; 16-byte Folded Spill
	sub	sp, sp, #576
	ldp	q16, q1, [x4, #224]
	ldp	q17, q3, [x4, #192]
	ldp	q18, q4, [x4, #160]
	ldp	q19, q5, [x4, #128]
	ldp	q20, q6, [x4, #96]
	ldp	q21, q0, [x4, #64]
	ldp	q22, q2, [x4, #32]
	ldp	q23, q7, [x4]
	stp	q23, q22, [sp, #304]            ; 32-byte Folded Spill
	dup.2d	v24, v23[1]
	dup.2d	v23, v7[0]
	stp	q23, q24, [sp, #272]            ; 32-byte Folded Spill
	dup.2d	v7, v7[1]
	str	q7, [sp, #256]                  ; 16-byte Folded Spill
	dup.2d	v7, v22[1]
	str	q7, [sp, #240]                  ; 16-byte Folded Spill
	dup.2d	v7, v2[0]
	dup.2d	v2, v2[1]
	stp	q2, q7, [sp, #528]              ; 32-byte Folded Spill
	str	q21, [sp, #560]                 ; 16-byte Folded Spill
	dup.2d	v7, v21[1]
	dup.2d	v2, v0[0]
	stp	q7, q2, [sp, #496]              ; 32-byte Folded Spill
	dup.2d	v0, v0[1]
	str	q0, [sp, #480]                  ; 16-byte Folded Spill
	stp	q20, q19, [sp, #336]            ; 32-byte Folded Spill
	dup.2d	v2, v20[1]
	dup.2d	v0, v6[0]
	stp	q0, q2, [sp, #208]              ; 32-byte Folded Spill
	dup.2d	v2, v6[1]
	dup.2d	v6, v19[0]
	dup.2d	v0, v19[1]
	stp	q0, q2, [sp, #176]              ; 32-byte Folded Spill
	dup.2d	v0, v5[0]
	stp	q0, q6, [sp, #144]              ; 32-byte Folded Spill
	dup.2d	v2, v5[1]
	dup.2d	v0, v18[0]
	stp	q0, q2, [sp, #112]              ; 32-byte Folded Spill
	str	q18, [sp, #368]                 ; 16-byte Folded Spill
	dup.2d	v2, v18[1]
	dup.2d	v0, v4[0]
	stp	q0, q2, [sp, #448]              ; 32-byte Folded Spill
	dup.2d	v2, v4[1]
	dup.2d	v4, v3[0]
	dup.2d	v0, v3[1]
	stp	q0, q4, [sp, #80]               ; 32-byte Folded Spill
	dup.2d	v3, v1[0]
	dup.2d	v0, v1[1]
	stp	q0, q3, [sp, #48]               ; 32-byte Folded Spill
	lsl	x9, x2, #5
	add	x10, x9, #128
	add	x9, x1, x10
	add	x10, x0, x10
	dup.2d	v1, v17[0]
	stp	q17, q16, [sp, #384]            ; 32-byte Folded Spill
	dup.2d	v0, v17[1]
	stp	q0, q2, [sp, #416]              ; 32-byte Folded Spill
	dup.2d	v0, v16[0]
	stp	q0, q1, [sp, #16]               ; 32-byte Folded Spill
	dup.2d	v0, v16[1]
	str	q0, [sp]                        ; 16-byte Folded Spill
	ldp	q27, q26, [sp, #272]            ; 32-byte Folded Reload
	ldp	q30, q29, [sp, #240]            ; 32-byte Folded Reload
	ldp	q1, q15, [sp, #176]             ; 32-byte Folded Reload
	ldp	q28, q3, [sp, #128]             ; 32-byte Folded Reload
	ldp	q2, q10, [sp, #320]             ; 32-byte Folded Reload
	ldp	q11, q12, [sp, #80]             ; 32-byte Folded Reload
	ldp	q14, q13, [sp, #208]            ; 32-byte Folded Reload
	ldr	q9, [sp]                        ; 16-byte Folded Reload
	ldp	q31, q8, [sp, #48]              ; 32-byte Folded Reload
LBB3_2:                                 ; %loopBody
                                        ; =>This Inner Loop Header: Depth=1
	sub	x12, x10, #128
	ld2.2d	{ v4, v5 }, [x12]
	ld2.2d	{ v16, v17 }, [x10]
	sub	x11, x9, #128
	ld2.2d	{ v6, v7 }, [x11]
	ld2.2d	{ v18, v19 }, [x9]
	ldr	q0, [sp, #304]                  ; 16-byte Folded Reload
	fmul.2d	v20, v4, v0[0]
	fmla.2d	v20, v5, v26
	fmla.2d	v20, v16, v27
	fmla.2d	v20, v17, v29
	ldr	q21, [sp, #352]                 ; 16-byte Folded Reload
	fmul.2d	v21, v6, v21[0]
	fmla.2d	v21, v7, v1
	fmla.2d	v21, v18, v3
	fmla.2d	v21, v19, v28
	fsub.2d	v20, v20, v21
	fmul.2d	v22, v4, v2[0]
	fmla.2d	v22, v5, v30
	ldr	q23, [sp, #544]                 ; 16-byte Folded Reload
	fmla.2d	v22, v16, v23
	ldr	q23, [sp, #528]                 ; 16-byte Folded Reload
	fmla.2d	v22, v17, v23
	ldr	q23, [sp, #368]                 ; 16-byte Folded Reload
	fmul.2d	v23, v6, v23[0]
	ldr	q24, [sp, #464]                 ; 16-byte Folded Reload
	fmla.2d	v23, v7, v24
	ldr	q24, [sp, #448]                 ; 16-byte Folded Reload
	fmla.2d	v23, v18, v24
	ldr	q24, [sp, #432]                 ; 16-byte Folded Reload
	fmla.2d	v23, v19, v24
	fsub.2d	v21, v22, v23
	ldr	q22, [sp, #560]                 ; 16-byte Folded Reload
	fmul.2d	v22, v4, v22[0]
	ldr	q23, [sp, #496]                 ; 16-byte Folded Reload
	fmla.2d	v22, v5, v23
	ldr	q23, [sp, #512]                 ; 16-byte Folded Reload
	fmla.2d	v22, v16, v23
	ldr	q23, [sp, #480]                 ; 16-byte Folded Reload
	fmla.2d	v22, v17, v23
	ldp	q23, q25, [sp, #384]            ; 32-byte Folded Reload
	fmul.2d	v23, v6, v23[0]
	ldr	q24, [sp, #416]                 ; 16-byte Folded Reload
	fmla.2d	v23, v7, v24
	fmla.2d	v23, v18, v12
	fmla.2d	v23, v19, v11
	fsub.2d	v22, v22, v23
	fmul.2d	v24, v4, v10[0]
	fmla.2d	v24, v5, v13
	fmla.2d	v24, v16, v14
	fmla.2d	v24, v17, v15
	fmul.2d	v25, v6, v25[0]
	fmla.2d	v25, v7, v9
	fmla.2d	v25, v18, v8
	fmla.2d	v25, v19, v31
	fsub.2d	v23, v24, v25
	fmul.2d	v24, v6, v0[0]
	fmla.2d	v24, v7, v26
	fmla.2d	v24, v18, v27
	fmla.2d	v24, v19, v29
	ldr	q0, [sp, #160]                  ; 16-byte Folded Reload
	fmla.2d	v24, v4, v0
	fmla.2d	v24, v5, v1
	fmla.2d	v24, v16, v3
	fmla.2d	v24, v17, v28
	fmul.2d	v25, v6, v2[0]
	fmla.2d	v25, v7, v30
	st2.2d	{ v20, v21 }, [x12]
	ldr	q0, [sp, #560]                  ; 16-byte Folded Reload
	fmul.2d	v20, v6, v0[0]
	ldr	q0, [sp, #496]                  ; 16-byte Folded Reload
	fmla.2d	v20, v7, v0
	ldr	q0, [sp, #512]                  ; 16-byte Folded Reload
	fmla.2d	v20, v18, v0
	ldr	q0, [sp, #480]                  ; 16-byte Folded Reload
	fmla.2d	v20, v19, v0
	ldr	q0, [sp, #32]                   ; 16-byte Folded Reload
	fmla.2d	v20, v4, v0
	ldr	q0, [sp, #416]                  ; 16-byte Folded Reload
	fmla.2d	v20, v5, v0
	fmla.2d	v20, v16, v12
	fmla.2d	v20, v17, v11
	fmul.2d	v21, v6, v10[0]
	fmla.2d	v21, v7, v13
	ldr	q0, [sp, #544]                  ; 16-byte Folded Reload
	fmla.2d	v25, v18, v0
	ldr	q0, [sp, #528]                  ; 16-byte Folded Reload
	fmla.2d	v25, v19, v0
	fmla.2d	v21, v18, v14
	fmla.2d	v21, v19, v15
	st2.2d	{ v22, v23 }, [x10], #32
	ldr	q0, [sp, #112]                  ; 16-byte Folded Reload
	fmla.2d	v25, v4, v0
	ldr	q0, [sp, #464]                  ; 16-byte Folded Reload
	fmla.2d	v25, v5, v0
	ldr	q0, [sp, #16]                   ; 16-byte Folded Reload
	fmla.2d	v21, v4, v0
	fmla.2d	v21, v5, v9
	ldr	q0, [sp, #448]                  ; 16-byte Folded Reload
	fmla.2d	v25, v16, v0
	ldr	q0, [sp, #432]                  ; 16-byte Folded Reload
	fmla.2d	v25, v17, v0
	fmla.2d	v21, v16, v8
	fmla.2d	v21, v17, v31
	st2.2d	{ v24, v25 }, [x11]
	st2.2d	{ v20, v21 }, [x9], #32
	subs	x8, x8, #1
	b.ne	LBB3_2
; %bb.3:
	add	sp, sp, #576
	ldp	x28, x27, [sp, #64]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             ; 16-byte Folded Reload
	ldp	d15, d14, [sp], #80             ; 16-byte Folded Reload
LBB3_4:                                 ; %ret
	ret
                                        ; -- End function
	.globl	_f64_s2_sep_u3_k0_10010000      ; -- Begin function f64_s2_sep_u3_k0_10010000
	.p2align	2
_f64_s2_sep_u3_k0_10010000:             ; @f64_s2_sep_u3_k0_10010000
; %bb.0:                                ; %entry
	ret
                                        ; -- End function
	.globl	_f64_s2_sep_u3_k1_33330333      ; -- Begin function f64_s2_sep_u3_k1_33330333
	.p2align	2
_f64_s2_sep_u3_k1_33330333:             ; @f64_s2_sep_u3_k1_33330333
; %bb.0:                                ; %entry
	subs	x8, x3, x2
	b.le	LBB5_4
; %bb.1:                                ; %loopBody.preheader
	stp	d9, d8, [sp, #-16]!             ; 16-byte Folded Spill
	ldp	q1, q0, [x4, #32]
	ldp	q3, q2, [x4]
	dup.2d	v4, v3[1]
	dup.2d	v5, v2[1]
	dup.2d	v6, v1[1]
	dup.2d	v7, v0[0]
	dup.2d	v16, v0[1]
	lsl	x10, x2, #6
	add	x9, x1, x10
	add	x10, x0, x10
LBB5_2:                                 ; %loopBody
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q18, q17, [x10, #32]
	ldp	q20, q19, [x10]
	ldp	q21, q22, [x9]
	ldp	q23, q24, [x9, #32]
	fmul.2d	v25, v20, v3[0]
	fmul.2d	v26, v18, v3[0]
	fmla.2d	v26, v17, v4
	fmla.2d	v25, v19, v4
	fmul.2d	v27, v24, v1[1]
	fmul.2d	v28, v22, v1[1]
	fsub.2d	v25, v25, v28
	fsub.2d	v26, v26, v27
	fmul.2d	v27, v21, v3[0]
	fmul.2d	v28, v23, v3[0]
	fmla.2d	v28, v24, v4
	fmla.2d	v27, v22, v4
	fmla.2d	v27, v19, v6
	fmla.2d	v28, v17, v6
	fmul.2d	v29, v18, v2[0]
	fmul.2d	v30, v20, v2[0]
	fmla.2d	v30, v19, v5
	fmla.2d	v29, v17, v5
	fmul.2d	v31, v21, v0[0]
	fmul.2d	v8, v23, v0[0]
	fsub.2d	v29, v29, v8
	fsub.2d	v30, v30, v31
	fmul.2d	v31, v24, v0[1]
	fmul.2d	v8, v22, v0[1]
	fsub.2d	v30, v30, v8
	fsub.2d	v29, v29, v31
	fmul.2d	v23, v23, v2[0]
	fmul.2d	v21, v21, v2[0]
	fmla.2d	v21, v20, v7
	fmla.2d	v23, v18, v7
	fmla.2d	v23, v17, v16
	fmla.2d	v21, v19, v16
	fmla.2d	v23, v24, v5
	stp	q26, q29, [x10, #32]
	stp	q25, q30, [x10], #64
	stp	q28, q23, [x9, #32]
	fmla.2d	v21, v22, v5
	stp	q27, q21, [x9], #64
	subs	x8, x8, #1
	b.ne	LBB5_2
; %bb.3:
	ldp	d9, d8, [sp], #16               ; 16-byte Folded Reload
LBB5_4:                                 ; %ret
	ret
                                        ; -- End function
	.globl	_f64_s2_sep_u3_k2_33330333      ; -- Begin function f64_s2_sep_u3_k2_33330333
	.p2align	2
_f64_s2_sep_u3_k2_33330333:             ; @f64_s2_sep_u3_k2_33330333
; %bb.0:                                ; %entry
	subs	x8, x3, x2
	b.le	LBB6_4
; %bb.1:                                ; %loopBody.preheader
	stp	d9, d8, [sp, #-16]!             ; 16-byte Folded Spill
	ldp	q1, q0, [x4, #32]
	ldp	q3, q2, [x4]
	dup.2d	v4, v3[1]
	dup.2d	v5, v2[1]
	dup.2d	v6, v1[1]
	dup.2d	v7, v0[0]
	dup.2d	v16, v0[1]
	mov	w9, #32                         ; =0x20
	orr	x10, x9, x2, lsl #6
	add	x9, x0, x10
	add	x10, x1, x10
LBB6_2:                                 ; %loopBody
                                        ; =>This Inner Loop Header: Depth=1
	ldp	q17, q18, [x9, #-32]
	ldp	q19, q20, [x10, #-32]
	ldp	q22, q21, [x9]
	ldp	q24, q23, [x10]
	fmul.2d	v25, v18, v3[0]
	fmul.2d	v26, v17, v3[0]
	fmla.2d	v26, v22, v4
	fmla.2d	v25, v21, v4
	fmul.2d	v27, v24, v1[1]
	fmul.2d	v28, v23, v1[1]
	fsub.2d	v25, v25, v28
	fsub.2d	v26, v26, v27
	fmul.2d	v27, v20, v3[0]
	fmul.2d	v28, v19, v3[0]
	fmla.2d	v28, v24, v4
	fmla.2d	v27, v23, v4
	fmla.2d	v27, v21, v6
	fmla.2d	v28, v22, v6
	fmul.2d	v29, v17, v2[0]
	fmul.2d	v30, v18, v2[0]
	fmla.2d	v30, v21, v5
	fmla.2d	v29, v22, v5
	fmul.2d	v31, v20, v0[0]
	fmul.2d	v8, v19, v0[0]
	fsub.2d	v29, v29, v8
	fsub.2d	v30, v30, v31
	fmul.2d	v31, v24, v0[1]
	fmul.2d	v8, v23, v0[1]
	fsub.2d	v30, v30, v8
	fsub.2d	v29, v29, v31
	fmul.2d	v19, v19, v2[0]
	fmul.2d	v20, v20, v2[0]
	fmla.2d	v20, v18, v7
	fmla.2d	v19, v17, v7
	fmla.2d	v19, v22, v16
	fmla.2d	v20, v21, v16
	fmla.2d	v20, v23, v5
	stp	q26, q25, [x9, #-32]
	stp	q28, q27, [x10, #-32]
	stp	q29, q30, [x9], #64
	fmla.2d	v19, v24, v5
	stp	q19, q20, [x10], #64
	subs	x8, x8, #1
	b.ne	LBB6_2
; %bb.3:
	ldp	d9, d8, [sp], #16               ; 16-byte Folded Reload
LBB6_4:                                 ; %ret
	ret
                                        ; -- End function
	.globl	_f64_s2_sep_u3_k3_33330333      ; -- Begin function f64_s2_sep_u3_k3_33330333
	.p2align	2
_f64_s2_sep_u3_k3_33330333:             ; @f64_s2_sep_u3_k3_33330333
; %bb.0:                                ; %entry
	subs	x8, x3, x2
	b.le	LBB7_4
; %bb.1:                                ; %loopBody.preheader
	stp	d9, d8, [sp, #-16]!             ; 16-byte Folded Spill
	ldp	q1, q0, [x4, #32]
	ldp	q3, q2, [x4]
	dup.2d	v4, v3[1]
	dup.2d	v5, v2[1]
	dup.2d	v6, v1[1]
	dup.2d	v7, v0[0]
	dup.2d	v16, v0[1]
	lsl	x9, x2, #3
	lsl	x10, x2, #2
LBB7_2:                                 ; %loopBody
                                        ; =>This Inner Loop Header: Depth=1
	and	x11, x9, #0x1ffffffffffffff0
	and	x12, x10, #0x4
	orr	x11, x11, x12
	lsl	x11, x11, #3
	add	x14, x0, x11
	add	x12, x1, x11
	orr	x11, x11, #0x40
	add	x13, x0, x11
	add	x11, x1, x11
	ldp	q18, q17, [x14]
	ldp	q20, q19, [x12]
	ldp	q21, q22, [x13]
	ldp	q23, q24, [x11]
	fmul.2d	v25, v18, v3[0]
	fmul.2d	v26, v17, v3[0]
	fmla.2d	v26, v22, v4
	fmla.2d	v25, v21, v4
	fmul.2d	v27, v24, v1[1]
	fmul.2d	v28, v23, v1[1]
	fmul.2d	v29, v20, v3[0]
	fmul.2d	v30, v19, v3[0]
	fsub.2d	v25, v25, v28
	fmla.2d	v30, v24, v4
	fmla.2d	v29, v23, v4
	fmla.2d	v29, v21, v6
	fmla.2d	v30, v22, v6
	fmul.2d	v28, v17, v2[0]
	fsub.2d	v26, v26, v27
	fmul.2d	v27, v18, v2[0]
	fmla.2d	v27, v21, v5
	fmla.2d	v28, v22, v5
	fmul.2d	v31, v20, v0[0]
	fmul.2d	v8, v19, v0[0]
	fsub.2d	v28, v28, v8
	fsub.2d	v27, v27, v31
	fmul.2d	v31, v24, v0[1]
	fmul.2d	v8, v23, v0[1]
	fmul.2d	v19, v19, v2[0]
	fmul.2d	v20, v20, v2[0]
	fsub.2d	v27, v27, v8
	fmla.2d	v20, v18, v7
	fmla.2d	v19, v17, v7
	fmla.2d	v19, v22, v16
	fmla.2d	v20, v21, v16
	fmla.2d	v20, v23, v5
	fsub.2d	v17, v28, v31
	stp	q25, q26, [x14]
	stp	q29, q30, [x12]
	stp	q27, q17, [x13]
	fmla.2d	v19, v24, v5
	stp	q20, q19, [x11]
	add	x9, x9, #8
	add	x10, x10, #4
	subs	x8, x8, #1
	b.ne	LBB7_2
; %bb.3:
	ldp	d9, d8, [sp], #16               ; 16-byte Folded Reload
LBB7_4:                                 ; %ret
	ret
                                        ; -- End function
.subsections_via_symbols
