; ModuleID = 'myModule'
source_filename = "myModule"

define void @f64_s2_sep_u2q_k7l6_ffccffccffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 0
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 15
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 64
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 192
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm1, %Im1
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_4)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_5
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_6)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_7)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_5)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_9)
  %newRe1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_10)
  %newRe1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_11)
  %newRe1 = fsub <4 x double> %newRe1_8, %newRe1_12
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_8)
  %newRe2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_13)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_14)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe1_12)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_16)
  %newRe2 = fsub <4 x double> %newRe2_15, %newRe2_17
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_15)
  %newRe3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_18)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_19)
  %newRe3_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_17)
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_21)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_23)
  %newRe3 = fsub <4 x double> %newRe3_20, %newRe3_24
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_25)
  %newIm0_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_26)
  %newIm0_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_27)
  %newIm0_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_28)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_30)
  %newIm1_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_31)
  %newIm1_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_32)
  %newIm1_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_33)
  %newIm1_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_34)
  %newIm1_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_35)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_37)
  %newIm2_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_38)
  %newIm2_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_39)
  %newIm2_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_40)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_42)
  %newIm3_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_43)
  %newIm3_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_44)
  %newIm3_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_45)
  %newIm3_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_46)
  %newIm3_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_47)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_29, ptr %pIm0, align 32
  store <4 x double> %newIm1_36, ptr %pIm1, align 32
  store <4 x double> %newIm2_41, ptr %pIm2, align 32
  store <4 x double> %newIm3_48, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #0

define void @f64_s2_sep_u2q_k7l6_0030c0000430c001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 0
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 15
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 64
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 192
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %Re0)
  %newRe1_1 = fmul <4 x double> %mIm7, %Im3
  %newRe1 = fsub <4 x double> %newRe1_, %newRe1_1
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_)
  %newRe2_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe1_1)
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_2
  %newRe3_ = fadd <4 x double> %newRe2_, %Re1
  %newRe3 = fsub <4 x double> %newRe3_, %newRe2_2
  %newIm1_ = fmul <4 x double> %mRe7, %Im3
  %newIm1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_)
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %newIm1_3, ptr %pIm1, align 32
  store <4 x double> %newIm2_4, ptr %pIm2, align 32
  store <4 x double> %Im1, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k4l0_0000000010400401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 16
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re3
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1_, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2_, <4 x double> %newRe3_, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %Im1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %Im3, <4 x double> %Im2, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l0_0ff000000ff0f00f(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 64
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_2)
  %newRe2_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2_4 = fmul <4 x double> %mIm10, %Im2
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_4)
  %newRe2 = fsub <4 x double> %newRe2_3, %newRe2_5
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_3)
  %newRe3_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_5)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_6, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe6, %Im2
  %newIm1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm2_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_11)
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_12)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_14)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_15)
  %newReLo = shufflevector <4 x double> %newRe0_1, <4 x double> %newRe1_2, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_9, <4 x double> %newIm1_10, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_13, <4 x double> %newIm3_16, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l0_000000000ff0f00f(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 64
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_2)
  %newRe2_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_3)
  %newRe3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe6, %Im2
  %newIm1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newReLo = shufflevector <4 x double> %newRe0_1, <4 x double> %newRe1_2, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2_3, <4 x double> %newRe3_4, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_5, <4 x double> %newIm1_6, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_7, <4 x double> %newIm3_8, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l0_fffffff0fff0ffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 128
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm2, %Im2
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_4)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_5
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_6)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_7)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_5)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_9)
  %newRe1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_10)
  %newRe1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_11)
  %newRe1 = fsub <4 x double> %newRe1_8, %newRe1_12
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_8)
  %newRe2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_12)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_14)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_16)
  %newRe2 = fsub <4 x double> %newRe2_13, %newRe2_17
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_13)
  %newRe3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_18)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_19)
  %newRe3_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_17)
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_21)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_23)
  %newRe3 = fsub <4 x double> %newRe3_20, %newRe3_24
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_25)
  %newIm0_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_26)
  %newIm0_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_27)
  %newIm0_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_28)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_30)
  %newIm1_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_31)
  %newIm1_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_32)
  %newIm1_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_33)
  %newIm1_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_34)
  %newIm1_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_35)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm2_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_37)
  %newIm2_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_38)
  %newIm2_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_39)
  %newIm2_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_40)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_42)
  %newIm3_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_43)
  %newIm3_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_44)
  %newIm3_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_45)
  %newIm3_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_46)
  %newIm3_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_47)
  %newReLo = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_29, <4 x double> %newIm1_36, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_41, <4 x double> %newIm3_48, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l3_0c0cc0c03c3cc3c3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 2
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 40
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm3, %Im3
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_2
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_1)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe0_2)
  %newRe1 = fsub <4 x double> %newRe1_3, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe1_3)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe1_4)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe2_6
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_5)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_6)
  %newRe3 = fsub <4 x double> %newRe3_7, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm0_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_9)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe9, %Im1
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_)
  %newIm2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_13)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_15)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_10, ptr %pIm0, align 32
  store <4 x double> %newIm1_12, ptr %pIm1, align 32
  store <4 x double> %newIm2_14, ptr %pIm2, align 32
  store <4 x double> %newIm3_16, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l1_3c3cc3c03c3cc303(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = fmul <4 x double> %mIm3, %Im3
  %newRe0 = fsub <4 x double> %newRe0_, %newRe0_1
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_1)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_2, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe1_2)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe1_4)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_6)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe2_7
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_5)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_7)
  %newRe3_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_9)
  %newRe3 = fsub <4 x double> %newRe3_8, %newRe3_10
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_12)
  %newIm1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_13)
  %newIm2_ = fmul <4 x double> %mRe9, %Im1
  %newIm2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_)
  %newIm2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_15)
  %newIm2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_16)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_18)
  %newIm3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_19)
  %newReLo = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_11, <4 x double> %newIm1_14, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_17, <4 x double> %newIm3_20, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l1_0ff0f00000f0f00f(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_3 = fmul <4 x double> %mIm6, %Im2
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_2, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_2)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe1_4)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_6)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe2_7
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_7)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_)
  %newRe3 = fsub <4 x double> %newRe2_5, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe6, %Im2
  %newIm1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_10)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_13)
  %newIm2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_14)
  %newIm3_ = fmul <4 x double> %mIm12, %Re0
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_)
  %newReLo = shufflevector <4 x double> %newRe0_1, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_9, <4 x double> %newIm1_12, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_15, <4 x double> %newIm3_16, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l3_cc330000cc33cc33(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 6
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 72
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_2)
  %newRe2_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe2_4 = fmul <4 x double> %mIm8, %Im0
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_4)
  %newRe2 = fsub <4 x double> %newRe2_3, %newRe2_5
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_3)
  %newRe3_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_5)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_6, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_)
  %newIm2_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_11)
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_12)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_)
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_14)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_15)
  store <4 x double> %newRe0_1, ptr %pRe0, align 32
  store <4 x double> %newRe1_2, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_9, ptr %pIm0, align 32
  store <4 x double> %newIm1_10, ptr %pIm1, align 32
  store <4 x double> %newIm2_13, ptr %pIm2, align 32
  store <4 x double> %newIm3_16, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l6_ffffffffffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 0
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 15
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 64
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 192
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm0, %Im0
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Im1, <4 x double> %newRe0_4)
  %newRe0_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_5)
  %newRe0_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_6)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_7
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_9)
  %newRe1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_7)
  %newRe1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_11)
  %newRe1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_12)
  %newRe1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_13)
  %newRe1 = fsub <4 x double> %newRe1_10, %newRe1_14
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_10)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_16)
  %newRe2_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_14)
  %newRe2_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_18)
  %newRe2_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_19)
  %newRe2_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_20)
  %newRe2 = fsub <4 x double> %newRe2_17, %newRe2_21
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_17)
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_23)
  %newRe3_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_21)
  %newRe3_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_25)
  %newRe3_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_26)
  %newRe3_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_27)
  %newRe3 = fsub <4 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_29)
  %newIm0_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_30)
  %newIm0_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm0, <4 x double> %Re0, <4 x double> %newIm0_31)
  %newIm0_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_32)
  %newIm0_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_33)
  %newIm0_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_34)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_36)
  %newIm1_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_37)
  %newIm1_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_38)
  %newIm1_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_39)
  %newIm1_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_40)
  %newIm1_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_41)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_43)
  %newIm2_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_44)
  %newIm2_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_45)
  %newIm2_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_46)
  %newIm2_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_47)
  %newIm2_49 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_48)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_50 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_51 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_50)
  %newIm3_52 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_51)
  %newIm3_53 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_52)
  %newIm3_54 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_53)
  %newIm3_55 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_54)
  %newIm3_56 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_55)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_35, ptr %pIm0, align 32
  store <4 x double> %newIm1_42, ptr %pIm1, align 32
  store <4 x double> %newIm2_49, ptr %pIm2, align 32
  store <4 x double> %newIm3_56, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l2_3cc33cc03cc33c03(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 3
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 4
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 36
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = fmul <4 x double> %mIm3, %Im3
  %newRe0 = fsub <4 x double> %newRe0_, %newRe0_1
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe0_1)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_2, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_2)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_4)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_6)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe2_7
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_5)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_7)
  %newRe3_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_9)
  %newRe3 = fsub <4 x double> %newRe3_8, %newRe3_10
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_)
  %newIm1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_12)
  %newIm1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_13)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_15)
  %newIm2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_16)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_18)
  %newIm3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_19)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_11, ptr %pIm0, align 32
  store <4 x double> %newIm1_14, ptr %pIm1, align 32
  store <4 x double> %newIm2_17, ptr %pIm2, align 32
  store <4 x double> %newIm3_20, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l2_c3ffff3c3cffffc3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 3
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 4
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 36
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm1, %Im1
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_2)
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_3
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_1)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_4)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_5)
  %newRe1_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_3)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_7)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_9)
  %newRe1 = fsub <4 x double> %newRe1_6, %newRe1_10
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_6)
  %newRe2_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_11)
  %newRe2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_12)
  %newRe2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_10)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_14)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_16)
  %newRe2 = fsub <4 x double> %newRe2_13, %newRe2_17
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_13)
  %newRe3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_17)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_19)
  %newRe3 = fsub <4 x double> %newRe3_18, %newRe3_20
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm0_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_21)
  %newIm0_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_22)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_24)
  %newIm1_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_25)
  %newIm1_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_26)
  %newIm1_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_27)
  %newIm1_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_28)
  %newIm1_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_29)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_31)
  %newIm2_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_32)
  %newIm2_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_33)
  %newIm2_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_34)
  %newIm2_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_35)
  %newIm2_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_36)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_38)
  %newIm3_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_39)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_23, ptr %pIm0, align 32
  store <4 x double> %newIm1_30, ptr %pIm1, align 32
  store <4 x double> %newIm2_37, ptr %pIm2, align 32
  store <4 x double> %newIm3_40, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l2_30c0000030c00401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 3
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 4
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 36
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe2_1 = fmul <4 x double> %mIm11, %Im3
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_1
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe3_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe2_1)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_2
  %newIm2_ = fmul <4 x double> %mRe11, %Im3
  %newIm2_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe14, %Im2
  %newIm3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_)
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im1, ptr %pIm1, align 32
  store <4 x double> %newIm2_3, ptr %pIm2, align 32
  store <4 x double> %newIm3_4, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k2l1_ffffffffffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pReal = getelementptr <16 x double>, ptr %preal, i64 %idx
  %pImag = getelementptr <16 x double>, ptr %pimag, i64 %idx
  %Real = load <16 x double>, ptr %pReal, align 128
  %Imag = load <16 x double>, ptr %pImag, align 128
  %Re0 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 0, i32 1, i32 8, i32 9>
  %Re1 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 2, i32 3, i32 10, i32 11>
  %Re2 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 4, i32 5, i32 12, i32 13>
  %Re3 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 6, i32 7, i32 14, i32 15>
  %Im0 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 0, i32 1, i32 8, i32 9>
  %Im1 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 2, i32 3, i32 10, i32 11>
  %Im2 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 4, i32 5, i32 12, i32 13>
  %Im3 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 6, i32 7, i32 14, i32 15>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm0, %Im0
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Im1, <4 x double> %newRe0_4)
  %newRe0_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_5)
  %newRe0_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_6)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_7
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_9)
  %newRe1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_7)
  %newRe1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_11)
  %newRe1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_12)
  %newRe1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_13)
  %newRe1 = fsub <4 x double> %newRe1_10, %newRe1_14
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_10)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_16)
  %newRe2_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_14)
  %newRe2_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_18)
  %newRe2_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_19)
  %newRe2_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_20)
  %newRe2 = fsub <4 x double> %newRe2_17, %newRe2_21
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_17)
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_23)
  %newRe3_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_21)
  %newRe3_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_25)
  %newRe3_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_26)
  %newRe3_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_27)
  %newRe3 = fsub <4 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_29)
  %newIm0_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_30)
  %newIm0_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm0, <4 x double> %Re0, <4 x double> %newIm0_31)
  %newIm0_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_32)
  %newIm0_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_33)
  %newIm0_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_34)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_36)
  %newIm1_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_37)
  %newIm1_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_38)
  %newIm1_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_39)
  %newIm1_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_40)
  %newIm1_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_41)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_43)
  %newIm2_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_44)
  %newIm2_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_45)
  %newIm2_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_46)
  %newIm2_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_47)
  %newIm2_49 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_48)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_50 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_51 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_50)
  %newIm3_52 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_51)
  %newIm3_53 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_52)
  %newIm3_54 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_53)
  %newIm3_55 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_54)
  %newIm3_56 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_55)
  %vecRe0 = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecRe1 = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newReal = shufflevector <8 x double> %vecRe0, <8 x double> %vecRe1, <16 x i32> <i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13, i32 2, i32 3, i32 6, i32 7, i32 10, i32 11, i32 14, i32 15>
  store <16 x double> %newReal, ptr %pReal, align 128
  %vecIm0 = shufflevector <4 x double> %newIm0_35, <4 x double> %newIm1_42, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecIm1 = shufflevector <4 x double> %newIm2_49, <4 x double> %newIm3_56, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newImag = shufflevector <8 x double> %vecIm0, <8 x double> %vecIm1, <16 x i32> <i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13, i32 2, i32 3, i32 6, i32 7, i32 10, i32 11, i32 14, i32 15>
  store <16 x double> %newImag, ptr %pImag, align 128
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k1l0_ccccccccffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pReal = getelementptr <16 x double>, ptr %preal, i64 %idx
  %pImag = getelementptr <16 x double>, ptr %pimag, i64 %idx
  %Real = load <16 x double>, ptr %pReal, align 128
  %Imag = load <16 x double>, ptr %pImag, align 128
  %Re0 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %Re1 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %Re2 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %Re3 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %Im0 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %Im1 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %Im2 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %Im3 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm1, %Im1
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_4)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_5
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_6)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_7)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe0_5)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_9)
  %newRe1 = fsub <4 x double> %newRe1_8, %newRe1_10
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_8)
  %newRe2_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_11)
  %newRe2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_12)
  %newRe2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe1_10)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_14)
  %newRe2 = fsub <4 x double> %newRe2_13, %newRe2_15
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_13)
  %newRe3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_16)
  %newRe3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_17)
  %newRe3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_15)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_19)
  %newRe3 = fsub <4 x double> %newRe3_18, %newRe3_20
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_21)
  %newIm0_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_22)
  %newIm0_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_23)
  %newIm0_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_24)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_26)
  %newIm1_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_27)
  %newIm1_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_28)
  %newIm1_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_29)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_31)
  %newIm2_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_32)
  %newIm2_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_33)
  %newIm2_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_34)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_36)
  %newIm3_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_37)
  %newIm3_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_38)
  %newIm3_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_39)
  %vecRe0 = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecRe1 = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newReal = shufflevector <8 x double> %vecRe0, <8 x double> %vecRe1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x double> %newReal, ptr %pReal, align 128
  %vecIm0 = shufflevector <4 x double> %newIm0_25, <4 x double> %newIm1_30, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecIm1 = shufflevector <4 x double> %newIm2_35, <4 x double> %newIm3_40, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newImag = shufflevector <8 x double> %vecIm0, <8 x double> %vecIm1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x double> %newImag, ptr %pImag, align 128
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k4l0_0c00c0000c10c001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 16
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %Re0)
  %newRe1_1 = fmul <4 x double> %mIm7, %Im3
  %newRe1 = fsub <4 x double> %newRe1_, %newRe1_1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re2
  %newRe2 = fsub <4 x double> %newRe2_, %newRe1_1
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe3_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe1_1)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_2
  %newIm1_ = fmul <4 x double> %mRe7, %Im3
  %newIm1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_)
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %newIm1_3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %Im2, <4 x double> %newIm3_4, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l4_0c00c0000c10c001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 12
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 3
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 16
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 144
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %Re0)
  %newRe1_1 = fmul <4 x double> %mIm7, %Im3
  %newRe1 = fsub <4 x double> %newRe1_, %newRe1_1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re2
  %newRe2 = fsub <4 x double> %newRe2_, %newRe1_1
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe3_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe1_1)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_2
  %newIm1_ = fmul <4 x double> %mRe7, %Im3
  %newIm1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_)
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %newIm1_3, ptr %pIm1, align 32
  store <4 x double> %Im2, ptr %pIm2, align 32
  store <4 x double> %newIm3_4, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l0_ffccffccffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 128
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm1, %Im1
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_4)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_5
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_6)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_7)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_5)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_9)
  %newRe1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_10)
  %newRe1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_11)
  %newRe1 = fsub <4 x double> %newRe1_8, %newRe1_12
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_8)
  %newRe2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_13)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_14)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe1_12)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_16)
  %newRe2 = fsub <4 x double> %newRe2_15, %newRe2_17
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_15)
  %newRe3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_18)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_19)
  %newRe3_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_17)
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_21)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_23)
  %newRe3 = fsub <4 x double> %newRe3_20, %newRe3_24
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_25)
  %newIm0_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_26)
  %newIm0_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_27)
  %newIm0_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_28)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_30)
  %newIm1_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_31)
  %newIm1_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_32)
  %newIm1_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_33)
  %newIm1_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_34)
  %newIm1_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_35)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_37)
  %newIm2_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_38)
  %newIm2_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_39)
  %newIm2_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_40)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_42)
  %newIm3_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_43)
  %newIm3_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_44)
  %newIm3_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_45)
  %newIm3_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_46)
  %newIm3_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_47)
  %newReLo = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_29, <4 x double> %newIm1_36, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_41, <4 x double> %newIm3_48, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l0_3c3cc0c03c3cc3c3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 128
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm3, %Im3
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_2
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_1)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe0_2)
  %newRe1 = fsub <4 x double> %newRe1_3, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe1_3)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe1_4)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_6)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe2_7
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_5)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_7)
  %newRe3_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_9)
  %newRe3 = fsub <4 x double> %newRe3_8, %newRe3_10
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm0_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_11)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_13)
  %newIm2_ = fmul <4 x double> %mRe9, %Im1
  %newIm2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_)
  %newIm2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_15)
  %newIm2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_16)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_18)
  %newIm3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_19)
  %newReLo = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_12, <4 x double> %newIm1_14, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_17, <4 x double> %newIm3_20, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l3_3300cc0033cccc33(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 2
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 40
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_3 = fmul <4 x double> %mIm5, %Im1
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_2, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe1_2)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe1_4
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_5)
  %newRe3_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe1_4)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_6, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_10)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe9, %Im1
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_14)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_15)
  store <4 x double> %newRe0_1, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_9, ptr %pIm0, align 32
  store <4 x double> %newIm1_12, ptr %pIm1, align 32
  store <4 x double> %newIm2_13, ptr %pIm2, align 32
  store <4 x double> %newIm3_16, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l3_00ffff00ff0000ff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 2
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 40
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe1_ = fmul <4 x double> %mIm4, %Im0
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_)
  %newRe1_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_4)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_5)
  %newRe1 = fsub <4 x double> %newRe0_3, %newRe1_6
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_6)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_)
  %newRe2_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_7)
  %newRe2_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_8)
  %newRe2 = fsub <4 x double> %newRe0_3, %newRe2_9
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe3_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_10)
  %newRe3_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_11)
  %newRe3 = fsub <4 x double> %newRe3_12, %newRe2_9
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_13)
  %newIm0_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_14)
  %newIm1_ = fmul <4 x double> %mIm4, %Re0
  %newIm1_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_)
  %newIm1_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_16)
  %newIm1_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_17)
  %newIm2_ = fmul <4 x double> %mIm8, %Re0
  %newIm2_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_)
  %newIm2_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_19)
  %newIm2_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_20)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_22)
  %newIm3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_23)
  store <4 x double> %newRe0_3, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_15, ptr %pIm0, align 32
  store <4 x double> %newIm1_18, ptr %pIm1, align 32
  store <4 x double> %newIm2_21, ptr %pIm2, align 32
  store <4 x double> %newIm3_24, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l5_ff00ff00ffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 8
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 7
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 32
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 160
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_4)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_5)
  %newRe1_7 = fmul <4 x double> %mIm4, %Im0
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_7)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_9)
  %newRe1 = fsub <4 x double> %newRe1_6, %newRe1_10
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_6)
  %newRe2_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_11)
  %newRe2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_12)
  %newRe2 = fsub <4 x double> %newRe2_13, %newRe1_10
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_13)
  %newRe3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_14)
  %newRe3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_15)
  %newRe3_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe1_10)
  %newRe3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_17)
  %newRe3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_18)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_19)
  %newRe3 = fsub <4 x double> %newRe3_16, %newRe3_20
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_21)
  %newIm0_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_22)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_24)
  %newIm1_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_25)
  %newIm1_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_26)
  %newIm1_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_27)
  %newIm1_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_28)
  %newIm1_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_29)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_31)
  %newIm2_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_32)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_34)
  %newIm3_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_35)
  %newIm3_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_36)
  %newIm3_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_37)
  %newIm3_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_38)
  %newIm3_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_39)
  store <4 x double> %newRe0_3, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_23, ptr %pIm0, align 32
  store <4 x double> %newIm1_30, ptr %pIm1, align 32
  store <4 x double> %newIm2_33, ptr %pIm2, align 32
  store <4 x double> %newIm3_40, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l3_0000000004104001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 2
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 40
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re3
  %newRe2_ = fadd <4 x double> %newRe1_, %Re2
  %newRe3_ = fadd <4 x double> %newRe2_, %Re1
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2_, ptr %pRe2, align 32
  store <4 x double> %newRe3_, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im3, ptr %pIm1, align 32
  store <4 x double> %Im2, ptr %pIm2, align 32
  store <4 x double> %Im1, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l5_ffff0ff0fffff00f(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 8
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 7
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 32
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 160
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm2, %Im2
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_2)
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_3
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_3)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_5)
  %newRe1 = fsub <4 x double> %newRe1_4, %newRe1_6
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_4)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_7)
  %newRe2_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_8)
  %newRe2_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_6)
  %newRe2_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_10)
  %newRe2_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_11)
  %newRe2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_12)
  %newRe2 = fsub <4 x double> %newRe2_9, %newRe2_13
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_9)
  %newRe3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_14)
  %newRe3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_15)
  %newRe3_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_13)
  %newRe3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_17)
  %newRe3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_18)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_19)
  %newRe3 = fsub <4 x double> %newRe3_16, %newRe3_20
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_21)
  %newIm0_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_22)
  %newIm1_ = fmul <4 x double> %mRe6, %Im2
  %newIm1_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_24)
  %newIm1_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_25)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_27)
  %newIm2_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_28)
  %newIm2_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_29)
  %newIm2_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_30)
  %newIm2_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_31)
  %newIm2_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_32)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_34)
  %newIm3_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_35)
  %newIm3_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_36)
  %newIm3_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_37)
  %newIm3_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_38)
  %newIm3_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_39)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_23, ptr %pIm0, align 32
  store <4 x double> %newIm1_26, ptr %pIm1, align 32
  store <4 x double> %newIm2_33, ptr %pIm2, align 32
  store <4 x double> %newIm3_40, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l0_0000000004104001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 128
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe1_ = fadd <4 x double> %Re0, %Re3
  %newRe2_ = fadd <4 x double> %newRe1_, %Re2
  %newRe3_ = fadd <4 x double> %newRe2_, %Re1
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1_, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2_, <4 x double> %newRe3_, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %Im3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %Im2, <4 x double> %Im1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l0_fffff0f0ffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 32
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm2, %Im2
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_4)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_5
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_6)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_7)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe0_5)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_9)
  %newRe1 = fsub <4 x double> %newRe1_8, %newRe1_10
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_8)
  %newRe2_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_11)
  %newRe2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_12)
  %newRe2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_10)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_14)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_16)
  %newRe2 = fsub <4 x double> %newRe2_13, %newRe2_17
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_13)
  %newRe3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_18)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_19)
  %newRe3_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_17)
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_21)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_23)
  %newRe3 = fsub <4 x double> %newRe3_20, %newRe3_24
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_25)
  %newIm0_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_26)
  %newIm0_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_27)
  %newIm0_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_28)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_30)
  %newIm1_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_31)
  %newIm1_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_32)
  %newIm1_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_33)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_35)
  %newIm2_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_36)
  %newIm2_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_37)
  %newIm2_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_38)
  %newIm2_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_39)
  %newIm2_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_40)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_42)
  %newIm3_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_43)
  %newIm3_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_44)
  %newIm3_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_45)
  %newIm3_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_46)
  %newIm3_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_47)
  %newReLo = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_29, <4 x double> %newIm1_34, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_41, <4 x double> %newIm3_48, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l1_ff00ff00ffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 64
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_4)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_5)
  %newRe1_7 = fmul <4 x double> %mIm4, %Im0
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_7)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_9)
  %newRe1 = fsub <4 x double> %newRe1_6, %newRe1_10
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_6)
  %newRe2_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_11)
  %newRe2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_12)
  %newRe2 = fsub <4 x double> %newRe2_13, %newRe1_10
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_13)
  %newRe3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_14)
  %newRe3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_15)
  %newRe3_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe1_10)
  %newRe3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_17)
  %newRe3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_18)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_19)
  %newRe3 = fsub <4 x double> %newRe3_16, %newRe3_20
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_21)
  %newIm0_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_22)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_24)
  %newIm1_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_25)
  %newIm1_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_26)
  %newIm1_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_27)
  %newIm1_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_28)
  %newIm1_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_29)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_31)
  %newIm2_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_32)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_34)
  %newIm3_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_35)
  %newIm3_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_36)
  %newIm3_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_37)
  %newIm3_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_38)
  %newIm3_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_39)
  %newReLo = shufflevector <4 x double> %newRe0_3, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_23, <4 x double> %newIm1_30, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_33, <4 x double> %newIm3_40, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l5_ffffffffffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 0
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 7
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 32
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 96
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm0, %Im0
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Im1, <4 x double> %newRe0_4)
  %newRe0_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_5)
  %newRe0_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_6)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_7
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_9)
  %newRe1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_7)
  %newRe1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_11)
  %newRe1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_12)
  %newRe1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_13)
  %newRe1 = fsub <4 x double> %newRe1_10, %newRe1_14
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_10)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_16)
  %newRe2_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_14)
  %newRe2_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_18)
  %newRe2_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_19)
  %newRe2_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_20)
  %newRe2 = fsub <4 x double> %newRe2_17, %newRe2_21
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_17)
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_23)
  %newRe3_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_21)
  %newRe3_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_25)
  %newRe3_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_26)
  %newRe3_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_27)
  %newRe3 = fsub <4 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_29)
  %newIm0_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_30)
  %newIm0_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm0, <4 x double> %Re0, <4 x double> %newIm0_31)
  %newIm0_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_32)
  %newIm0_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_33)
  %newIm0_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_34)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_36)
  %newIm1_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_37)
  %newIm1_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_38)
  %newIm1_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_39)
  %newIm1_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_40)
  %newIm1_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_41)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_43)
  %newIm2_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_44)
  %newIm2_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_45)
  %newIm2_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_46)
  %newIm2_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_47)
  %newIm2_49 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_48)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_50 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_51 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_50)
  %newIm3_52 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_51)
  %newIm3_53 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_52)
  %newIm3_54 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_53)
  %newIm3_55 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_54)
  %newIm3_56 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_55)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_35, ptr %pIm0, align 32
  store <4 x double> %newIm1_42, ptr %pIm1, align 32
  store <4 x double> %newIm2_49, ptr %pIm2, align 32
  store <4 x double> %newIm3_56, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l2_fffcfffcffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 7
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 4
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 68
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm1, %Im1
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_4)
  %newRe0_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_5)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_6
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_7)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_6)
  %newRe1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_10)
  %newRe1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_11)
  %newRe1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_12)
  %newRe1 = fsub <4 x double> %newRe1_9, %newRe1_13
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_9)
  %newRe2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_14)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe1_13)
  %newRe2_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_17)
  %newRe2_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_18)
  %newRe2 = fsub <4 x double> %newRe2_16, %newRe2_19
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_16)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_20)
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_21)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_19)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_23)
  %newRe3_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_24)
  %newRe3_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_25)
  %newRe3 = fsub <4 x double> %newRe3_22, %newRe3_26
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_27)
  %newIm0_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_28)
  %newIm0_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_29)
  %newIm0_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_30)
  %newIm0_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_31)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_33)
  %newIm1_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_34)
  %newIm1_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_35)
  %newIm1_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_36)
  %newIm1_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_37)
  %newIm1_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_38)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_40)
  %newIm2_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_41)
  %newIm2_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_42)
  %newIm2_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_43)
  %newIm2_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_44)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_46)
  %newIm3_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_47)
  %newIm3_49 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_48)
  %newIm3_50 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_49)
  %newIm3_51 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_50)
  %newIm3_52 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_51)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_32, ptr %pIm0, align 32
  store <4 x double> %newIm1_39, ptr %pIm1, align 32
  store <4 x double> %newIm2_45, ptr %pIm2, align 32
  store <4 x double> %newIm3_52, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l1_3c003c003cc33cc3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 64
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_)
  %newRe1_3 = fmul <4 x double> %mIm5, %Im1
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_2, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_2)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe1_4
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_5)
  %newRe3_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe1_4)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_6, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_)
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_10)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_14)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_15)
  %newReLo = shufflevector <4 x double> %newRe0_1, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_9, <4 x double> %newIm1_12, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_13, <4 x double> %newIm3_16, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l2_33ccc03033cccc33(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 7
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 4
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 68
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm2, %Im2
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_2
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_1)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe0_2)
  %newRe1 = fsub <4 x double> %newRe1_3, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe1_3)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe1_4)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_6)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe2_7
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_5)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_7)
  %newRe3_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_9)
  %newRe3 = fsub <4 x double> %newRe3_8, %newRe3_10
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_)
  %newIm0_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_11)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_13)
  %newIm2_ = fmul <4 x double> %mRe9, %Im1
  %newIm2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_15)
  %newIm2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_16)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_18)
  %newIm3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_19)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_12, ptr %pIm0, align 32
  store <4 x double> %newIm1_14, ptr %pIm1, align 32
  store <4 x double> %newIm2_17, ptr %pIm2, align 32
  store <4 x double> %newIm3_20, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l0_00c00c0010c00c01(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 32
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %Re0)
  %newRe1_1 = fmul <4 x double> %mIm5, %Im1
  %newRe1 = fsub <4 x double> %newRe1_, %newRe1_1
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe2_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe1_1)
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_2
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  %newRe3 = fsub <4 x double> %newRe3_, %newRe2_2
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_)
  %newIm2_ = fmul <4 x double> %mRe11, %Im3
  %newIm2_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_)
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %newIm1_3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_4, <4 x double> %Im2, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l0_2080000000000401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fneg <4 x double> %Im3
  %newRe2 = fsub <4 x double> %newRe1_, %newRe2_
  %newRe3_ = fsub <4 x double> %newRe2_, %Im2
  %newRe3 = fsub <4 x double> %newRe1_, %newRe3_
  %newIm2_ = fneg <4 x double> %Re3
  %newIm3_ = fneg <4 x double> %Re2
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1_, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %Im1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_, <4 x double> %newIm3_, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k4l1_0000000010040140(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 16
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe1_ = fadd <4 x double> %Re3, %Re0
  %newRe2_ = fadd <4 x double> %newRe1_, %Re1
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  %newReLo = shufflevector <4 x double> %Re3, <4 x double> %newRe1_, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2_, <4 x double> %newRe3_, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im3, <4 x double> %Im0, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %Im1, <4 x double> %Im2, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l0_ffffff3cffffffc3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm1, %Im1
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_2)
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_3
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_1)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_4)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_5)
  %newRe1_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_3)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_7)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_9)
  %newRe1 = fsub <4 x double> %newRe1_6, %newRe1_10
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_6)
  %newRe2_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_11)
  %newRe2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_12)
  %newRe2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_10)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_14)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_16)
  %newRe2 = fsub <4 x double> %newRe2_13, %newRe2_17
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_13)
  %newRe3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_18)
  %newRe3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_19)
  %newRe3_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_17)
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_21)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_23)
  %newRe3 = fsub <4 x double> %newRe3_20, %newRe3_24
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm0_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_25)
  %newIm0_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_26)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_28)
  %newIm1_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_29)
  %newIm1_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_30)
  %newIm1_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_31)
  %newIm1_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_32)
  %newIm1_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_33)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_35)
  %newIm2_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_36)
  %newIm2_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_37)
  %newIm2_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_38)
  %newIm2_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_39)
  %newIm2_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_40)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_42)
  %newIm3_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_43)
  %newIm3_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_44)
  %newIm3_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_45)
  %newIm3_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_46)
  %newIm3_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_47)
  %newReLo = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_27, <4 x double> %newIm1_34, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_41, <4 x double> %newIm3_48, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l3_3c003c003cc33cc3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 6
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 72
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_)
  %newRe1_3 = fmul <4 x double> %mIm5, %Im1
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_2, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_2)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe1_4
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_5)
  %newRe3_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe1_4)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_6, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_)
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_10)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_14)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_15)
  store <4 x double> %newRe0_1, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_9, ptr %pIm0, align 32
  store <4 x double> %newIm1_12, ptr %pIm1, align 32
  store <4 x double> %newIm2_13, ptr %pIm2, align 32
  store <4 x double> %newIm3_16, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k4l1_ffffffffffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 16
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm0, %Im0
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Im1, <4 x double> %newRe0_4)
  %newRe0_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_5)
  %newRe0_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im3, <4 x double> %newRe0_6)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_7
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_9)
  %newRe1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Im0, <4 x double> %newRe0_7)
  %newRe1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_11)
  %newRe1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_12)
  %newRe1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_13)
  %newRe1 = fsub <4 x double> %newRe1_10, %newRe1_14
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_10)
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_16)
  %newRe2_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_14)
  %newRe2_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_18)
  %newRe2_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_19)
  %newRe2_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_20)
  %newRe2 = fsub <4 x double> %newRe2_17, %newRe2_21
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_17)
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_23)
  %newRe3_25 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_21)
  %newRe3_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_25)
  %newRe3_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_26)
  %newRe3_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_27)
  %newRe3 = fsub <4 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_29)
  %newIm0_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_30)
  %newIm0_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm0, <4 x double> %Re0, <4 x double> %newIm0_31)
  %newIm0_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_32)
  %newIm0_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_33)
  %newIm0_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_34)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_36)
  %newIm1_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_37)
  %newIm1_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_38)
  %newIm1_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_39)
  %newIm1_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_40)
  %newIm1_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_41)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_43)
  %newIm2_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_44)
  %newIm2_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_45)
  %newIm2_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re1, <4 x double> %newIm2_46)
  %newIm2_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_47)
  %newIm2_49 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_48)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_50 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_51 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_50)
  %newIm3_52 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_51)
  %newIm3_53 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_52)
  %newIm3_54 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_53)
  %newIm3_55 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_54)
  %newIm3_56 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_55)
  %newReLo = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_35, <4 x double> %newIm1_42, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_49, <4 x double> %newIm3_56, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u3_k5_33330000(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %ai = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %and_outer = and i64 %idx, -8
  %shl_outer = shl i64 %and_outer, 3
  %and_inner = and i64 %idx, 7
  %shl_inner = shl i64 %and_inner, 2
  %idxA = add i64 %shl_outer, %shl_inner
  %idxB = add i64 %idxA, 32
  %pAr = getelementptr double, ptr %preal, i64 %idxA
  %pAi = getelementptr double, ptr %pimag, i64 %idxA
  %pBr = getelementptr double, ptr %preal, i64 %idxB
  %pBi = getelementptr double, ptr %pimag, i64 %idxB
  %Ar = load <4 x double>, ptr %pAr, align 32
  %Ai = load <4 x double>, ptr %pAi, align 32
  %Br = load <4 x double>, ptr %pBr, align 32
  %Bi = load <4 x double>, ptr %pBi, align 32
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi)
  store <4 x double> %newAr1, ptr %pAr, align 32
  store <4 x double> %newAi2, ptr %pAi, align 32
  store <4 x double> %newBr3, ptr %pBr, align 32
  store <4 x double> %newBi4, ptr %pBi, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l0_0c3000000c304001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 64
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe1_ = fadd <4 x double> %Re0, %Re3
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_)
  %newRe2_1 = fmul <4 x double> %mIm10, %Im2
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_1
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe3_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_1)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_2
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_)
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1_, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %Im3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_3, <4 x double> %newIm3_4, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l1_3cc03cc00cc30cc3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm3, %Im3
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_2
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_1)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe0_2)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe1_4)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe2_6
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_5)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_6)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm0_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_9)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_13)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_15)
  %newReLo = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_10, <4 x double> %newIm1_12, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_14, <4 x double> %newIm3_16, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l4_000000003cc33cc3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 4
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 3
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 16
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 80
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_)
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_2)
  %newRe2_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_3)
  %newRe3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  store <4 x double> %newRe0_1, ptr %pRe0, align 32
  store <4 x double> %newRe1_2, ptr %pRe1, align 32
  store <4 x double> %newRe2_3, ptr %pRe2, align 32
  store <4 x double> %newRe3_4, ptr %pRe3, align 32
  store <4 x double> %newIm0_5, ptr %pIm0, align 32
  store <4 x double> %newIm1_6, ptr %pIm1, align 32
  store <4 x double> %newIm2_7, ptr %pIm2, align 32
  store <4 x double> %newIm3_8, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l1_0c00c0000c10c001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %Re0)
  %newRe1_1 = fmul <4 x double> %mIm7, %Im3
  %newRe1 = fsub <4 x double> %newRe1_, %newRe1_1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re2
  %newRe2 = fsub <4 x double> %newRe2_, %newRe1_1
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe3_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe1_1)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_2
  %newIm1_ = fmul <4 x double> %mRe7, %Im3
  %newIm1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_)
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %newIm1_3, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %Im2, <4 x double> %newIm3_4, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l1_0030c0000430c001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %Re0)
  %newRe1_1 = fmul <4 x double> %mIm7, %Im3
  %newRe1 = fsub <4 x double> %newRe1_, %newRe1_1
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_)
  %newRe2_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe1_1)
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_2
  %newRe3_ = fadd <4 x double> %newRe2_, %Re1
  %newRe3 = fsub <4 x double> %newRe3_, %newRe2_2
  %newIm1_ = fmul <4 x double> %mRe7, %Im3
  %newIm1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_)
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %newIm1_3, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_4, <4 x double> %Im1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l3_30c0000030c00401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 6
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 72
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe2_1 = fmul <4 x double> %mIm11, %Im3
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_1
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe3_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe2_1)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_2
  %newIm2_ = fmul <4 x double> %mRe11, %Im3
  %newIm2_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe14, %Im2
  %newIm3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_)
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im1, ptr %pIm1, align 32
  store <4 x double> %newIm2_3, ptr %pIm2, align 32
  store <4 x double> %newIm3_4, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l3_0000000010400401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 6
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 72
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re3
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2_, ptr %pRe2, align 32
  store <4 x double> %newRe3_, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im1, ptr %pIm1, align 32
  store <4 x double> %Im3, ptr %pIm2, align 32
  store <4 x double> %Im2, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k4l1_0000000010400401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 16
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re3
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1_, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2_, <4 x double> %newRe3_, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %Im1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %Im3, <4 x double> %Im2, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k1l0_0fc0f00c0ff0f00f(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pReal = getelementptr <16 x double>, ptr %preal, i64 %idx
  %pImag = getelementptr <16 x double>, ptr %pimag, i64 %idx
  %Real = load <16 x double>, ptr %pReal, align 128
  %Imag = load <16 x double>, ptr %pImag, align 128
  %Re0 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %Re1 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %Re2 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %Re3 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %Im0 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 0, i32 4, i32 8, i32 12>
  %Im1 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 1, i32 5, i32 9, i32 13>
  %Im2 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 2, i32 6, i32 10, i32 14>
  %Im3 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 3, i32 7, i32 11, i32 15>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm1, %Im1
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_2
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe0_2)
  %newRe1_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_4)
  %newRe1 = fsub <4 x double> %newRe1_3, %newRe1_5
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_3)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe1_5)
  %newRe2 = fsub <4 x double> %newRe2_6, %newRe2_7
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_6)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_7)
  %newRe3_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_9)
  %newRe3 = fsub <4 x double> %newRe3_8, %newRe3_10
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_11)
  %newIm1_ = fmul <4 x double> %mRe6, %Im2
  %newIm1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_13)
  %newIm1_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_14)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_16)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_18)
  %newIm3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_19)
  %vecRe0 = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecRe1 = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newReal = shufflevector <8 x double> %vecRe0, <8 x double> %vecRe1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x double> %newReal, ptr %pReal, align 128
  %vecIm0 = shufflevector <4 x double> %newIm0_12, <4 x double> %newIm1_15, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecIm1 = shufflevector <4 x double> %newIm2_17, <4 x double> %newIm3_20, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newImag = shufflevector <8 x double> %vecIm0, <8 x double> %vecIm1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x double> %newImag, ptr %pImag, align 128
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l3_0000000004104001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 14
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 136
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re3
  %newRe2_ = fadd <4 x double> %newRe1_, %Re2
  %newRe3_ = fadd <4 x double> %newRe2_, %Re1
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2_, ptr %pRe2, align 32
  store <4 x double> %newRe3_, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im3, ptr %pIm1, align 32
  store <4 x double> %Im2, ptr %pIm2, align 32
  store <4 x double> %Im1, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l1_0000000010400401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 64
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re3
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1_, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2_, <4 x double> %newRe3_, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %Im1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %Im3, <4 x double> %Im2, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l1_0000000010400401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re3
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1_, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2_, <4 x double> %newRe3_, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %Im1, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newImHi = shufflevector <4 x double> %Im3, <4 x double> %Im2, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l4_0030c0000430c001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 12
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 3
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 16
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 144
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %Re0)
  %newRe1_1 = fmul <4 x double> %mIm7, %Im3
  %newRe1 = fsub <4 x double> %newRe1_, %newRe1_1
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_)
  %newRe2_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe1_1)
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_2
  %newRe3_ = fadd <4 x double> %newRe2_, %Re1
  %newRe3 = fsub <4 x double> %newRe3_, %newRe2_2
  %newIm1_ = fmul <4 x double> %mRe7, %Im3
  %newIm1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_)
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %newIm1_3, ptr %pIm1, align 32
  store <4 x double> %newIm2_4, ptr %pIm2, align 32
  store <4 x double> %Im1, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l3_3c00c3003c3cc3c3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 2
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 40
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_3 = fmul <4 x double> %mIm4, %Im0
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_2, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe1_2)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe1_4
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_5)
  %newRe3_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe1_4)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_6, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re0, <4 x double> %newIm1_10)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe9, %Im1
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_14)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_15)
  store <4 x double> %newRe0_1, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_9, ptr %pIm0, align 32
  store <4 x double> %newIm1_12, ptr %pIm1, align 32
  store <4 x double> %newIm2_13, ptr %pIm2, align 32
  store <4 x double> %newIm3_16, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l5_00000000cc33cc33(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 0
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 7
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 32
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 96
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_2)
  %newRe2_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_3)
  %newRe3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_)
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_)
  store <4 x double> %newRe0_1, ptr %pRe0, align 32
  store <4 x double> %newRe1_2, ptr %pRe1, align 32
  store <4 x double> %newRe2_3, ptr %pRe2, align 32
  store <4 x double> %newRe3_4, ptr %pRe3, align 32
  store <4 x double> %newIm0_5, ptr %pIm0, align 32
  store <4 x double> %newIm1_6, ptr %pIm1, align 32
  store <4 x double> %newIm2_7, ptr %pIm2, align 32
  store <4 x double> %newIm3_8, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l3_0000000010400401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 2
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 40
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re3
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2_, ptr %pRe2, align 32
  store <4 x double> %newRe3_, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im1, ptr %pIm1, align 32
  store <4 x double> %Im3, ptr %pIm2, align 32
  store <4 x double> %Im2, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l5_cc03cc30cc30cc03(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 8
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 7
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 32
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 160
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = fmul <4 x double> %mIm2, %Im2
  %newRe0 = fsub <4 x double> %newRe0_, %newRe0_1
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe0_1)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_2, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_2)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_4)
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_5
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe3_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_5)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_6, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_10)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_)
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_14)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_15)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_9, ptr %pIm0, align 32
  store <4 x double> %newIm1_12, ptr %pIm1, align 32
  store <4 x double> %newIm2_13, ptr %pIm2, align 32
  store <4 x double> %newIm3_16, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l0_0c00c0000c10c001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %Re0)
  %newRe1_1 = fmul <4 x double> %mIm7, %Im3
  %newRe1 = fsub <4 x double> %newRe1_, %newRe1_1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re2
  %newRe2 = fsub <4 x double> %newRe2_, %newRe1_1
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe3_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe1_1)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_2
  %newIm1_ = fmul <4 x double> %mRe7, %Im3
  %newIm1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_)
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %newIm1_3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %Im2, <4 x double> %newIm3_4, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l2_0000000010400401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 3
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 4
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 36
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re3
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2_, ptr %pRe2, align 32
  store <4 x double> %newRe3_, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im1, ptr %pIm1, align 32
  store <4 x double> %Im3, ptr %pIm2, align 32
  store <4 x double> %Im2, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l0_00f0f0000ff0f00f(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_3 = fmul <4 x double> %mIm6, %Im2
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_2, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_2)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe1_4)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_6)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe2_7
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_5)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3 = fsub <4 x double> %newRe3_8, %newRe2_7
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe6, %Im2
  %newIm1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_10)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_13)
  %newIm2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_14)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newReLo = shufflevector <4 x double> %newRe0_1, <4 x double> %newRe1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_9, <4 x double> %newIm1_12, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_15, <4 x double> %newIm3_16, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l4_00000000ffffffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 4
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 3
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 16
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 80
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_2)
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_3)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_4)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_5)
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_6)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_7)
  %newRe2_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_8)
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_9)
  %newRe3_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_10)
  %newRe3_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_11)
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im2, <4 x double> %newIm0_13)
  %newIm0_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_14)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im1, <4 x double> %newIm1_)
  %newIm1_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_16)
  %newIm1_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_17)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im1, <4 x double> %newIm2_)
  %newIm2_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_19)
  %newIm2_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_20)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_22)
  %newIm3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_23)
  store <4 x double> %newRe0_3, ptr %pRe0, align 32
  store <4 x double> %newRe1_6, ptr %pRe1, align 32
  store <4 x double> %newRe2_9, ptr %pRe2, align 32
  store <4 x double> %newRe3_12, ptr %pRe3, align 32
  store <4 x double> %newIm0_15, ptr %pIm0, align 32
  store <4 x double> %newIm1_18, ptr %pIm1, align 32
  store <4 x double> %newIm2_21, ptr %pIm2, align 32
  store <4 x double> %newIm3_24, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k4l2_cc30cc30cc03cc03(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -2
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 1
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 4
  %idx2 = or i64 %idx0, 16
  %idx3 = or i64 %idx0, 20
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = fmul <4 x double> %mIm2, %Im2
  %newRe0 = fsub <4 x double> %newRe0_, %newRe0_1
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe0_1)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_3)
  %newRe1 = fsub <4 x double> %newRe1_2, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe8, <4 x double> %Re0, <4 x double> %newRe1_2)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe1_4)
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_5
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe3_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_5)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_6, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_10)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe8, %Im0
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im3, <4 x double> %newIm3_)
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_14)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_15)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_9, ptr %pIm0, align 32
  store <4 x double> %newIm1_12, ptr %pIm1, align 32
  store <4 x double> %newIm2_13, ptr %pIm2, align 32
  store <4 x double> %newIm3_16, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k3l0_0fc0f00c0ff0f00f(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 8
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re1, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm1, %Im1
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_2
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe0_1)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe0_2)
  %newRe1_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_4)
  %newRe1 = fsub <4 x double> %newRe1_3, %newRe1_5
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_3)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe1_5)
  %newRe2 = fsub <4 x double> %newRe2_6, %newRe2_7
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe12, <4 x double> %Re0, <4 x double> %newRe2_6)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_7)
  %newRe3_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe3_9)
  %newRe3 = fsub <4 x double> %newRe3_8, %newRe3_10
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im1, <4 x double> %newIm0_)
  %newIm0_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_11)
  %newIm1_ = fmul <4 x double> %mRe6, %Im2
  %newIm1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_13)
  %newIm1_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_14)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im3, <4 x double> %newIm2_)
  %newIm2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_16)
  %newIm3_ = fmul <4 x double> %mRe12, %Im0
  %newIm3_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im1, <4 x double> %newIm3_)
  %newIm3_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_18)
  %newIm3_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_19)
  %newReLo = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %newIm0_12, <4 x double> %newIm1_15, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_17, <4 x double> %newIm3_20, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l3_0000000010400401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 14
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 136
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re3
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2_, ptr %pRe2, align 32
  store <4 x double> %newRe3_, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im1, ptr %pIm1, align 32
  store <4 x double> %Im3, ptr %pIm2, align 32
  store <4 x double> %Im2, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l6_0000000010400401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 0
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 15
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 64
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 192
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re3
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2_, ptr %pRe2, align 32
  store <4 x double> %newRe3_, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im1, ptr %pIm1, align 32
  store <4 x double> %Im3, ptr %pIm2, align 32
  store <4 x double> %Im2, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k2l1_0000000010400401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pReal = getelementptr <16 x double>, ptr %preal, i64 %idx
  %pImag = getelementptr <16 x double>, ptr %pimag, i64 %idx
  %Real = load <16 x double>, ptr %pReal, align 128
  %Imag = load <16 x double>, ptr %pImag, align 128
  %Re0 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 0, i32 1, i32 8, i32 9>
  %Re1 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 2, i32 3, i32 10, i32 11>
  %Re2 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 4, i32 5, i32 12, i32 13>
  %Re3 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 6, i32 7, i32 14, i32 15>
  %Im0 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 0, i32 1, i32 8, i32 9>
  %Im1 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 2, i32 3, i32 10, i32 11>
  %Im2 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 4, i32 5, i32 12, i32 13>
  %Im3 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 6, i32 7, i32 14, i32 15>
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = fadd <4 x double> %newRe1_, %Re3
  %newRe3_ = fadd <4 x double> %newRe2_, %Re2
  %vecRe0 = shufflevector <4 x double> %Re0, <4 x double> %newRe1_, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecRe1 = shufflevector <4 x double> %newRe2_, <4 x double> %newRe3_, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newReal = shufflevector <8 x double> %vecRe0, <8 x double> %vecRe1, <16 x i32> <i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13, i32 2, i32 3, i32 6, i32 7, i32 10, i32 11, i32 14, i32 15>
  store <16 x double> %newReal, ptr %pReal, align 128
  %vecIm0 = shufflevector <4 x double> %Im0, <4 x double> %Im1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecIm1 = shufflevector <4 x double> %Im3, <4 x double> %Im2, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newImag = shufflevector <8 x double> %vecIm0, <8 x double> %vecIm1, <16 x i32> <i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13, i32 2, i32 3, i32 6, i32 7, i32 10, i32 11, i32 14, i32 15>
  store <16 x double> %newImag, ptr %pImag, align 128
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k6l3_0000000004104001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -8
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 6
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 64
  %idx3 = or i64 %idx0, 72
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re3
  %newRe2_ = fadd <4 x double> %newRe1_, %Re2
  %newRe3_ = fadd <4 x double> %newRe2_, %Re1
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2_, ptr %pRe2, align 32
  store <4 x double> %newRe3_, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im3, ptr %pIm1, align 32
  store <4 x double> %Im2, ptr %pIm2, align 32
  store <4 x double> %Im1, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k7l5_0000000004104001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -16
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 8
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 7
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 32
  %idx2 = or i64 %idx0, 128
  %idx3 = or i64 %idx0, 160
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe1_ = fadd <4 x double> %Re0, %Re3
  %newRe2_ = fadd <4 x double> %newRe1_, %Re2
  %newRe3_ = fadd <4 x double> %newRe2_, %Re1
  store <4 x double> %Re0, ptr %pRe0, align 32
  store <4 x double> %newRe1_, ptr %pRe1, align 32
  store <4 x double> %newRe2_, ptr %pRe2, align 32
  store <4 x double> %newRe3_, ptr %pRe3, align 32
  store <4 x double> %Im0, ptr %pIm0, align 32
  store <4 x double> %Im3, ptr %pIm1, align 32
  store <4 x double> %Im2, ptr %pIm2, align 32
  store <4 x double> %Im1, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l3_3cc330c030c03cc3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 2
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 1
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 40
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm3, %Im3
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_2
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe0_1)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe0_2)
  %newRe1 = fsub <4 x double> %newRe1_3, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe1_3)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_4)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_5)
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_6
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im1, <4 x double> %newRe2_6)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm0_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re3, <4 x double> %newIm0_9)
  %newIm1_ = fmul <4 x double> %mRe5, %Im1
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im2, <4 x double> %newIm1_)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe11, %Im3
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_)
  %newIm2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_13)
  %newIm3_ = fmul <4 x double> %mRe14, %Im2
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re1, <4 x double> %newIm3_)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_15)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_10, ptr %pIm0, align 32
  store <4 x double> %newIm1_12, ptr %pIm1, align 32
  store <4 x double> %newIm2_14, ptr %pIm2, align 32
  store <4 x double> %newIm3_16, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k4l0_30c0000030c00401(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 16
  %pReLo = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <8 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %pImHi = getelementptr <8 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <8 x double>, ptr %pReLo, align 64
  %ReHi = load <8 x double>, ptr %pReHi, align 64
  %ImLo = load <8 x double>, ptr %pImLo, align 64
  %ImHi = load <8 x double>, ptr %pImHi, align 64
  %Re0 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re1 = shufflevector <8 x double> %ReLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Re2 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Re3 = shufflevector <8 x double> %ReHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im0 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im1 = shufflevector <8 x double> %ImLo, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Im2 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Im3 = shufflevector <8 x double> %ImHi, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newRe1_ = fadd <4 x double> %Re0, %Re1
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe2_1 = fmul <4 x double> %mIm11, %Im3
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_1
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe3_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im2, <4 x double> %newRe2_1)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_2
  %newIm2_ = fmul <4 x double> %mRe11, %Im3
  %newIm2_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe14, %Im2
  %newIm3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re2, <4 x double> %newIm3_)
  %newReLo = shufflevector <4 x double> %Re0, <4 x double> %newRe1_, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newReHi = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newReLo, ptr %pReLo, align 64
  store <8 x double> %newReHi, ptr %pReHi, align 64
  %newImLo = shufflevector <4 x double> %Im0, <4 x double> %Im1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newImHi = shufflevector <4 x double> %newIm2_3, <4 x double> %newIm3_4, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newImLo, ptr %pImLo, align 64
  store <8 x double> %newImHi, ptr %pImHi, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k5l2_c0030c300c30c003(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -4
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 3
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 4
  %idx2 = or i64 %idx0, 32
  %idx3 = or i64 %idx0, 36
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = fmul <4 x double> %mIm2, %Im2
  %newRe0 = fsub <4 x double> %newRe0_, %newRe0_1
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe1_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe0_1)
  %newRe1 = fsub <4 x double> %newRe1_, %newRe1_2
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe1_)
  %newRe2_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_2)
  %newRe2 = fsub <4 x double> %newRe2_, %newRe2_3
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe3_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe2_3)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_4
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_)
  %newIm1_ = fmul <4 x double> %mRe7, %Im3
  %newIm1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_)
  %newIm2_ = fmul <4 x double> %mRe10, %Im2
  %newIm2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_5, ptr %pIm0, align 32
  store <4 x double> %newIm1_6, ptr %pIm1, align 32
  store <4 x double> %newIm2_7, ptr %pIm2, align 32
  store <4 x double> %newIm3_8, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k2l1_c330c03cc03cc330(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pReal = getelementptr <16 x double>, ptr %preal, i64 %idx
  %pImag = getelementptr <16 x double>, ptr %pimag, i64 %idx
  %Real = load <16 x double>, ptr %pReal, align 128
  %Imag = load <16 x double>, ptr %pImag, align 128
  %Re0 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 0, i32 1, i32 8, i32 9>
  %Re1 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 2, i32 3, i32 10, i32 11>
  %Re2 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 4, i32 5, i32 12, i32 13>
  %Re3 = shufflevector <16 x double> %Real, <16 x double> poison, <4 x i32> <i32 6, i32 7, i32 14, i32 15>
  %Im0 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 0, i32 1, i32 8, i32 9>
  %Im1 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 2, i32 3, i32 10, i32 11>
  %Im2 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 4, i32 5, i32 12, i32 13>
  %Im3 = shufflevector <16 x double> %Imag, <16 x double> poison, <4 x i32> <i32 6, i32 7, i32 14, i32 15>
  %newRe0_ = fmul <4 x double> %mRe2, %Re2
  %newRe0_1 = fmul <4 x double> %mIm1, %Im1
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_1)
  %newRe0 = fsub <4 x double> %newRe0_, %newRe0_2
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_)
  %newRe1_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe0_2)
  %newRe1 = fsub <4 x double> %newRe1_3, %newRe1_4
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe1_3)
  %newRe2_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe2_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe1_4)
  %newRe2 = fsub <4 x double> %newRe2_5, %newRe2_6
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe2_5)
  %newRe3_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_6)
  %newRe3_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_7)
  %newRe3 = fsub <4 x double> %newRe3_, %newRe3_8
  %newIm0_ = fmul <4 x double> %mRe2, %Im2
  %newIm0_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_)
  %newIm0_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_9)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re3, <4 x double> %newIm1_11)
  %newIm2_ = fmul <4 x double> %mRe9, %Im1
  %newIm2_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_)
  %newIm2_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re2, <4 x double> %newIm2_13)
  %newIm3_ = fmul <4 x double> %mRe15, %Im3
  %newIm3_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_)
  %newIm3_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_15)
  %vecRe0 = shufflevector <4 x double> %newRe0, <4 x double> %newRe1, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecRe1 = shufflevector <4 x double> %newRe2, <4 x double> %newRe3, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newReal = shufflevector <8 x double> %vecRe0, <8 x double> %vecRe1, <16 x i32> <i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13, i32 2, i32 3, i32 6, i32 7, i32 10, i32 11, i32 14, i32 15>
  store <16 x double> %newReal, ptr %pReal, align 128
  %vecIm0 = shufflevector <4 x double> %newIm0_10, <4 x double> %newIm1_12, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecIm1 = shufflevector <4 x double> %newIm2_14, <4 x double> %newIm3_16, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newImag = shufflevector <8 x double> %vecIm0, <8 x double> %vecIm1, <16 x i32> <i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13, i32 2, i32 3, i32 6, i32 7, i32 10, i32 11, i32 14, i32 15>
  store <16 x double> %newImag, ptr %pImag, align 128
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_s2_sep_u2q_k4l2_c3c33c3c3c3cc3c3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -2
  %left = shl i64 %left_tmp, 4
  %middle_tmp = and i64 %idx, 1
  %middle = shl i64 %middle_tmp, 3
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 2
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 4
  %idx2 = or i64 %idx0, 16
  %idx3 = or i64 %idx0, 20
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <4 x double>, ptr %pRe0, align 32
  %Re1 = load <4 x double>, ptr %pRe1, align 32
  %Re2 = load <4 x double>, ptr %pRe2, align 32
  %Re3 = load <4 x double>, ptr %pRe3, align 32
  %Im0 = load <4 x double>, ptr %pIm0, align 32
  %Im1 = load <4 x double>, ptr %pIm1, align 32
  %Im2 = load <4 x double>, ptr %pIm2, align 32
  %Im3 = load <4 x double>, ptr %pIm3, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re0
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re3, <4 x double> %newRe0_)
  %newRe0_2 = fmul <4 x double> %mIm1, %Im1
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im2, <4 x double> %newRe0_2)
  %newRe0 = fsub <4 x double> %newRe0_1, %newRe0_3
  %newRe1_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe4, <4 x double> %Re0, <4 x double> %newRe0_1)
  %newRe1_4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_)
  %newRe1_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe0_3)
  %newRe1_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_5)
  %newRe1 = fsub <4 x double> %newRe1_4, %newRe1_6
  %newRe2_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe1_4)
  %newRe2_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_)
  %newRe2_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Im0, <4 x double> %newRe1_6)
  %newRe2_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_8)
  %newRe2 = fsub <4 x double> %newRe2_7, %newRe2_9
  %newRe3_ = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe2_7)
  %newRe3_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_)
  %newRe3_11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Im0, <4 x double> %newRe2_9)
  %newRe3_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im3, <4 x double> %newRe3_11)
  %newRe3 = fsub <4 x double> %newRe3_10, %newRe3_12
  %newIm0_ = fmul <4 x double> %mRe0, %Im0
  %newIm0_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im3, <4 x double> %newIm0_)
  %newIm0_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re1, <4 x double> %newIm0_13)
  %newIm0_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re2, <4 x double> %newIm0_14)
  %newIm1_ = fmul <4 x double> %mRe4, %Im0
  %newIm1_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im3, <4 x double> %newIm1_)
  %newIm1_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re1, <4 x double> %newIm1_16)
  %newIm1_18 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re2, <4 x double> %newIm1_17)
  %newIm2_ = fmul <4 x double> %mRe9, %Im1
  %newIm2_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im2, <4 x double> %newIm2_)
  %newIm2_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re0, <4 x double> %newIm2_19)
  %newIm2_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re3, <4 x double> %newIm2_20)
  %newIm3_ = fmul <4 x double> %mRe13, %Im1
  %newIm3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im2, <4 x double> %newIm3_)
  %newIm3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re0, <4 x double> %newIm3_22)
  %newIm3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re3, <4 x double> %newIm3_23)
  store <4 x double> %newRe0, ptr %pRe0, align 32
  store <4 x double> %newRe1, ptr %pRe1, align 32
  store <4 x double> %newRe2, ptr %pRe2, align 32
  store <4 x double> %newRe3, ptr %pRe3, align 32
  store <4 x double> %newIm0_15, ptr %pIm0, align 32
  store <4 x double> %newIm1_18, ptr %pIm1, align 32
  store <4 x double> %newIm2_21, ptr %pIm2, align 32
  store <4 x double> %newIm3_24, ptr %pIm3, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
