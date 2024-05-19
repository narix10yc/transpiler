; ModuleID = 'myModule'
source_filename = "myModule"

define void @f64_s2_sep_u2q_k1l0(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
  %newReal = shufflevector <8 x double> %vecRe0, <8 x double> %vecRe1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x double> %newReal, ptr %pReal, align 128
  %vecIm0 = shufflevector <4 x double> %newIm0_35, <4 x double> %newIm1_42, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %vecIm1 = shufflevector <4 x double> %newIm2_49, <4 x double> %newIm3_56, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %newImag = shufflevector <8 x double> %vecIm0, <8 x double> %vecIm1, <16 x i32> <i32 0, i32 4, i32 8, i32 12, i32 1, i32 5, i32 9, i32 13, i32 2, i32 6, i32 10, i32 14, i32 3, i32 7, i32 11, i32 15>
  store <16 x double> %newImag, ptr %pImag, align 128
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
