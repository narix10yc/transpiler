; ModuleID = '../performance/gen_file.ll'
source_filename = "myModule"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s1_sep_u2q_k2l1(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 31, i32 31>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx4 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %left = shl i64 %idx4, 3
  %idx1 = or i64 %left, 2
  %idx2 = or i64 %left, 4
  %idx3 = or i64 %left, 6
  %pRe0 = getelementptr double, ptr %preal, i64 %left
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %left
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %Re0 = load <2 x double>, ptr %pRe0, align 16
  %Re1 = load <2 x double>, ptr %pRe1, align 16
  %Re2 = load <2 x double>, ptr %pRe2, align 16
  %Re3 = load <2 x double>, ptr %pRe3, align 16
  %Im0 = load <2 x double>, ptr %pIm0, align 16
  %Im1 = load <2 x double>, ptr %pIm1, align 16
  %Im2 = load <2 x double>, ptr %pIm2, align 16
  %Im3 = load <2 x double>, ptr %pIm3, align 16
  %newRe0_ = fmul <2 x double> %mRe0, %Re0
  %newRe0_1 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Re1, <2 x double> %newRe0_)
  %newRe0_2 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Re2, <2 x double> %newRe0_1)
  %newRe0_3 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Re3, <2 x double> %newRe0_2)
  %newRe0_4 = fmul <2 x double> %mIm0, %Im0
  %newRe0_5 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Im1, <2 x double> %newRe0_4)
  %newRe0_6 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Im2, <2 x double> %newRe0_5)
  %newRe0_7 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Im3, <2 x double> %newRe0_6)
  %newRe0 = fsub <2 x double> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <2 x double> %mRe4, %Re0
  %newRe1_8 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Re1, <2 x double> %newRe1_)
  %newRe1_9 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Re2, <2 x double> %newRe1_8)
  %newRe1_10 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Re3, <2 x double> %newRe1_9)
  %newRe1_11 = fmul <2 x double> %mIm4, %Im0
  %newRe1_12 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Im1, <2 x double> %newRe1_11)
  %newRe1_13 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Im2, <2 x double> %newRe1_12)
  %newRe1_14 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Im3, <2 x double> %newRe1_13)
  %newRe1 = fsub <2 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <2 x double> %mRe8, %Re0
  %newRe2_15 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Re1, <2 x double> %newRe2_)
  %newRe2_16 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Re2, <2 x double> %newRe2_15)
  %newRe2_17 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Re3, <2 x double> %newRe2_16)
  %newRe2_18 = fmul <2 x double> %mIm8, %Im0
  %newRe2_19 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Im1, <2 x double> %newRe2_18)
  %newRe2_20 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Im2, <2 x double> %newRe2_19)
  %newRe2_21 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Im3, <2 x double> %newRe2_20)
  %newRe2 = fsub <2 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <2 x double> %mRe12, %Re0
  %newRe3_22 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Re1, <2 x double> %newRe3_)
  %newRe3_23 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Re2, <2 x double> %newRe3_22)
  %newRe3_24 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Re3, <2 x double> %newRe3_23)
  %newRe3_25 = fmul <2 x double> %mIm12, %Im0
  %newRe3_26 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Im1, <2 x double> %newRe3_25)
  %newRe3_27 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Im2, <2 x double> %newRe3_26)
  %newRe3_28 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Im3, <2 x double> %newRe3_27)
  %newRe3 = fsub <2 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <2 x double> %mRe0, %Im0
  %newIm0_29 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Im1, <2 x double> %newIm0_)
  %newIm0_30 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Im2, <2 x double> %newIm0_29)
  %newIm0_31 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Im3, <2 x double> %newIm0_30)
  %newIm0_32 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm0, <2 x double> %Re0, <2 x double> %newIm0_31)
  %newIm0_33 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Re1, <2 x double> %newIm0_32)
  %newIm0_34 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Re2, <2 x double> %newIm0_33)
  %newIm0_35 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Re3, <2 x double> %newIm0_34)
  %newIm1_ = fmul <2 x double> %mRe4, %Im0
  %newIm1_36 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Im1, <2 x double> %newIm1_)
  %newIm1_37 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Im2, <2 x double> %newIm1_36)
  %newIm1_38 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Im3, <2 x double> %newIm1_37)
  %newIm1_39 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm4, <2 x double> %Re0, <2 x double> %newIm1_38)
  %newIm1_40 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Re1, <2 x double> %newIm1_39)
  %newIm1_41 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Re2, <2 x double> %newIm1_40)
  %newIm1_42 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Re3, <2 x double> %newIm1_41)
  %newIm2_ = fmul <2 x double> %mRe8, %Im0
  %newIm2_43 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Im1, <2 x double> %newIm2_)
  %newIm2_44 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Im2, <2 x double> %newIm2_43)
  %newIm2_45 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Im3, <2 x double> %newIm2_44)
  %newIm2_46 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm8, <2 x double> %Re0, <2 x double> %newIm2_45)
  %newIm2_47 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Re1, <2 x double> %newIm2_46)
  %newIm2_48 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Re2, <2 x double> %newIm2_47)
  %newIm2_49 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Re3, <2 x double> %newIm2_48)
  %newIm3_ = fmul <2 x double> %mRe12, %Im0
  %newIm3_50 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Im1, <2 x double> %newIm3_)
  %newIm3_51 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Im2, <2 x double> %newIm3_50)
  %newIm3_52 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Im3, <2 x double> %newIm3_51)
  %newIm3_53 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm12, <2 x double> %Re0, <2 x double> %newIm3_52)
  %newIm3_54 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Re1, <2 x double> %newIm3_53)
  %newIm3_55 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Re2, <2 x double> %newIm3_54)
  %newIm3_56 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Re3, <2 x double> %newIm3_55)
  store <2 x double> %newRe0, ptr %pRe0, align 16
  store <2 x double> %newRe1, ptr %pRe1, align 16
  store <2 x double> %newRe2, ptr %pRe2, align 16
  store <2 x double> %newRe3, ptr %pRe3, align 16
  store <2 x double> %newIm0_35, ptr %pIm0, align 16
  store <2 x double> %newIm1_42, ptr %pIm1, align 16
  store <2 x double> %newIm2_49, ptr %pIm2, align 16
  store <2 x double> %newIm3_56, ptr %pIm3, align 16
  %idx_next = add nsw i64 %idx4, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s1_sep_u2q_k2l1_batched(ptr nocapture readonly %preal, ptr nocapture readonly %pimag, ptr nocapture writeonly %preal_another, ptr nocapture writeonly %pimag_another, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
global_entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 3, i32 3>
  %mRe16 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 16, i32 16>
  %mRe17 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 17, i32 17>
  %mRe18 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 18, i32 18>
  %mRe19 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 19, i32 19>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody_batch_0, label %entry_batch_1

loopBody_batch_0:                                 ; preds = %global_entry, %loopBody_batch_0
  %idx4 = phi i64 [ %idx_next, %loopBody_batch_0 ], [ %idx_start, %global_entry ]
  %left = shl i64 %idx4, 3
  %idx1 = or i64 %left, 2
  %idx2 = or i64 %left, 4
  %idx3 = or i64 %left, 6
  %pRe0 = getelementptr double, ptr %preal, i64 %left
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %left
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %pReAnother0 = getelementptr double, ptr %preal_another, i64 %left
  %pImAnother0 = getelementptr double, ptr %pimag_another, i64 %left
  %Re0 = load <2 x double>, ptr %pRe0, align 16
  %Re1 = load <2 x double>, ptr %pRe1, align 16
  %Re2 = load <2 x double>, ptr %pRe2, align 16
  %Re3 = load <2 x double>, ptr %pRe3, align 16
  %Im0 = load <2 x double>, ptr %pIm0, align 16
  %Im1 = load <2 x double>, ptr %pIm1, align 16
  %Im2 = load <2 x double>, ptr %pIm2, align 16
  %Im3 = load <2 x double>, ptr %pIm3, align 16
  %newRe0_ = fmul <2 x double> %mRe0, %Re0
  %newRe0_1 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Re1, <2 x double> %newRe0_)
  %newRe0_2 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Re2, <2 x double> %newRe0_1)
  %newRe0_3 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Re3, <2 x double> %newRe0_2)
  %newRe0_4 = fmul <2 x double> %mRe16, %Im0
  %newRe0_5 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe17, <2 x double> %Im1, <2 x double> %newRe0_4)
  %newRe0_6 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe18, <2 x double> %Im2, <2 x double> %newRe0_5)
  %newRe0_7 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe19, <2 x double> %Im3, <2 x double> %newRe0_6)
  %newRe0 = fsub <2 x double> %newRe0_3, %newRe0_7
  %newIm0_ = fmul <2 x double> %mRe0, %Im0
  %newIm0_8 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Im1, <2 x double> %newIm0_)
  %newIm0_9 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Im2, <2 x double> %newIm0_8)
  %newIm0_10 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Im3, <2 x double> %newIm0_9)
  %newIm0_11 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe16, <2 x double> %Re0, <2 x double> %newIm0_10)
  %newIm0_12 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe17, <2 x double> %Re1, <2 x double> %newIm0_11)
  %newIm0_13 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe18, <2 x double> %Re2, <2 x double> %newIm0_12)
  %newIm0_14 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe19, <2 x double> %Re3, <2 x double> %newIm0_13)
  store <2 x double> %newRe0, ptr %pReAnother0, align 16
  store <2 x double> %newIm0_14, ptr %pImAnother0, align 16
  %idx_next = add nsw i64 %idx4, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %entry_batch_1.loopexit, label %loopBody_batch_0

entry_batch_1.loopexit:                           ; preds = %loopBody_batch_0
  %mat15.pre = load <32 x double>, ptr %pmat, align 256
  br label %entry_batch_1

entry_batch_1:                                    ; preds = %entry_batch_1.loopexit, %global_entry
  %mat15 = phi <32 x double> [ %mat15.pre, %entry_batch_1.loopexit ], [ %mat, %global_entry ]
  %mRe4 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 7, i32 7>
  %mRe20 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 20, i32 20>
  %mRe21 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 21, i32 21>
  %mRe22 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 22, i32 22>
  %mRe23 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 23, i32 23>
  br i1 %cond1, label %loopBody_batch_1, label %entry_batch_2

loopBody_batch_1:                                 ; preds = %entry_batch_1, %loopBody_batch_1
  %idx166 = phi i64 [ %idx_next67, %loopBody_batch_1 ], [ %idx_start, %entry_batch_1 ]
  %left19 = shl i64 %idx166, 3
  %idx126 = or i64 %left19, 2
  %idx227 = or i64 %left19, 4
  %idx328 = or i64 %left19, 6
  %pRe029 = getelementptr double, ptr %preal, i64 %left19
  %pRe130 = getelementptr double, ptr %preal, i64 %idx126
  %pRe231 = getelementptr double, ptr %preal, i64 %idx227
  %pRe332 = getelementptr double, ptr %preal, i64 %idx328
  %pIm033 = getelementptr double, ptr %pimag, i64 %left19
  %pIm134 = getelementptr double, ptr %pimag, i64 %idx126
  %pIm235 = getelementptr double, ptr %pimag, i64 %idx227
  %pIm336 = getelementptr double, ptr %pimag, i64 %idx328
  %pReAnother138 = getelementptr double, ptr %preal_another, i64 %idx126
  %pImAnother142 = getelementptr double, ptr %pimag_another, i64 %idx126
  %Re045 = load <2 x double>, ptr %pRe029, align 16
  %Re146 = load <2 x double>, ptr %pRe130, align 16
  %Re247 = load <2 x double>, ptr %pRe231, align 16
  %Re348 = load <2 x double>, ptr %pRe332, align 16
  %Im049 = load <2 x double>, ptr %pIm033, align 16
  %Im150 = load <2 x double>, ptr %pIm134, align 16
  %Im251 = load <2 x double>, ptr %pIm235, align 16
  %Im352 = load <2 x double>, ptr %pIm336, align 16
  %newRe1_ = fmul <2 x double> %mRe4, %Re045
  %newRe1_53 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Re146, <2 x double> %newRe1_)
  %newRe1_54 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Re247, <2 x double> %newRe1_53)
  %newRe1_55 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Re348, <2 x double> %newRe1_54)
  %newRe1_56 = fmul <2 x double> %mRe20, %Im049
  %newRe1_57 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe21, <2 x double> %Im150, <2 x double> %newRe1_56)
  %newRe1_58 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe22, <2 x double> %Im251, <2 x double> %newRe1_57)
  %newRe1_59 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe23, <2 x double> %Im352, <2 x double> %newRe1_58)
  %newRe1 = fsub <2 x double> %newRe1_55, %newRe1_59
  %newIm1_ = fmul <2 x double> %mRe4, %Im049
  %newIm1_60 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Im150, <2 x double> %newIm1_)
  %newIm1_61 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Im251, <2 x double> %newIm1_60)
  %newIm1_62 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Im352, <2 x double> %newIm1_61)
  %newIm1_63 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe20, <2 x double> %Re045, <2 x double> %newIm1_62)
  %newIm1_64 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe21, <2 x double> %Re146, <2 x double> %newIm1_63)
  %newIm1_65 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe22, <2 x double> %Re247, <2 x double> %newIm1_64)
  %newIm1_66 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe23, <2 x double> %Re348, <2 x double> %newIm1_65)
  store <2 x double> %newRe1, ptr %pReAnother138, align 16
  store <2 x double> %newIm1_66, ptr %pImAnother142, align 16
  %idx_next67 = add nsw i64 %idx166, 1
  %exitcond11.not = icmp eq i64 %idx_next67, %idx_end
  br i1 %exitcond11.not, label %entry_batch_2.loopexit, label %loopBody_batch_1

entry_batch_2.loopexit:                           ; preds = %loopBody_batch_1
  %mat68.pre = load <32 x double>, ptr %pmat, align 256
  br label %entry_batch_2

entry_batch_2:                                    ; preds = %entry_batch_2.loopexit, %entry_batch_1
  %mat68 = phi <32 x double> [ %mat68.pre, %entry_batch_2.loopexit ], [ %mat15, %entry_batch_1 ]
  %mRe8 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 11, i32 11>
  %mRe24 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 24, i32 24>
  %mRe25 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 25, i32 25>
  %mRe26 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 26, i32 26>
  %mRe27 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 27, i32 27>
  br i1 %cond1, label %loopBody_batch_2, label %entry_batch_3

loopBody_batch_2:                                 ; preds = %entry_batch_2, %loopBody_batch_2
  %idx698 = phi i64 [ %idx_next120, %loopBody_batch_2 ], [ %idx_start, %entry_batch_2 ]
  %left72 = shl i64 %idx698, 3
  %idx179 = or i64 %left72, 2
  %idx280 = or i64 %left72, 4
  %idx381 = or i64 %left72, 6
  %pRe082 = getelementptr double, ptr %preal, i64 %left72
  %pRe183 = getelementptr double, ptr %preal, i64 %idx179
  %pRe284 = getelementptr double, ptr %preal, i64 %idx280
  %pRe385 = getelementptr double, ptr %preal, i64 %idx381
  %pIm086 = getelementptr double, ptr %pimag, i64 %left72
  %pIm187 = getelementptr double, ptr %pimag, i64 %idx179
  %pIm288 = getelementptr double, ptr %pimag, i64 %idx280
  %pIm389 = getelementptr double, ptr %pimag, i64 %idx381
  %pReAnother292 = getelementptr double, ptr %preal_another, i64 %idx280
  %pImAnother296 = getelementptr double, ptr %pimag_another, i64 %idx280
  %Re098 = load <2 x double>, ptr %pRe082, align 16
  %Re199 = load <2 x double>, ptr %pRe183, align 16
  %Re2100 = load <2 x double>, ptr %pRe284, align 16
  %Re3101 = load <2 x double>, ptr %pRe385, align 16
  %Im0102 = load <2 x double>, ptr %pIm086, align 16
  %Im1103 = load <2 x double>, ptr %pIm187, align 16
  %Im2104 = load <2 x double>, ptr %pIm288, align 16
  %Im3105 = load <2 x double>, ptr %pIm389, align 16
  %newRe2_ = fmul <2 x double> %mRe8, %Re098
  %newRe2_106 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Re199, <2 x double> %newRe2_)
  %newRe2_107 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Re2100, <2 x double> %newRe2_106)
  %newRe2_108 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Re3101, <2 x double> %newRe2_107)
  %newRe2_109 = fmul <2 x double> %mRe24, %Im0102
  %newRe2_110 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe25, <2 x double> %Im1103, <2 x double> %newRe2_109)
  %newRe2_111 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe26, <2 x double> %Im2104, <2 x double> %newRe2_110)
  %newRe2_112 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe27, <2 x double> %Im3105, <2 x double> %newRe2_111)
  %newRe2 = fsub <2 x double> %newRe2_108, %newRe2_112
  %newIm2_ = fmul <2 x double> %mRe8, %Im0102
  %newIm2_113 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Im1103, <2 x double> %newIm2_)
  %newIm2_114 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Im2104, <2 x double> %newIm2_113)
  %newIm2_115 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Im3105, <2 x double> %newIm2_114)
  %newIm2_116 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe24, <2 x double> %Re098, <2 x double> %newIm2_115)
  %newIm2_117 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe25, <2 x double> %Re199, <2 x double> %newIm2_116)
  %newIm2_118 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe26, <2 x double> %Re2100, <2 x double> %newIm2_117)
  %newIm2_119 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe27, <2 x double> %Re3101, <2 x double> %newIm2_118)
  store <2 x double> %newRe2, ptr %pReAnother292, align 16
  store <2 x double> %newIm2_119, ptr %pImAnother296, align 16
  %idx_next120 = add nsw i64 %idx698, 1
  %exitcond12.not = icmp eq i64 %idx_next120, %idx_end
  br i1 %exitcond12.not, label %entry_batch_3.loopexit, label %loopBody_batch_2

entry_batch_3.loopexit:                           ; preds = %loopBody_batch_2
  %mat121.pre = load <32 x double>, ptr %pmat, align 256
  br label %entry_batch_3

entry_batch_3:                                    ; preds = %entry_batch_3.loopexit, %entry_batch_2
  %mat121 = phi <32 x double> [ %mat121.pre, %entry_batch_3.loopexit ], [ %mat68, %entry_batch_2 ]
  %mRe12 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 15, i32 15>
  %mRe28 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 28, i32 28>
  %mRe29 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 29, i32 29>
  %mRe30 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 30, i32 30>
  %mRe31 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 31, i32 31>
  br i1 %cond1, label %loopBody_batch_3, label %ret

loopBody_batch_3:                                 ; preds = %entry_batch_3, %loopBody_batch_3
  %idx12210 = phi i64 [ %idx_next173, %loopBody_batch_3 ], [ %idx_start, %entry_batch_3 ]
  %left125 = shl i64 %idx12210, 3
  %idx1132 = or i64 %left125, 2
  %idx2133 = or i64 %left125, 4
  %idx3134 = or i64 %left125, 6
  %pRe0135 = getelementptr double, ptr %preal, i64 %left125
  %pRe1136 = getelementptr double, ptr %preal, i64 %idx1132
  %pRe2137 = getelementptr double, ptr %preal, i64 %idx2133
  %pRe3138 = getelementptr double, ptr %preal, i64 %idx3134
  %pIm0139 = getelementptr double, ptr %pimag, i64 %left125
  %pIm1140 = getelementptr double, ptr %pimag, i64 %idx1132
  %pIm2141 = getelementptr double, ptr %pimag, i64 %idx2133
  %pIm3142 = getelementptr double, ptr %pimag, i64 %idx3134
  %pReAnother3146 = getelementptr double, ptr %preal_another, i64 %idx3134
  %pImAnother3150 = getelementptr double, ptr %pimag_another, i64 %idx3134
  %Re0151 = load <2 x double>, ptr %pRe0135, align 16
  %Re1152 = load <2 x double>, ptr %pRe1136, align 16
  %Re2153 = load <2 x double>, ptr %pRe2137, align 16
  %Re3154 = load <2 x double>, ptr %pRe3138, align 16
  %Im0155 = load <2 x double>, ptr %pIm0139, align 16
  %Im1156 = load <2 x double>, ptr %pIm1140, align 16
  %Im2157 = load <2 x double>, ptr %pIm2141, align 16
  %Im3158 = load <2 x double>, ptr %pIm3142, align 16
  %newRe3_ = fmul <2 x double> %mRe12, %Re0151
  %newRe3_159 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Re1152, <2 x double> %newRe3_)
  %newRe3_160 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Re2153, <2 x double> %newRe3_159)
  %newRe3_161 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Re3154, <2 x double> %newRe3_160)
  %newRe3_162 = fmul <2 x double> %mRe28, %Im0155
  %newRe3_163 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe29, <2 x double> %Im1156, <2 x double> %newRe3_162)
  %newRe3_164 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe30, <2 x double> %Im2157, <2 x double> %newRe3_163)
  %newRe3_165 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe31, <2 x double> %Im3158, <2 x double> %newRe3_164)
  %newRe3 = fsub <2 x double> %newRe3_161, %newRe3_165
  %newIm3_ = fmul <2 x double> %mRe12, %Im0155
  %newIm3_166 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Im1156, <2 x double> %newIm3_)
  %newIm3_167 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Im2157, <2 x double> %newIm3_166)
  %newIm3_168 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Im3158, <2 x double> %newIm3_167)
  %newIm3_169 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe28, <2 x double> %Re0151, <2 x double> %newIm3_168)
  %newIm3_170 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe29, <2 x double> %Re1152, <2 x double> %newIm3_169)
  %newIm3_171 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe30, <2 x double> %Re2153, <2 x double> %newIm3_170)
  %newIm3_172 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe31, <2 x double> %Re3154, <2 x double> %newIm3_171)
  store <2 x double> %newRe3, ptr %pReAnother3146, align 16
  store <2 x double> %newIm3_172, ptr %pImAnother3150, align 16
  %idx_next173 = add nsw i64 %idx12210, 1
  %exitcond13.not = icmp eq i64 %idx_next173, %idx_end
  br i1 %exitcond13.not, label %ret, label %loopBody_batch_3

ret:                                              ; preds = %loopBody_batch_3, %entry_batch_3
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f32_s2_sep_u2q_k4l3(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <32 x float>, ptr %pmat, align 128
  %mRe0 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %mRe2 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %mRe3 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %mRe4 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %mRe5 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %mRe6 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %mRe7 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %mRe8 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 8, i32 8, i32 8, i32 8>
  %mRe9 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 9, i32 9, i32 9, i32 9>
  %mRe10 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 10, i32 10, i32 10, i32 10>
  %mRe11 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 11, i32 11, i32 11, i32 11>
  %mRe12 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 12, i32 12, i32 12, i32 12>
  %mRe13 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 13, i32 13, i32 13, i32 13>
  %mRe14 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 14, i32 14, i32 14, i32 14>
  %mRe15 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 15, i32 15, i32 15, i32 15>
  %mIm0 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  %mIm1 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 17, i32 17, i32 17, i32 17>
  %mIm2 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 18, i32 18, i32 18, i32 18>
  %mIm3 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 19, i32 19, i32 19, i32 19>
  %mIm4 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 20, i32 20, i32 20, i32 20>
  %mIm5 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 21, i32 21, i32 21, i32 21>
  %mIm6 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 22, i32 22, i32 22, i32 22>
  %mIm7 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 23, i32 23, i32 23, i32 23>
  %mIm8 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 24, i32 24, i32 24, i32 24>
  %mIm9 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 25, i32 25, i32 25, i32 25>
  %mIm10 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 26, i32 26, i32 26, i32 26>
  %mIm11 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 27, i32 27, i32 27, i32 27>
  %mIm12 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 28, i32 28, i32 28, i32 28>
  %mIm13 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 29, i32 29, i32 29, i32 29>
  %mIm14 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 30, i32 30, i32 30, i32 30>
  %mIm15 = shufflevector <32 x float> %mat, <32 x float> poison, <4 x i32> <i32 31, i32 31, i32 31, i32 31>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx4 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %left_tmp = shl i64 %idx4, 4
  %left = and i64 %left_tmp, -32
  %right_tmp = shl i64 %idx4, 2
  %right = and i64 %right_tmp, 4
  %idx0 = or i64 %left, %right
  %idx1 = or i64 %idx0, 8
  %idx2 = or i64 %idx0, 16
  %idx3 = or i64 %idx0, 24
  %pRe0 = getelementptr float, ptr %preal, i64 %idx0
  %pRe1 = getelementptr float, ptr %preal, i64 %idx1
  %pRe2 = getelementptr float, ptr %preal, i64 %idx2
  %pRe3 = getelementptr float, ptr %preal, i64 %idx3
  %pIm0 = getelementptr float, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr float, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr float, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr float, ptr %pimag, i64 %idx3
  %Re0 = load <4 x float>, ptr %pRe0, align 16
  %Re1 = load <4 x float>, ptr %pRe1, align 16
  %Re2 = load <4 x float>, ptr %pRe2, align 16
  %Re3 = load <4 x float>, ptr %pRe3, align 16
  %Im0 = load <4 x float>, ptr %pIm0, align 16
  %Im1 = load <4 x float>, ptr %pIm1, align 16
  %Im2 = load <4 x float>, ptr %pIm2, align 16
  %Im3 = load <4 x float>, ptr %pIm3, align 16
  %newRe0_ = fmul <4 x float> %mRe0, %Re0
  %newRe0_1 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe1, <4 x float> %Re1, <4 x float> %newRe0_)
  %newRe0_2 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe2, <4 x float> %Re2, <4 x float> %newRe0_1)
  %newRe0_3 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe3, <4 x float> %Re3, <4 x float> %newRe0_2)
  %newRe0_4 = fmul <4 x float> %mIm0, %Im0
  %newRe0_5 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm1, <4 x float> %Im1, <4 x float> %newRe0_4)
  %newRe0_6 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm2, <4 x float> %Im2, <4 x float> %newRe0_5)
  %newRe0_7 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm3, <4 x float> %Im3, <4 x float> %newRe0_6)
  %newRe0 = fsub <4 x float> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <4 x float> %mRe4, %Re0
  %newRe1_8 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe5, <4 x float> %Re1, <4 x float> %newRe1_)
  %newRe1_9 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe6, <4 x float> %Re2, <4 x float> %newRe1_8)
  %newRe1_10 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe7, <4 x float> %Re3, <4 x float> %newRe1_9)
  %newRe1_11 = fmul <4 x float> %mIm4, %Im0
  %newRe1_12 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm5, <4 x float> %Im1, <4 x float> %newRe1_11)
  %newRe1_13 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm6, <4 x float> %Im2, <4 x float> %newRe1_12)
  %newRe1_14 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm7, <4 x float> %Im3, <4 x float> %newRe1_13)
  %newRe1 = fsub <4 x float> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <4 x float> %mRe8, %Re0
  %newRe2_15 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe9, <4 x float> %Re1, <4 x float> %newRe2_)
  %newRe2_16 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe10, <4 x float> %Re2, <4 x float> %newRe2_15)
  %newRe2_17 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe11, <4 x float> %Re3, <4 x float> %newRe2_16)
  %newRe2_18 = fmul <4 x float> %mIm8, %Im0
  %newRe2_19 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm9, <4 x float> %Im1, <4 x float> %newRe2_18)
  %newRe2_20 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm10, <4 x float> %Im2, <4 x float> %newRe2_19)
  %newRe2_21 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm11, <4 x float> %Im3, <4 x float> %newRe2_20)
  %newRe2 = fsub <4 x float> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <4 x float> %mRe12, %Re0
  %newRe3_22 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe13, <4 x float> %Re1, <4 x float> %newRe3_)
  %newRe3_23 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe14, <4 x float> %Re2, <4 x float> %newRe3_22)
  %newRe3_24 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe15, <4 x float> %Re3, <4 x float> %newRe3_23)
  %newRe3_25 = fmul <4 x float> %mIm12, %Im0
  %newRe3_26 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm13, <4 x float> %Im1, <4 x float> %newRe3_25)
  %newRe3_27 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm14, <4 x float> %Im2, <4 x float> %newRe3_26)
  %newRe3_28 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm15, <4 x float> %Im3, <4 x float> %newRe3_27)
  %newRe3 = fsub <4 x float> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <4 x float> %mRe0, %Im0
  %newIm0_29 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe1, <4 x float> %Im1, <4 x float> %newIm0_)
  %newIm0_30 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe2, <4 x float> %Im2, <4 x float> %newIm0_29)
  %newIm0_31 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe3, <4 x float> %Im3, <4 x float> %newIm0_30)
  %newIm0_32 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm0, <4 x float> %Re0, <4 x float> %newIm0_31)
  %newIm0_33 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm1, <4 x float> %Re1, <4 x float> %newIm0_32)
  %newIm0_34 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm2, <4 x float> %Re2, <4 x float> %newIm0_33)
  %newIm0_35 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm3, <4 x float> %Re3, <4 x float> %newIm0_34)
  %newIm1_ = fmul <4 x float> %mRe4, %Im0
  %newIm1_36 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe5, <4 x float> %Im1, <4 x float> %newIm1_)
  %newIm1_37 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe6, <4 x float> %Im2, <4 x float> %newIm1_36)
  %newIm1_38 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe7, <4 x float> %Im3, <4 x float> %newIm1_37)
  %newIm1_39 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm4, <4 x float> %Re0, <4 x float> %newIm1_38)
  %newIm1_40 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm5, <4 x float> %Re1, <4 x float> %newIm1_39)
  %newIm1_41 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm6, <4 x float> %Re2, <4 x float> %newIm1_40)
  %newIm1_42 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm7, <4 x float> %Re3, <4 x float> %newIm1_41)
  %newIm2_ = fmul <4 x float> %mRe8, %Im0
  %newIm2_43 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe9, <4 x float> %Im1, <4 x float> %newIm2_)
  %newIm2_44 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe10, <4 x float> %Im2, <4 x float> %newIm2_43)
  %newIm2_45 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe11, <4 x float> %Im3, <4 x float> %newIm2_44)
  %newIm2_46 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm8, <4 x float> %Re0, <4 x float> %newIm2_45)
  %newIm2_47 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm9, <4 x float> %Re1, <4 x float> %newIm2_46)
  %newIm2_48 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm10, <4 x float> %Re2, <4 x float> %newIm2_47)
  %newIm2_49 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm11, <4 x float> %Re3, <4 x float> %newIm2_48)
  %newIm3_ = fmul <4 x float> %mRe12, %Im0
  %newIm3_50 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe13, <4 x float> %Im1, <4 x float> %newIm3_)
  %newIm3_51 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe14, <4 x float> %Im2, <4 x float> %newIm3_50)
  %newIm3_52 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mRe15, <4 x float> %Im3, <4 x float> %newIm3_51)
  %newIm3_53 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm12, <4 x float> %Re0, <4 x float> %newIm3_52)
  %newIm3_54 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm13, <4 x float> %Re1, <4 x float> %newIm3_53)
  %newIm3_55 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm14, <4 x float> %Re2, <4 x float> %newIm3_54)
  %newIm3_56 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %mIm15, <4 x float> %Re3, <4 x float> %newIm3_55)
  store <4 x float> %newRe0, ptr %pRe0, align 16
  store <4 x float> %newRe1, ptr %pRe1, align 16
  store <4 x float> %newRe2, ptr %pRe2, align 16
  store <4 x float> %newRe3, ptr %pRe3, align 16
  store <4 x float> %newIm0_35, ptr %pIm0, align 16
  store <4 x float> %newIm1_42, ptr %pIm1, align 16
  store <4 x float> %newIm2_49, ptr %pIm2, align 16
  store <4 x float> %newIm3_56, ptr %pIm3, align 16
  %idx_next = add nsw i64 %idx4, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #1

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s1_sep_u2q_k2l0(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 3, i32 3>
  %mRe4 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 7, i32 7>
  %mRe8 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 11, i32 11>
  %mRe12 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 15, i32 15>
  %mIm0 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 16, i32 16>
  %mIm1 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 17, i32 17>
  %mIm2 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 18, i32 18>
  %mIm3 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 19, i32 19>
  %mIm4 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 20, i32 20>
  %mIm5 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 21, i32 21>
  %mIm6 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 22, i32 22>
  %mIm7 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 23, i32 23>
  %mIm8 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 24, i32 24>
  %mIm9 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 25, i32 25>
  %mIm10 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 26, i32 26>
  %mIm11 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 27, i32 27>
  %mIm12 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 28, i32 28>
  %mIm13 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 29, i32 29>
  %mIm14 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 30, i32 30>
  %mIm15 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 31, i32 31>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %idxHi = add i64 %idx2, 4
  %pReLo = getelementptr <4 x double>, ptr %preal, i64 %idx2
  %pReHi = getelementptr <4 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <4 x double>, ptr %pimag, i64 %idx2
  %pImHi = getelementptr <4 x double>, ptr %pimag, i64 %idxHi
  %ReLo = load <4 x double>, ptr %pReLo, align 32
  %ReHi = load <4 x double>, ptr %pReHi, align 32
  %ImLo = load <4 x double>, ptr %pImLo, align 32
  %ImHi = load <4 x double>, ptr %pImHi, align 32
  %Re0 = shufflevector <4 x double> %ReLo, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %Re1 = shufflevector <4 x double> %ReLo, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %Re2 = shufflevector <4 x double> %ReHi, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %Re3 = shufflevector <4 x double> %ReHi, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %Im0 = shufflevector <4 x double> %ImLo, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %Im1 = shufflevector <4 x double> %ImLo, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %Im2 = shufflevector <4 x double> %ImHi, <4 x double> poison, <2 x i32> <i32 0, i32 2>
  %Im3 = shufflevector <4 x double> %ImHi, <4 x double> poison, <2 x i32> <i32 1, i32 3>
  %newRe0_ = fmul <2 x double> %mRe0, %Re0
  %newRe0_1 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Re1, <2 x double> %newRe0_)
  %newRe0_2 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Re2, <2 x double> %newRe0_1)
  %newRe0_3 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Re3, <2 x double> %newRe0_2)
  %newRe0_4 = fmul <2 x double> %mIm0, %Im0
  %newRe0_5 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Im1, <2 x double> %newRe0_4)
  %newRe0_6 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Im2, <2 x double> %newRe0_5)
  %newRe0_7 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Im3, <2 x double> %newRe0_6)
  %newRe0 = fsub <2 x double> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <2 x double> %mRe4, %Re0
  %newRe1_8 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Re1, <2 x double> %newRe1_)
  %newRe1_9 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Re2, <2 x double> %newRe1_8)
  %newRe1_10 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Re3, <2 x double> %newRe1_9)
  %newRe1_11 = fmul <2 x double> %mIm4, %Im0
  %newRe1_12 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Im1, <2 x double> %newRe1_11)
  %newRe1_13 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Im2, <2 x double> %newRe1_12)
  %newRe1_14 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Im3, <2 x double> %newRe1_13)
  %newRe1 = fsub <2 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <2 x double> %mRe8, %Re0
  %newRe2_15 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Re1, <2 x double> %newRe2_)
  %newRe2_16 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Re2, <2 x double> %newRe2_15)
  %newRe2_17 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Re3, <2 x double> %newRe2_16)
  %newRe2_18 = fmul <2 x double> %mIm8, %Im0
  %newRe2_19 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Im1, <2 x double> %newRe2_18)
  %newRe2_20 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Im2, <2 x double> %newRe2_19)
  %newRe2_21 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Im3, <2 x double> %newRe2_20)
  %newRe2 = fsub <2 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <2 x double> %mRe12, %Re0
  %newRe3_22 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Re1, <2 x double> %newRe3_)
  %newRe3_23 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Re2, <2 x double> %newRe3_22)
  %newRe3_24 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Re3, <2 x double> %newRe3_23)
  %newRe3_25 = fmul <2 x double> %mIm12, %Im0
  %newRe3_26 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Im1, <2 x double> %newRe3_25)
  %newRe3_27 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Im2, <2 x double> %newRe3_26)
  %newRe3_28 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Im3, <2 x double> %newRe3_27)
  %newRe3 = fsub <2 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <2 x double> %mRe0, %Im0
  %newIm0_29 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Im1, <2 x double> %newIm0_)
  %newIm0_30 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Im2, <2 x double> %newIm0_29)
  %newIm0_31 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Im3, <2 x double> %newIm0_30)
  %newIm0_32 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm0, <2 x double> %Re0, <2 x double> %newIm0_31)
  %newIm0_33 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Re1, <2 x double> %newIm0_32)
  %newIm0_34 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Re2, <2 x double> %newIm0_33)
  %newIm0_35 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Re3, <2 x double> %newIm0_34)
  %newIm1_ = fmul <2 x double> %mRe4, %Im0
  %newIm1_36 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Im1, <2 x double> %newIm1_)
  %newIm1_37 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Im2, <2 x double> %newIm1_36)
  %newIm1_38 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Im3, <2 x double> %newIm1_37)
  %newIm1_39 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm4, <2 x double> %Re0, <2 x double> %newIm1_38)
  %newIm1_40 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Re1, <2 x double> %newIm1_39)
  %newIm1_41 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Re2, <2 x double> %newIm1_40)
  %newIm1_42 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Re3, <2 x double> %newIm1_41)
  %newIm2_ = fmul <2 x double> %mRe8, %Im0
  %newIm2_43 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Im1, <2 x double> %newIm2_)
  %newIm2_44 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Im2, <2 x double> %newIm2_43)
  %newIm2_45 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Im3, <2 x double> %newIm2_44)
  %newIm2_46 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm8, <2 x double> %Re0, <2 x double> %newIm2_45)
  %newIm2_47 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Re1, <2 x double> %newIm2_46)
  %newIm2_48 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Re2, <2 x double> %newIm2_47)
  %newIm2_49 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Re3, <2 x double> %newIm2_48)
  %newIm3_ = fmul <2 x double> %mRe12, %Im0
  %newIm3_50 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Im1, <2 x double> %newIm3_)
  %newIm3_51 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Im2, <2 x double> %newIm3_50)
  %newIm3_52 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Im3, <2 x double> %newIm3_51)
  %newIm3_53 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm12, <2 x double> %Re0, <2 x double> %newIm3_52)
  %newIm3_54 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Re1, <2 x double> %newIm3_53)
  %newIm3_55 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Re2, <2 x double> %newIm3_54)
  %newIm3_56 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Re3, <2 x double> %newIm3_55)
  %newReLo = shufflevector <2 x double> %newRe0, <2 x double> %newRe1, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  %newReHi = shufflevector <2 x double> %newRe2, <2 x double> %newRe3, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %newReLo, ptr %pReLo, align 32
  store <4 x double> %newReHi, ptr %pReHi, align 32
  %newImLo = shufflevector <2 x double> %newIm0_35, <2 x double> %newIm1_42, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  %newImHi = shufflevector <2 x double> %newIm2_49, <2 x double> %newIm3_56, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %newImLo, ptr %pImLo, align 32
  store <4 x double> %newImHi, ptr %pImHi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define void @f64_s2_sep_u3_k0_10010000(ptr nocapture readnone %preal, ptr nocapture readnone %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readnone %pmat) local_unnamed_addr #2 {
entry:
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_sep_u3_k1_33330333(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %pRe = getelementptr <8 x double>, ptr %preal, i64 %idx2
  %pIm = getelementptr <8 x double>, ptr %pimag, i64 %idx2
  %Re = load <8 x double>, ptr %pRe, align 64
  %Im = load <8 x double>, ptr %pIm, align 64
  %Ar = shufflevector <8 x double> %Re, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Ai = shufflevector <8 x double> %Im, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Br = shufflevector <8 x double> %Re, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Bi = shufflevector <8 x double> %Im, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %biBi = fmul <4 x double> %bi, %Bi
  %newAr2 = fsub <4 x double> %newAr1, %biBi
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi)
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi, <4 x double> %Br, <4 x double> %newAi3)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci, %Ai
  %newBr6 = fsub <4 x double> %newBr5, %ciAi
  %diBi = fmul <4 x double> %di, %Bi
  %newBr7 = fsub <4 x double> %newBr6, %diBi
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi8 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci, <4 x double> %Ar, <4 x double> %newBi)
  %newBi9 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di, <4 x double> %Br, <4 x double> %newBi8)
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi9)
  %newRe = shufflevector <4 x double> %newAr2, <4 x double> %newBr7, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newIm = shufflevector <4 x double> %newAi4, <4 x double> %newBi10, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newRe, ptr %pRe, align 64
  store <8 x double> %newIm, ptr %pIm, align 64
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #1

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_sep_u3_k2_33330333(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %shl_outer = shl i64 %idx2, 3
  %idxB = or i64 %shl_outer, 4
  %pAr = getelementptr double, ptr %preal, i64 %shl_outer
  %pAi = getelementptr double, ptr %pimag, i64 %shl_outer
  %pBr = getelementptr double, ptr %preal, i64 %idxB
  %pBi = getelementptr double, ptr %pimag, i64 %idxB
  %Ar = load <4 x double>, ptr %pAr, align 32
  %Ai = load <4 x double>, ptr %pAi, align 32
  %Br = load <4 x double>, ptr %pBr, align 32
  %Bi = load <4 x double>, ptr %pBi, align 32
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %biBi = fmul <4 x double> %bi, %Bi
  %newAr2 = fsub <4 x double> %newAr1, %biBi
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi)
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi, <4 x double> %Br, <4 x double> %newAi3)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci, %Ai
  %newBr6 = fsub <4 x double> %newBr5, %ciAi
  %diBi = fmul <4 x double> %di, %Bi
  %newBr7 = fsub <4 x double> %newBr6, %diBi
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi8 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci, <4 x double> %Ar, <4 x double> %newBi)
  %newBi9 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di, <4 x double> %Br, <4 x double> %newBi8)
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi9)
  store <4 x double> %newAr2, ptr %pAr, align 32
  store <4 x double> %newAi4, ptr %pAi, align 32
  store <4 x double> %newBr7, ptr %pBr, align 32
  store <4 x double> %newBi10, ptr %pBi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f64_s2_sep_u3_k3_33330333(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %idx_and_outer = shl i64 %idx2, 3
  %shl_outer = and i64 %idx_and_outer, -16
  %idx_and_inner = shl i64 %idx2, 2
  %shl_inner = and i64 %idx_and_inner, 4
  %idxA = or i64 %shl_outer, %shl_inner
  %idxB = or i64 %idxA, 8
  %pAr = getelementptr double, ptr %preal, i64 %idxA
  %pAi = getelementptr double, ptr %pimag, i64 %idxA
  %pBr = getelementptr double, ptr %preal, i64 %idxB
  %pBi = getelementptr double, ptr %pimag, i64 %idxB
  %Ar = load <4 x double>, ptr %pAr, align 32
  %Ai = load <4 x double>, ptr %pAi, align 32
  %Br = load <4 x double>, ptr %pBr, align 32
  %Bi = load <4 x double>, ptr %pBi, align 32
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %biBi = fmul <4 x double> %bi, %Bi
  %newAr2 = fsub <4 x double> %newAr1, %biBi
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi)
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi, <4 x double> %Br, <4 x double> %newAi3)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci, %Ai
  %newBr6 = fsub <4 x double> %newBr5, %ciAi
  %diBi = fmul <4 x double> %di, %Bi
  %newBr7 = fsub <4 x double> %newBr6, %diBi
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi8 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci, <4 x double> %Ar, <4 x double> %newBi)
  %newBi9 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di, <4 x double> %Br, <4 x double> %newBi8)
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi9)
  store <4 x double> %newAr2, ptr %pAr, align 32
  store <4 x double> %newAi4, ptr %pAi, align 32
  store <4 x double> %newBr7, ptr %pBr, align 32
  store <4 x double> %newBi10, ptr %pBi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #1

attributes #0 = { nofree nosync nounwind memory(argmem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
