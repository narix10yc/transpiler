; ModuleID = 'myModule'
source_filename = "myModule"

define void @f64_s1_sep_u2q_k2l1_batched(ptr %preal, ptr %pimag, ptr %preal_another, ptr %pimag_another, i64 %idx_start, i64 %idx_end, ptr %pmat) {
global_entry:
  br label %entry_batch_0

entry_batch_0:                                    ; preds = %global_entry
  %mat = load <32 x double>, ptr %pmat, align 256
  %mRe0 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> zeroinitializer
  %mRe1 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 1, i32 1>
  %mRe2 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 2, i32 2>
  %mRe3 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 3, i32 3>
  %mRe16 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 16, i32 16>
  %mRe17 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 17, i32 17>
  %mRe18 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 18, i32 18>
  %mRe19 = shufflevector <32 x double> %mat, <32 x double> poison, <2 x i32> <i32 19, i32 19>
  br label %loop_batch_0

loop_batch_0:                                     ; preds = %loopBody_batch_0, %entry_batch_0
  %idx = phi i64 [ %idx_start, %entry_batch_0 ], [ %idx_next, %loopBody_batch_0 ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody_batch_0, label %entry_batch_1

loopBody_batch_0:                                 ; preds = %loop_batch_0
  %left_tmp = and i64 %idx, -1
  %left = shl i64 %left_tmp, 3
  %middle_tmp = and i64 %idx, 0
  %middle = shl i64 %middle_tmp, 2
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 1
  %idx0_tmp = or i64 %left, %middle
  %idx0 = or i64 %idx0_tmp, %right
  %idx1 = or i64 %idx0, 2
  %idx2 = or i64 %idx0, 4
  %idx3 = or i64 %idx0, 6
  %pRe0 = getelementptr double, ptr %preal, i64 %idx0
  %pRe1 = getelementptr double, ptr %preal, i64 %idx1
  %pRe2 = getelementptr double, ptr %preal, i64 %idx2
  %pRe3 = getelementptr double, ptr %preal, i64 %idx3
  %pIm0 = getelementptr double, ptr %pimag, i64 %idx0
  %pIm1 = getelementptr double, ptr %pimag, i64 %idx1
  %pIm2 = getelementptr double, ptr %pimag, i64 %idx2
  %pIm3 = getelementptr double, ptr %pimag, i64 %idx3
  %pReAnother0 = getelementptr double, ptr %preal_another, i64 %idx0
  %pReAnother1 = getelementptr double, ptr %preal_another, i64 %idx1
  %pReAnother2 = getelementptr double, ptr %preal_another, i64 %idx2
  %pReAnother3 = getelementptr double, ptr %preal_another, i64 %idx3
  %pImAnother0 = getelementptr double, ptr %pimag_another, i64 %idx0
  %pImAnother1 = getelementptr double, ptr %pimag_another, i64 %idx1
  %pImAnother2 = getelementptr double, ptr %pimag_another, i64 %idx2
  %pImAnother3 = getelementptr double, ptr %pimag_another, i64 %idx3
  %Re0 = load <2 x double>, ptr %pRe0, align 16
  %Re1 = load <2 x double>, ptr %pRe1, align 16
  %Re2 = load <2 x double>, ptr %pRe2, align 16
  %Re3 = load <2 x double>, ptr %pRe3, align 16
  %Im0 = load <2 x double>, ptr %pIm0, align 16
  %Im1 = load <2 x double>, ptr %pIm1, align 16
  %Im2 = load <2 x double>, ptr %pIm2, align 16
  %Im3 = load <2 x double>, ptr %pIm3, align 16
  %newRe0_ = fmul <2 x double> %mRe0, %Re0
  %newRe0_1 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Re1, <2 x double> %newRe0_)
  %newRe0_2 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Re2, <2 x double> %newRe0_1)
  %newRe0_3 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Re3, <2 x double> %newRe0_2)
  %newRe0_4 = fmul <2 x double> %mRe16, %Im0
  %newRe0_5 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe17, <2 x double> %Im1, <2 x double> %newRe0_4)
  %newRe0_6 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe18, <2 x double> %Im2, <2 x double> %newRe0_5)
  %newRe0_7 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe19, <2 x double> %Im3, <2 x double> %newRe0_6)
  %newRe0 = fsub <2 x double> %newRe0_3, %newRe0_7
  %newIm0_ = fmul <2 x double> %mRe0, %Im0
  %newIm0_8 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Im1, <2 x double> %newIm0_)
  %newIm0_9 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Im2, <2 x double> %newIm0_8)
  %newIm0_10 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Im3, <2 x double> %newIm0_9)
  %newIm0_11 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe16, <2 x double> %Re0, <2 x double> %newIm0_10)
  %newIm0_12 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe17, <2 x double> %Re1, <2 x double> %newIm0_11)
  %newIm0_13 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe18, <2 x double> %Re2, <2 x double> %newIm0_12)
  %newIm0_14 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe19, <2 x double> %Re3, <2 x double> %newIm0_13)
  store <2 x double> %newRe0, ptr %pReAnother0, align 16
  store <2 x double> %newIm0_14, ptr %pImAnother0, align 16
  %idx_next = add i64 %idx, 1
  br label %loop_batch_0

entry_batch_1:                                    ; preds = %loop_batch_0
  %mat15 = load <32 x double>, ptr %pmat, align 256
  %mRe4 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 4, i32 4>
  %mRe5 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 5, i32 5>
  %mRe6 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 6, i32 6>
  %mRe7 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 7, i32 7>
  %mRe20 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 20, i32 20>
  %mRe21 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 21, i32 21>
  %mRe22 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 22, i32 22>
  %mRe23 = shufflevector <32 x double> %mat15, <32 x double> poison, <2 x i32> <i32 23, i32 23>
  br label %loop_batch_1

loop_batch_1:                                     ; preds = %loopBody_batch_1, %entry_batch_1
  %idx16 = phi i64 [ %idx_start, %entry_batch_1 ], [ %idx_next67, %loopBody_batch_1 ]
  %cond17 = icmp slt i64 %idx16, %idx_end
  br i1 %cond17, label %loopBody_batch_1, label %entry_batch_2

loopBody_batch_1:                                 ; preds = %loop_batch_1
  %left_tmp18 = and i64 %idx16, -1
  %left19 = shl i64 %left_tmp18, 3
  %middle_tmp20 = and i64 %idx16, 0
  %middle21 = shl i64 %middle_tmp20, 2
  %right_tmp22 = and i64 %idx16, 0
  %right23 = shl i64 %right_tmp22, 1
  %idx0_tmp24 = or i64 %left19, %middle21
  %idx025 = or i64 %idx0_tmp24, %right23
  %idx126 = or i64 %idx025, 2
  %idx227 = or i64 %idx025, 4
  %idx328 = or i64 %idx025, 6
  %pRe029 = getelementptr double, ptr %preal, i64 %idx025
  %pRe130 = getelementptr double, ptr %preal, i64 %idx126
  %pRe231 = getelementptr double, ptr %preal, i64 %idx227
  %pRe332 = getelementptr double, ptr %preal, i64 %idx328
  %pIm033 = getelementptr double, ptr %pimag, i64 %idx025
  %pIm134 = getelementptr double, ptr %pimag, i64 %idx126
  %pIm235 = getelementptr double, ptr %pimag, i64 %idx227
  %pIm336 = getelementptr double, ptr %pimag, i64 %idx328
  %pReAnother037 = getelementptr double, ptr %preal_another, i64 %idx025
  %pReAnother138 = getelementptr double, ptr %preal_another, i64 %idx126
  %pReAnother239 = getelementptr double, ptr %preal_another, i64 %idx227
  %pReAnother340 = getelementptr double, ptr %preal_another, i64 %idx328
  %pImAnother041 = getelementptr double, ptr %pimag_another, i64 %idx025
  %pImAnother142 = getelementptr double, ptr %pimag_another, i64 %idx126
  %pImAnother243 = getelementptr double, ptr %pimag_another, i64 %idx227
  %pImAnother344 = getelementptr double, ptr %pimag_another, i64 %idx328
  %Re045 = load <2 x double>, ptr %pRe029, align 16
  %Re146 = load <2 x double>, ptr %pRe130, align 16
  %Re247 = load <2 x double>, ptr %pRe231, align 16
  %Re348 = load <2 x double>, ptr %pRe332, align 16
  %Im049 = load <2 x double>, ptr %pIm033, align 16
  %Im150 = load <2 x double>, ptr %pIm134, align 16
  %Im251 = load <2 x double>, ptr %pIm235, align 16
  %Im352 = load <2 x double>, ptr %pIm336, align 16
  %newRe1_ = fmul <2 x double> %mRe4, %Re045
  %newRe1_53 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Re146, <2 x double> %newRe1_)
  %newRe1_54 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Re247, <2 x double> %newRe1_53)
  %newRe1_55 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Re348, <2 x double> %newRe1_54)
  %newRe1_56 = fmul <2 x double> %mRe20, %Im049
  %newRe1_57 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe21, <2 x double> %Im150, <2 x double> %newRe1_56)
  %newRe1_58 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe22, <2 x double> %Im251, <2 x double> %newRe1_57)
  %newRe1_59 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe23, <2 x double> %Im352, <2 x double> %newRe1_58)
  %newRe1 = fsub <2 x double> %newRe1_55, %newRe1_59
  %newIm1_ = fmul <2 x double> %mRe4, %Im049
  %newIm1_60 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Im150, <2 x double> %newIm1_)
  %newIm1_61 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Im251, <2 x double> %newIm1_60)
  %newIm1_62 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Im352, <2 x double> %newIm1_61)
  %newIm1_63 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe20, <2 x double> %Re045, <2 x double> %newIm1_62)
  %newIm1_64 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe21, <2 x double> %Re146, <2 x double> %newIm1_63)
  %newIm1_65 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe22, <2 x double> %Re247, <2 x double> %newIm1_64)
  %newIm1_66 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe23, <2 x double> %Re348, <2 x double> %newIm1_65)
  store <2 x double> %newRe1, ptr %pReAnother138, align 16
  store <2 x double> %newIm1_66, ptr %pImAnother142, align 16
  %idx_next67 = add i64 %idx16, 1
  br label %loop_batch_1

entry_batch_2:                                    ; preds = %loop_batch_1
  %mat68 = load <32 x double>, ptr %pmat, align 256
  %mRe8 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 8, i32 8>
  %mRe9 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 9, i32 9>
  %mRe10 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 10, i32 10>
  %mRe11 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 11, i32 11>
  %mRe24 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 24, i32 24>
  %mRe25 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 25, i32 25>
  %mRe26 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 26, i32 26>
  %mRe27 = shufflevector <32 x double> %mat68, <32 x double> poison, <2 x i32> <i32 27, i32 27>
  br label %loop_batch_2

loop_batch_2:                                     ; preds = %loopBody_batch_2, %entry_batch_2
  %idx69 = phi i64 [ %idx_start, %entry_batch_2 ], [ %idx_next120, %loopBody_batch_2 ]
  %cond70 = icmp slt i64 %idx69, %idx_end
  br i1 %cond70, label %loopBody_batch_2, label %entry_batch_3

loopBody_batch_2:                                 ; preds = %loop_batch_2
  %left_tmp71 = and i64 %idx69, -1
  %left72 = shl i64 %left_tmp71, 3
  %middle_tmp73 = and i64 %idx69, 0
  %middle74 = shl i64 %middle_tmp73, 2
  %right_tmp75 = and i64 %idx69, 0
  %right76 = shl i64 %right_tmp75, 1
  %idx0_tmp77 = or i64 %left72, %middle74
  %idx078 = or i64 %idx0_tmp77, %right76
  %idx179 = or i64 %idx078, 2
  %idx280 = or i64 %idx078, 4
  %idx381 = or i64 %idx078, 6
  %pRe082 = getelementptr double, ptr %preal, i64 %idx078
  %pRe183 = getelementptr double, ptr %preal, i64 %idx179
  %pRe284 = getelementptr double, ptr %preal, i64 %idx280
  %pRe385 = getelementptr double, ptr %preal, i64 %idx381
  %pIm086 = getelementptr double, ptr %pimag, i64 %idx078
  %pIm187 = getelementptr double, ptr %pimag, i64 %idx179
  %pIm288 = getelementptr double, ptr %pimag, i64 %idx280
  %pIm389 = getelementptr double, ptr %pimag, i64 %idx381
  %pReAnother090 = getelementptr double, ptr %preal_another, i64 %idx078
  %pReAnother191 = getelementptr double, ptr %preal_another, i64 %idx179
  %pReAnother292 = getelementptr double, ptr %preal_another, i64 %idx280
  %pReAnother393 = getelementptr double, ptr %preal_another, i64 %idx381
  %pImAnother094 = getelementptr double, ptr %pimag_another, i64 %idx078
  %pImAnother195 = getelementptr double, ptr %pimag_another, i64 %idx179
  %pImAnother296 = getelementptr double, ptr %pimag_another, i64 %idx280
  %pImAnother397 = getelementptr double, ptr %pimag_another, i64 %idx381
  %Re098 = load <2 x double>, ptr %pRe082, align 16
  %Re199 = load <2 x double>, ptr %pRe183, align 16
  %Re2100 = load <2 x double>, ptr %pRe284, align 16
  %Re3101 = load <2 x double>, ptr %pRe385, align 16
  %Im0102 = load <2 x double>, ptr %pIm086, align 16
  %Im1103 = load <2 x double>, ptr %pIm187, align 16
  %Im2104 = load <2 x double>, ptr %pIm288, align 16
  %Im3105 = load <2 x double>, ptr %pIm389, align 16
  %newRe2_ = fmul <2 x double> %mRe8, %Re098
  %newRe2_106 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Re199, <2 x double> %newRe2_)
  %newRe2_107 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Re2100, <2 x double> %newRe2_106)
  %newRe2_108 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Re3101, <2 x double> %newRe2_107)
  %newRe2_109 = fmul <2 x double> %mRe24, %Im0102
  %newRe2_110 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe25, <2 x double> %Im1103, <2 x double> %newRe2_109)
  %newRe2_111 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe26, <2 x double> %Im2104, <2 x double> %newRe2_110)
  %newRe2_112 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe27, <2 x double> %Im3105, <2 x double> %newRe2_111)
  %newRe2 = fsub <2 x double> %newRe2_108, %newRe2_112
  %newIm2_ = fmul <2 x double> %mRe8, %Im0102
  %newIm2_113 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Im1103, <2 x double> %newIm2_)
  %newIm2_114 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Im2104, <2 x double> %newIm2_113)
  %newIm2_115 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Im3105, <2 x double> %newIm2_114)
  %newIm2_116 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe24, <2 x double> %Re098, <2 x double> %newIm2_115)
  %newIm2_117 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe25, <2 x double> %Re199, <2 x double> %newIm2_116)
  %newIm2_118 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe26, <2 x double> %Re2100, <2 x double> %newIm2_117)
  %newIm2_119 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe27, <2 x double> %Re3101, <2 x double> %newIm2_118)
  store <2 x double> %newRe2, ptr %pReAnother292, align 16
  store <2 x double> %newIm2_119, ptr %pImAnother296, align 16
  %idx_next120 = add i64 %idx69, 1
  br label %loop_batch_2

entry_batch_3:                                    ; preds = %loop_batch_2
  %mat121 = load <32 x double>, ptr %pmat, align 256
  %mRe12 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 12, i32 12>
  %mRe13 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 13, i32 13>
  %mRe14 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 14, i32 14>
  %mRe15 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 15, i32 15>
  %mRe28 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 28, i32 28>
  %mRe29 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 29, i32 29>
  %mRe30 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 30, i32 30>
  %mRe31 = shufflevector <32 x double> %mat121, <32 x double> poison, <2 x i32> <i32 31, i32 31>
  br label %loop_batch_3

loop_batch_3:                                     ; preds = %loopBody_batch_3, %entry_batch_3
  %idx122 = phi i64 [ %idx_start, %entry_batch_3 ], [ %idx_next173, %loopBody_batch_3 ]
  %cond123 = icmp slt i64 %idx122, %idx_end
  br i1 %cond123, label %loopBody_batch_3, label %ret

loopBody_batch_3:                                 ; preds = %loop_batch_3
  %left_tmp124 = and i64 %idx122, -1
  %left125 = shl i64 %left_tmp124, 3
  %middle_tmp126 = and i64 %idx122, 0
  %middle127 = shl i64 %middle_tmp126, 2
  %right_tmp128 = and i64 %idx122, 0
  %right129 = shl i64 %right_tmp128, 1
  %idx0_tmp130 = or i64 %left125, %middle127
  %idx0131 = or i64 %idx0_tmp130, %right129
  %idx1132 = or i64 %idx0131, 2
  %idx2133 = or i64 %idx0131, 4
  %idx3134 = or i64 %idx0131, 6
  %pRe0135 = getelementptr double, ptr %preal, i64 %idx0131
  %pRe1136 = getelementptr double, ptr %preal, i64 %idx1132
  %pRe2137 = getelementptr double, ptr %preal, i64 %idx2133
  %pRe3138 = getelementptr double, ptr %preal, i64 %idx3134
  %pIm0139 = getelementptr double, ptr %pimag, i64 %idx0131
  %pIm1140 = getelementptr double, ptr %pimag, i64 %idx1132
  %pIm2141 = getelementptr double, ptr %pimag, i64 %idx2133
  %pIm3142 = getelementptr double, ptr %pimag, i64 %idx3134
  %pReAnother0143 = getelementptr double, ptr %preal_another, i64 %idx0131
  %pReAnother1144 = getelementptr double, ptr %preal_another, i64 %idx1132
  %pReAnother2145 = getelementptr double, ptr %preal_another, i64 %idx2133
  %pReAnother3146 = getelementptr double, ptr %preal_another, i64 %idx3134
  %pImAnother0147 = getelementptr double, ptr %pimag_another, i64 %idx0131
  %pImAnother1148 = getelementptr double, ptr %pimag_another, i64 %idx1132
  %pImAnother2149 = getelementptr double, ptr %pimag_another, i64 %idx2133
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
  %newRe3_159 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Re1152, <2 x double> %newRe3_)
  %newRe3_160 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Re2153, <2 x double> %newRe3_159)
  %newRe3_161 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Re3154, <2 x double> %newRe3_160)
  %newRe3_162 = fmul <2 x double> %mRe28, %Im0155
  %newRe3_163 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe29, <2 x double> %Im1156, <2 x double> %newRe3_162)
  %newRe3_164 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe30, <2 x double> %Im2157, <2 x double> %newRe3_163)
  %newRe3_165 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe31, <2 x double> %Im3158, <2 x double> %newRe3_164)
  %newRe3 = fsub <2 x double> %newRe3_161, %newRe3_165
  %newIm3_ = fmul <2 x double> %mRe12, %Im0155
  %newIm3_166 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Im1156, <2 x double> %newIm3_)
  %newIm3_167 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Im2157, <2 x double> %newIm3_166)
  %newIm3_168 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Im3158, <2 x double> %newIm3_167)
  %newIm3_169 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe28, <2 x double> %Re0151, <2 x double> %newIm3_168)
  %newIm3_170 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe29, <2 x double> %Re1152, <2 x double> %newIm3_169)
  %newIm3_171 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe30, <2 x double> %Re2153, <2 x double> %newIm3_170)
  %newIm3_172 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe31, <2 x double> %Re3154, <2 x double> %newIm3_171)
  store <2 x double> %newRe3, ptr %pReAnother3146, align 16
  store <2 x double> %newIm3_172, ptr %pImAnother3150, align 16
  %idx_next173 = add i64 %idx122, 1
  br label %loop_batch_3

ret:                                              ; preds = %loop_batch_3
  ret void
}


define void @f64_s1_sep_u2q_k2l0(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idxHi = add i64 %idx, 4
  %pReLo = getelementptr <4 x double>, ptr %preal, i64 %idx
  %pReHi = getelementptr <4 x double>, ptr %preal, i64 %idxHi
  %pImLo = getelementptr <4 x double>, ptr %pimag, i64 %idx
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
  %newRe0_1 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Re1, <2 x double> %newRe0_)
  %newRe0_2 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Re2, <2 x double> %newRe0_1)
  %newRe0_3 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Re3, <2 x double> %newRe0_2)
  %newRe0_4 = fmul <2 x double> %mIm0, %Im0
  %newRe0_5 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Im1, <2 x double> %newRe0_4)
  %newRe0_6 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Im2, <2 x double> %newRe0_5)
  %newRe0_7 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Im3, <2 x double> %newRe0_6)
  %newRe0 = fsub <2 x double> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <2 x double> %mRe4, %Re0
  %newRe1_8 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Re1, <2 x double> %newRe1_)
  %newRe1_9 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Re2, <2 x double> %newRe1_8)
  %newRe1_10 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Re3, <2 x double> %newRe1_9)
  %newRe1_11 = fmul <2 x double> %mIm4, %Im0
  %newRe1_12 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Im1, <2 x double> %newRe1_11)
  %newRe1_13 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Im2, <2 x double> %newRe1_12)
  %newRe1_14 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Im3, <2 x double> %newRe1_13)
  %newRe1 = fsub <2 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <2 x double> %mRe8, %Re0
  %newRe2_15 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Re1, <2 x double> %newRe2_)
  %newRe2_16 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Re2, <2 x double> %newRe2_15)
  %newRe2_17 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Re3, <2 x double> %newRe2_16)
  %newRe2_18 = fmul <2 x double> %mIm8, %Im0
  %newRe2_19 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Im1, <2 x double> %newRe2_18)
  %newRe2_20 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Im2, <2 x double> %newRe2_19)
  %newRe2_21 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Im3, <2 x double> %newRe2_20)
  %newRe2 = fsub <2 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <2 x double> %mRe12, %Re0
  %newRe3_22 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Re1, <2 x double> %newRe3_)
  %newRe3_23 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Re2, <2 x double> %newRe3_22)
  %newRe3_24 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Re3, <2 x double> %newRe3_23)
  %newRe3_25 = fmul <2 x double> %mIm12, %Im0
  %newRe3_26 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Im1, <2 x double> %newRe3_25)
  %newRe3_27 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Im2, <2 x double> %newRe3_26)
  %newRe3_28 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Im3, <2 x double> %newRe3_27)
  %newRe3 = fsub <2 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <2 x double> %mRe0, %Im0
  %newIm0_29 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Im1, <2 x double> %newIm0_)
  %newIm0_30 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Im2, <2 x double> %newIm0_29)
  %newIm0_31 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Im3, <2 x double> %newIm0_30)
  %newIm0_32 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm0, <2 x double> %Re0, <2 x double> %newIm0_31)
  %newIm0_33 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Re1, <2 x double> %newIm0_32)
  %newIm0_34 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Re2, <2 x double> %newIm0_33)
  %newIm0_35 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Re3, <2 x double> %newIm0_34)
  %newIm1_ = fmul <2 x double> %mRe4, %Im0
  %newIm1_36 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Im1, <2 x double> %newIm1_)
  %newIm1_37 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Im2, <2 x double> %newIm1_36)
  %newIm1_38 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Im3, <2 x double> %newIm1_37)
  %newIm1_39 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm4, <2 x double> %Re0, <2 x double> %newIm1_38)
  %newIm1_40 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Re1, <2 x double> %newIm1_39)
  %newIm1_41 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Re2, <2 x double> %newIm1_40)
  %newIm1_42 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Re3, <2 x double> %newIm1_41)
  %newIm2_ = fmul <2 x double> %mRe8, %Im0
  %newIm2_43 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Im1, <2 x double> %newIm2_)
  %newIm2_44 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Im2, <2 x double> %newIm2_43)
  %newIm2_45 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Im3, <2 x double> %newIm2_44)
  %newIm2_46 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm8, <2 x double> %Re0, <2 x double> %newIm2_45)
  %newIm2_47 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Re1, <2 x double> %newIm2_46)
  %newIm2_48 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Re2, <2 x double> %newIm2_47)
  %newIm2_49 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Re3, <2 x double> %newIm2_48)
  %newIm3_ = fmul <2 x double> %mRe12, %Im0
  %newIm3_50 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Im1, <2 x double> %newIm3_)
  %newIm3_51 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Im2, <2 x double> %newIm3_50)
  %newIm3_52 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Im3, <2 x double> %newIm3_51)
  %newIm3_53 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm12, <2 x double> %Re0, <2 x double> %newIm3_52)
  %newIm3_54 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Re1, <2 x double> %newIm3_53)
  %newIm3_55 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Re2, <2 x double> %newIm3_54)
  %newIm3_56 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Re3, <2 x double> %newIm3_55)
  %newReLo = shufflevector <2 x double> %newRe0, <2 x double> %newRe1, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  %newReHi = shufflevector <2 x double> %newRe2, <2 x double> %newRe3, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %newReLo, ptr %pReLo, align 32
  store <4 x double> %newReHi, ptr %pReHi, align 32
  %newImLo = shufflevector <2 x double> %newIm0_35, <2 x double> %newIm1_42, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  %newImHi = shufflevector <2 x double> %newIm2_49, <2 x double> %newIm3_56, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  store <4 x double> %newImLo, ptr %pImLo, align 32
  store <4 x double> %newImHi, ptr %pImHi, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}
define void @f64_s1_sep_u2q_k1l0(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pReal = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pImag = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %Real = load <8 x double>, ptr %pReal, align 64
  %Imag = load <8 x double>, ptr %pImag, align 64
  %Re0 = shufflevector <8 x double> %Real, <8 x double> poison, <2 x i32> <i32 0, i32 4>
  %Re1 = shufflevector <8 x double> %Real, <8 x double> poison, <2 x i32> <i32 1, i32 5>
  %Re2 = shufflevector <8 x double> %Real, <8 x double> poison, <2 x i32> <i32 2, i32 6>
  %Re3 = shufflevector <8 x double> %Real, <8 x double> poison, <2 x i32> <i32 3, i32 7>
  %Im0 = shufflevector <8 x double> %Imag, <8 x double> poison, <2 x i32> <i32 0, i32 4>
  %Im1 = shufflevector <8 x double> %Imag, <8 x double> poison, <2 x i32> <i32 1, i32 5>
  %Im2 = shufflevector <8 x double> %Imag, <8 x double> poison, <2 x i32> <i32 2, i32 6>
  %Im3 = shufflevector <8 x double> %Imag, <8 x double> poison, <2 x i32> <i32 3, i32 7>
  %newRe0_ = fmul <2 x double> %mRe0, %Re0
  %newRe0_1 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Re1, <2 x double> %newRe0_)
  %newRe0_2 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Re2, <2 x double> %newRe0_1)
  %newRe0_3 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Re3, <2 x double> %newRe0_2)
  %newRe0_4 = fmul <2 x double> %mIm0, %Im0
  %newRe0_5 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Im1, <2 x double> %newRe0_4)
  %newRe0_6 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Im2, <2 x double> %newRe0_5)
  %newRe0_7 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Im3, <2 x double> %newRe0_6)
  %newRe0 = fsub <2 x double> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <2 x double> %mRe4, %Re0
  %newRe1_8 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Re1, <2 x double> %newRe1_)
  %newRe1_9 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Re2, <2 x double> %newRe1_8)
  %newRe1_10 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Re3, <2 x double> %newRe1_9)
  %newRe1_11 = fmul <2 x double> %mIm4, %Im0
  %newRe1_12 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Im1, <2 x double> %newRe1_11)
  %newRe1_13 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Im2, <2 x double> %newRe1_12)
  %newRe1_14 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Im3, <2 x double> %newRe1_13)
  %newRe1 = fsub <2 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <2 x double> %mRe8, %Re0
  %newRe2_15 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Re1, <2 x double> %newRe2_)
  %newRe2_16 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Re2, <2 x double> %newRe2_15)
  %newRe2_17 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Re3, <2 x double> %newRe2_16)
  %newRe2_18 = fmul <2 x double> %mIm8, %Im0
  %newRe2_19 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Im1, <2 x double> %newRe2_18)
  %newRe2_20 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Im2, <2 x double> %newRe2_19)
  %newRe2_21 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Im3, <2 x double> %newRe2_20)
  %newRe2 = fsub <2 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <2 x double> %mRe12, %Re0
  %newRe3_22 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Re1, <2 x double> %newRe3_)
  %newRe3_23 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Re2, <2 x double> %newRe3_22)
  %newRe3_24 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Re3, <2 x double> %newRe3_23)
  %newRe3_25 = fmul <2 x double> %mIm12, %Im0
  %newRe3_26 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Im1, <2 x double> %newRe3_25)
  %newRe3_27 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Im2, <2 x double> %newRe3_26)
  %newRe3_28 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Im3, <2 x double> %newRe3_27)
  %newRe3 = fsub <2 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <2 x double> %mRe0, %Im0
  %newIm0_29 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Im1, <2 x double> %newIm0_)
  %newIm0_30 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Im2, <2 x double> %newIm0_29)
  %newIm0_31 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Im3, <2 x double> %newIm0_30)
  %newIm0_32 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm0, <2 x double> %Re0, <2 x double> %newIm0_31)
  %newIm0_33 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Re1, <2 x double> %newIm0_32)
  %newIm0_34 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Re2, <2 x double> %newIm0_33)
  %newIm0_35 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Re3, <2 x double> %newIm0_34)
  %newIm1_ = fmul <2 x double> %mRe4, %Im0
  %newIm1_36 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Im1, <2 x double> %newIm1_)
  %newIm1_37 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Im2, <2 x double> %newIm1_36)
  %newIm1_38 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Im3, <2 x double> %newIm1_37)
  %newIm1_39 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm4, <2 x double> %Re0, <2 x double> %newIm1_38)
  %newIm1_40 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Re1, <2 x double> %newIm1_39)
  %newIm1_41 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Re2, <2 x double> %newIm1_40)
  %newIm1_42 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Re3, <2 x double> %newIm1_41)
  %newIm2_ = fmul <2 x double> %mRe8, %Im0
  %newIm2_43 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Im1, <2 x double> %newIm2_)
  %newIm2_44 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Im2, <2 x double> %newIm2_43)
  %newIm2_45 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Im3, <2 x double> %newIm2_44)
  %newIm2_46 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm8, <2 x double> %Re0, <2 x double> %newIm2_45)
  %newIm2_47 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Re1, <2 x double> %newIm2_46)
  %newIm2_48 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Re2, <2 x double> %newIm2_47)
  %newIm2_49 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Re3, <2 x double> %newIm2_48)
  %newIm3_ = fmul <2 x double> %mRe12, %Im0
  %newIm3_50 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Im1, <2 x double> %newIm3_)
  %newIm3_51 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Im2, <2 x double> %newIm3_50)
  %newIm3_52 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Im3, <2 x double> %newIm3_51)
  %newIm3_53 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm12, <2 x double> %Re0, <2 x double> %newIm3_52)
  %newIm3_54 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Re1, <2 x double> %newIm3_53)
  %newIm3_55 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Re2, <2 x double> %newIm3_54)
  %newIm3_56 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Re3, <2 x double> %newIm3_55)
  %vecRe0 = shufflevector <2 x double> %newRe0, <2 x double> %newRe1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %vecRe1 = shufflevector <2 x double> %newRe2, <2 x double> %newRe3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %newReal = shufflevector <4 x double> %vecRe0, <4 x double> %vecRe1, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 1, i32 3, i32 5, i32 7>
  store <8 x double> %newReal, ptr %pReal, align 64
  %vecIm0 = shufflevector <2 x double> %newIm0_35, <2 x double> %newIm1_42, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %vecIm1 = shufflevector <2 x double> %newIm2_49, <2 x double> %newIm3_56, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %newImag = shufflevector <4 x double> %vecIm0, <4 x double> %vecIm1, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 1, i32 3, i32 5, i32 7>
  store <8 x double> %newImag, ptr %pImag, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

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
  %newRe1_ = fmul <4 x double> %mRe4, %Re0
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re1, <4 x double> %newRe1_)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re2, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re3, <4 x double> %newRe1_9)
  %newRe1_11 = fmul <4 x double> %mIm4, %Im0
  %newRe1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im1, <4 x double> %newRe1_11)
  %newRe1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im2, <4 x double> %newRe1_12)
  %newRe1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im3, <4 x double> %newRe1_13)
  %newRe1 = fsub <4 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <4 x double> %mRe8, %Re0
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re1, <4 x double> %newRe2_)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re2, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re3, <4 x double> %newRe2_16)
  %newRe2_18 = fmul <4 x double> %mIm8, %Im0
  %newRe2_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im1, <4 x double> %newRe2_18)
  %newRe2_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im2, <4 x double> %newRe2_19)
  %newRe2_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im3, <4 x double> %newRe2_20)
  %newRe2 = fsub <4 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <4 x double> %mRe12, %Re0
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re1, <4 x double> %newRe3_)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re2, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re3, <4 x double> %newRe3_23)
  %newRe3_25 = fmul <4 x double> %mIm12, %Im0
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


define void @f64_s2_sep_u2q_k5l3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
  %idx00_tmp = or i64 %left, %middle
  %idx00 = or i64 %idx00_tmp, %right
  %idx01 = or i64 %idx00, 8
  %idx10 = or i64 %idx00, 32
  %idx11 = or i64 %idx00, 40
  %pRe00 = getelementptr double, ptr %preal, i64 %idx00
  %pRe01 = getelementptr double, ptr %preal, i64 %idx01
  %pRe10 = getelementptr double, ptr %preal, i64 %idx10
  %pRe11 = getelementptr double, ptr %preal, i64 %idx11
  %pIm00 = getelementptr double, ptr %pimag, i64 %idx00
  %pIm01 = getelementptr double, ptr %pimag, i64 %idx01
  %pIm10 = getelementptr double, ptr %pimag, i64 %idx10
  %pIm11 = getelementptr double, ptr %pimag, i64 %idx11
  %Re00 = load <4 x double>, ptr %pRe00, align 32
  %Re01 = load <4 x double>, ptr %pRe01, align 32
  %Re10 = load <4 x double>, ptr %pRe10, align 32
  %Re11 = load <4 x double>, ptr %pRe11, align 32
  %Im00 = load <4 x double>, ptr %pIm00, align 32
  %Im01 = load <4 x double>, ptr %pIm01, align 32
  %Im10 = load <4 x double>, ptr %pIm10, align 32
  %Im11 = load <4 x double>, ptr %pIm11, align 32
  %newRe0_ = fmul <4 x double> %mRe0, %Re00
  %newRe0_1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Re01, <4 x double> %newRe0_)
  %newRe0_2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Re10, <4 x double> %newRe0_1)
  %newRe0_3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Re11, <4 x double> %newRe0_2)
  %newRe0_4 = fmul <4 x double> %mIm0, %Im00
  %newRe0_5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Im01, <4 x double> %newRe0_4)
  %newRe0_6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Im10, <4 x double> %newRe0_5)
  %newRe0_7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Im11, <4 x double> %newRe0_6)
  %newRe0 = fsub <4 x double> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <4 x double> %mRe4, %Re00
  %newRe1_8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Re01, <4 x double> %newRe1_)
  %newRe1_9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Re10, <4 x double> %newRe1_8)
  %newRe1_10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Re11, <4 x double> %newRe1_9)
  %newRe1_11 = fmul <4 x double> %mIm4, %Im00
  %newRe1_12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Im01, <4 x double> %newRe1_11)
  %newRe1_13 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Im10, <4 x double> %newRe1_12)
  %newRe1_14 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Im11, <4 x double> %newRe1_13)
  %newRe1 = fsub <4 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <4 x double> %mRe8, %Re00
  %newRe2_15 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Re01, <4 x double> %newRe2_)
  %newRe2_16 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Re10, <4 x double> %newRe2_15)
  %newRe2_17 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Re11, <4 x double> %newRe2_16)
  %newRe2_18 = fmul <4 x double> %mIm8, %Im00
  %newRe2_19 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Im01, <4 x double> %newRe2_18)
  %newRe2_20 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Im10, <4 x double> %newRe2_19)
  %newRe2_21 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Im11, <4 x double> %newRe2_20)
  %newRe2 = fsub <4 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <4 x double> %mRe12, %Re00
  %newRe3_22 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Re01, <4 x double> %newRe3_)
  %newRe3_23 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Re10, <4 x double> %newRe3_22)
  %newRe3_24 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Re11, <4 x double> %newRe3_23)
  %newRe3_25 = fmul <4 x double> %mIm12, %Im00
  %newRe3_26 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Im01, <4 x double> %newRe3_25)
  %newRe3_27 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Im10, <4 x double> %newRe3_26)
  %newRe3_28 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Im11, <4 x double> %newRe3_27)
  %newRe3 = fsub <4 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <4 x double> %mRe0, %Im00
  %newIm0_29 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe1, <4 x double> %Im01, <4 x double> %newIm0_)
  %newIm0_30 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe2, <4 x double> %Im10, <4 x double> %newIm0_29)
  %newIm0_31 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe3, <4 x double> %Im11, <4 x double> %newIm0_30)
  %newIm0_32 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm0, <4 x double> %Re00, <4 x double> %newIm0_31)
  %newIm0_33 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm1, <4 x double> %Re01, <4 x double> %newIm0_32)
  %newIm0_34 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm2, <4 x double> %Re10, <4 x double> %newIm0_33)
  %newIm0_35 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm3, <4 x double> %Re11, <4 x double> %newIm0_34)
  %newIm1_ = fmul <4 x double> %mRe4, %Im00
  %newIm1_36 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe5, <4 x double> %Im01, <4 x double> %newIm1_)
  %newIm1_37 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe6, <4 x double> %Im10, <4 x double> %newIm1_36)
  %newIm1_38 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe7, <4 x double> %Im11, <4 x double> %newIm1_37)
  %newIm1_39 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm4, <4 x double> %Re00, <4 x double> %newIm1_38)
  %newIm1_40 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm5, <4 x double> %Re01, <4 x double> %newIm1_39)
  %newIm1_41 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm6, <4 x double> %Re10, <4 x double> %newIm1_40)
  %newIm1_42 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm7, <4 x double> %Re11, <4 x double> %newIm1_41)
  %newIm2_ = fmul <4 x double> %mRe8, %Im00
  %newIm2_43 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe9, <4 x double> %Im01, <4 x double> %newIm2_)
  %newIm2_44 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe10, <4 x double> %Im10, <4 x double> %newIm2_43)
  %newIm2_45 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe11, <4 x double> %Im11, <4 x double> %newIm2_44)
  %newIm2_46 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm8, <4 x double> %Re00, <4 x double> %newIm2_45)
  %newIm2_47 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm9, <4 x double> %Re01, <4 x double> %newIm2_46)
  %newIm2_48 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm10, <4 x double> %Re10, <4 x double> %newIm2_47)
  %newIm2_49 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm11, <4 x double> %Re11, <4 x double> %newIm2_48)
  %newIm3_ = fmul <4 x double> %mRe12, %Im00
  %newIm3_50 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe13, <4 x double> %Im01, <4 x double> %newIm3_)
  %newIm3_51 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe14, <4 x double> %Im10, <4 x double> %newIm3_50)
  %newIm3_52 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mRe15, <4 x double> %Im11, <4 x double> %newIm3_51)
  %newIm3_53 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm12, <4 x double> %Re00, <4 x double> %newIm3_52)
  %newIm3_54 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm13, <4 x double> %Re01, <4 x double> %newIm3_53)
  %newIm3_55 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm14, <4 x double> %Re10, <4 x double> %newIm3_54)
  %newIm3_56 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %mIm15, <4 x double> %Re11, <4 x double> %newIm3_55)
  store <4 x double> %newRe0, ptr %pRe00, align 32
  store <4 x double> %newRe1, ptr %pRe01, align 32
  store <4 x double> %newRe2, ptr %pRe10, align 32
  store <4 x double> %newRe3, ptr %pRe11, align 32
  store <4 x double> %newIm0_35, ptr %pIm00, align 32
  store <4 x double> %newIm1_42, ptr %pIm01, align 32
  store <4 x double> %newIm2_49, ptr %pIm10, align 32
  store <4 x double> %newIm3_56, ptr %pIm11, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}


define void @f64_s1_sep_u2q_k2l1(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %left_tmp = and i64 %idx, -1
  %left = shl i64 %left_tmp, 3
  %middle_tmp = and i64 %idx, 0
  %middle = shl i64 %middle_tmp, 2
  %right_tmp = and i64 %idx, 0
  %right = shl i64 %right_tmp, 1
  %idx00_tmp = or i64 %left, %middle
  %idx00 = or i64 %idx00_tmp, %right
  %idx01 = or i64 %idx00, 2
  %idx10 = or i64 %idx00, 4
  %idx11 = or i64 %idx00, 6
  %pRe00 = getelementptr double, ptr %preal, i64 %idx00
  %pRe01 = getelementptr double, ptr %preal, i64 %idx01
  %pRe10 = getelementptr double, ptr %preal, i64 %idx10
  %pRe11 = getelementptr double, ptr %preal, i64 %idx11
  %pIm00 = getelementptr double, ptr %pimag, i64 %idx00
  %pIm01 = getelementptr double, ptr %pimag, i64 %idx01
  %pIm10 = getelementptr double, ptr %pimag, i64 %idx10
  %pIm11 = getelementptr double, ptr %pimag, i64 %idx11
  %Re00 = load <2 x double>, ptr %pRe00, align 16
  %Re01 = load <2 x double>, ptr %pRe01, align 16
  %Re10 = load <2 x double>, ptr %pRe10, align 16
  %Re11 = load <2 x double>, ptr %pRe11, align 16
  %Im00 = load <2 x double>, ptr %pIm00, align 16
  %Im01 = load <2 x double>, ptr %pIm01, align 16
  %Im10 = load <2 x double>, ptr %pIm10, align 16
  %Im11 = load <2 x double>, ptr %pIm11, align 16
  %newRe0_ = fmul <2 x double> %mRe0, %Re00
  %newRe0_1 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Re01, <2 x double> %newRe0_)
  %newRe0_2 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Re10, <2 x double> %newRe0_1)
  %newRe0_3 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Re11, <2 x double> %newRe0_2)
  %newRe0_4 = fmul <2 x double> %mIm0, %Im00
  %newRe0_5 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Im01, <2 x double> %newRe0_4)
  %newRe0_6 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Im10, <2 x double> %newRe0_5)
  %newRe0_7 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Im11, <2 x double> %newRe0_6)
  %newRe0 = fsub <2 x double> %newRe0_3, %newRe0_7
  %newRe1_ = fmul <2 x double> %mRe4, %Re00
  %newRe1_8 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Re01, <2 x double> %newRe1_)
  %newRe1_9 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Re10, <2 x double> %newRe1_8)
  %newRe1_10 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Re11, <2 x double> %newRe1_9)
  %newRe1_11 = fmul <2 x double> %mIm4, %Im00
  %newRe1_12 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Im01, <2 x double> %newRe1_11)
  %newRe1_13 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Im10, <2 x double> %newRe1_12)
  %newRe1_14 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Im11, <2 x double> %newRe1_13)
  %newRe1 = fsub <2 x double> %newRe1_10, %newRe1_14
  %newRe2_ = fmul <2 x double> %mRe8, %Re00
  %newRe2_15 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Re01, <2 x double> %newRe2_)
  %newRe2_16 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Re10, <2 x double> %newRe2_15)
  %newRe2_17 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Re11, <2 x double> %newRe2_16)
  %newRe2_18 = fmul <2 x double> %mIm8, %Im00
  %newRe2_19 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Im01, <2 x double> %newRe2_18)
  %newRe2_20 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Im10, <2 x double> %newRe2_19)
  %newRe2_21 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Im11, <2 x double> %newRe2_20)
  %newRe2 = fsub <2 x double> %newRe2_17, %newRe2_21
  %newRe3_ = fmul <2 x double> %mRe12, %Re00
  %newRe3_22 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Re01, <2 x double> %newRe3_)
  %newRe3_23 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Re10, <2 x double> %newRe3_22)
  %newRe3_24 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Re11, <2 x double> %newRe3_23)
  %newRe3_25 = fmul <2 x double> %mIm12, %Im00
  %newRe3_26 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Im01, <2 x double> %newRe3_25)
  %newRe3_27 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Im10, <2 x double> %newRe3_26)
  %newRe3_28 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Im11, <2 x double> %newRe3_27)
  %newRe3 = fsub <2 x double> %newRe3_24, %newRe3_28
  %newIm0_ = fmul <2 x double> %mRe0, %Im00
  %newIm0_29 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe1, <2 x double> %Im01, <2 x double> %newIm0_)
  %newIm0_30 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe2, <2 x double> %Im10, <2 x double> %newIm0_29)
  %newIm0_31 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe3, <2 x double> %Im11, <2 x double> %newIm0_30)
  %newIm0_32 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm0, <2 x double> %Re00, <2 x double> %newIm0_31)
  %newIm0_33 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm1, <2 x double> %Re01, <2 x double> %newIm0_32)
  %newIm0_34 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm2, <2 x double> %Re10, <2 x double> %newIm0_33)
  %newIm0_35 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm3, <2 x double> %Re11, <2 x double> %newIm0_34)
  %newIm1_ = fmul <2 x double> %mRe4, %Im00
  %newIm1_36 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe5, <2 x double> %Im01, <2 x double> %newIm1_)
  %newIm1_37 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe6, <2 x double> %Im10, <2 x double> %newIm1_36)
  %newIm1_38 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe7, <2 x double> %Im11, <2 x double> %newIm1_37)
  %newIm1_39 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm4, <2 x double> %Re00, <2 x double> %newIm1_38)
  %newIm1_40 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm5, <2 x double> %Re01, <2 x double> %newIm1_39)
  %newIm1_41 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm6, <2 x double> %Re10, <2 x double> %newIm1_40)
  %newIm1_42 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm7, <2 x double> %Re11, <2 x double> %newIm1_41)
  %newIm2_ = fmul <2 x double> %mRe8, %Im00
  %newIm2_43 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe9, <2 x double> %Im01, <2 x double> %newIm2_)
  %newIm2_44 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe10, <2 x double> %Im10, <2 x double> %newIm2_43)
  %newIm2_45 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe11, <2 x double> %Im11, <2 x double> %newIm2_44)
  %newIm2_46 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm8, <2 x double> %Re00, <2 x double> %newIm2_45)
  %newIm2_47 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm9, <2 x double> %Re01, <2 x double> %newIm2_46)
  %newIm2_48 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm10, <2 x double> %Re10, <2 x double> %newIm2_47)
  %newIm2_49 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm11, <2 x double> %Re11, <2 x double> %newIm2_48)
  %newIm3_ = fmul <2 x double> %mRe12, %Im00
  %newIm3_50 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe13, <2 x double> %Im01, <2 x double> %newIm3_)
  %newIm3_51 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe14, <2 x double> %Im10, <2 x double> %newIm3_50)
  %newIm3_52 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mRe15, <2 x double> %Im11, <2 x double> %newIm3_51)
  %newIm3_53 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm12, <2 x double> %Re00, <2 x double> %newIm3_52)
  %newIm3_54 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm13, <2 x double> %Re01, <2 x double> %newIm3_53)
  %newIm3_55 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm14, <2 x double> %Re10, <2 x double> %newIm3_54)
  %newIm3_56 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %mIm15, <2 x double> %Re11, <2 x double> %newIm3_55)
  store <2 x double> %newRe0, ptr %pRe00, align 16
  store <2 x double> %newRe1, ptr %pRe01, align 16
  store <2 x double> %newRe2, ptr %pRe10, align 16
  store <2 x double> %newRe3, ptr %pRe11, align 16
  store <2 x double> %newIm0_35, ptr %pIm00, align 16
  store <2 x double> %newIm1_42, ptr %pIm01, align 16
  store <2 x double> %newIm2_49, ptr %pIm10, align 16
  store <2 x double> %newIm3_56, ptr %pIm11, align 16
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f32_s2_sep_u3_k1_33330333(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <8 x float>, ptr %pmat, align 32
  %ar = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %ai = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %bi = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pRe = getelementptr <8 x float>, ptr %preal, i64 %idx
  %pIm = getelementptr <8 x float>, ptr %pimag, i64 %idx
  %Re = load <8 x float>, ptr %pRe, align 32
  %Im = load <8 x float>, ptr %pIm, align 32
  %Ar = shufflevector <8 x float> %Re, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Ai = shufflevector <8 x float> %Im, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Br = shufflevector <8 x float> %Re, <8 x float> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Bi = shufflevector <8 x float> %Im, <8 x float> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newAr = fmul <4 x float> %ar, %Ar
  %newAr1 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %br, <4 x float> %Br, <4 x float> %newAr)
  %biBi = fmul <4 x float> %bi, %Bi
  %newAr2 = fsub <4 x float> %newAr1, %biBi
  %newAi = fmul <4 x float> %ar, %Ai
  %newAi3 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %br, <4 x float> %Bi, <4 x float> %newAi)
  %newAi4 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %bi, <4 x float> %Br, <4 x float> %newAi3)
  %newBr = fmul <4 x float> %cr, %Ar
  %newBr5 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %dr, <4 x float> %Br, <4 x float> %newBr)
  %ciAi = fmul <4 x float> %ci, %Ai
  %newBr6 = fsub <4 x float> %newBr5, %ciAi
  %diBi = fmul <4 x float> %di, %Bi
  %newBr7 = fsub <4 x float> %newBr6, %diBi
  %newBi = fmul <4 x float> %cr, %Ai
  %newBi8 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %ci, <4 x float> %Ar, <4 x float> %newBi)
  %newBi9 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %di, <4 x float> %Br, <4 x float> %newBi8)
  %newBi10 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %dr, <4 x float> %Bi, <4 x float> %newBi9)
  %newRe = shufflevector <4 x float> %newAr2, <4 x float> %newBr7, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newIm = shufflevector <4 x float> %newAi4, <4 x float> %newBi10, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x float> %newRe, ptr %pRe, align 32
  store <8 x float> %newIm, ptr %pIm, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f32_s2_sep_u3_k0_33330333(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <8 x float>, ptr %pmat, align 32
  %ar = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> zeroinitializer
  %br = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %ai = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %bi = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x float> %mat, <8 x float> poison, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pRe = getelementptr <8 x float>, ptr %preal, i64 %idx
  %pIm = getelementptr <8 x float>, ptr %pimag, i64 %idx
  %Re = load <8 x float>, ptr %pRe, align 32
  %Im = load <8 x float>, ptr %pIm, align 32
  %Ar = shufflevector <8 x float> %Re, <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Ai = shufflevector <8 x float> %Im, <8 x float> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Br = shufflevector <8 x float> %Re, <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Bi = shufflevector <8 x float> %Im, <8 x float> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newAr = fmul <4 x float> %ar, %Ar
  %newAr1 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %br, <4 x float> %Br, <4 x float> %newAr)
  %biBi = fmul <4 x float> %bi, %Bi
  %newAr2 = fsub <4 x float> %newAr1, %biBi
  %newAi = fmul <4 x float> %ar, %Ai
  %newAi3 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %br, <4 x float> %Bi, <4 x float> %newAi)
  %newAi4 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %bi, <4 x float> %Br, <4 x float> %newAi3)
  %newBr = fmul <4 x float> %cr, %Ar
  %newBr5 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %dr, <4 x float> %Br, <4 x float> %newBr)
  %ciAi = fmul <4 x float> %ci, %Ai
  %newBr6 = fsub <4 x float> %newBr5, %ciAi
  %diBi = fmul <4 x float> %di, %Bi
  %newBr7 = fsub <4 x float> %newBr6, %diBi
  %newBi = fmul <4 x float> %cr, %Ai
  %newBi8 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %ci, <4 x float> %Ar, <4 x float> %newBi)
  %newBi9 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %di, <4 x float> %Br, <4 x float> %newBi8)
  %newBi10 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %dr, <4 x float> %Bi, <4 x float> %newBi9)
  %newRe = shufflevector <4 x float> %newAr2, <4 x float> %newBr7, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newIm = shufflevector <4 x float> %newAi4, <4 x float> %newBi10, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x float> %newRe, ptr %pRe, align 32
  store <8 x float> %newIm, ptr %pIm, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}


define void @f64_sep_u3_k1_33330333(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
  %pRe = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pIm = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %Re = load <8 x double>, ptr %pRe, align 64
  %Im = load <8 x double>, ptr %pIm, align 64
  %Ar = shufflevector <8 x double> %Re, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Ai = shufflevector <8 x double> %Im, <8 x double> poison, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %Br = shufflevector <8 x double> %Re, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %Bi = shufflevector <8 x double> %Im, <8 x double> poison, <4 x i32> <i32 2, i32 3, i32 6, i32 7>
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %biBi = fmul <4 x double> %bi, %Bi
  %newAr2 = fsub <4 x double> %newAr1, %biBi
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi)
  %newAi4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi, <4 x double> %Br, <4 x double> %newAi3)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci, %Ai
  %newBr6 = fsub <4 x double> %newBr5, %ciAi
  %diBi = fmul <4 x double> %di, %Bi
  %newBr7 = fsub <4 x double> %newBr6, %diBi
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci, <4 x double> %Ar, <4 x double> %newBi)
  %newBi9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di, <4 x double> %Br, <4 x double> %newBi8)
  %newBi10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi9)
  %newRe = shufflevector <4 x double> %newAr2, <4 x double> %newBr7, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  %newIm = shufflevector <4 x double> %newAi4, <4 x double> %newBi10, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 2, i32 3, i32 6, i32 7>
  store <8 x double> %newRe, ptr %pRe, align 64
  store <8 x double> %newIm, ptr %pIm, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f64_sep_u3_k0_33330333(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
  %pRe = getelementptr <8 x double>, ptr %preal, i64 %idx
  %pIm = getelementptr <8 x double>, ptr %pimag, i64 %idx
  %Re = load <8 x double>, ptr %pRe, align 64
  %Im = load <8 x double>, ptr %pIm, align 64
  %Ar = shufflevector <8 x double> %Re, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Ai = shufflevector <8 x double> %Im, <8 x double> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %Br = shufflevector <8 x double> %Re, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %Bi = shufflevector <8 x double> %Im, <8 x double> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %biBi = fmul <4 x double> %bi, %Bi
  %newAr2 = fsub <4 x double> %newAr1, %biBi
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi)
  %newAi4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi, <4 x double> %Br, <4 x double> %newAi3)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci, %Ai
  %newBr6 = fsub <4 x double> %newBr5, %ciAi
  %diBi = fmul <4 x double> %di, %Bi
  %newBr7 = fsub <4 x double> %newBr6, %diBi
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi8 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci, <4 x double> %Ar, <4 x double> %newBi)
  %newBi9 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di, <4 x double> %Br, <4 x double> %newBi8)
  %newBi10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi9)
  %newRe = shufflevector <4 x double> %newAr2, <4 x double> %newBr7, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %newIm = shufflevector <4 x double> %newAi4, <4 x double> %newBi10, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  store <8 x double> %newRe, ptr %pRe, align 64
  store <8 x double> %newIm, ptr %pIm, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @u3_f64_alt_0000ffff(ptr %psv, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <8 x double>, ptr %pmat, align 64
  %ar = shufflevector <8 x double> %mat, <8 x double> poison, <2 x i32> zeroinitializer
  %br = shufflevector <8 x double> %mat, <8 x double> poison, <2 x i32> <i32 1, i32 1>
  %cr = shufflevector <8 x double> %mat, <8 x double> poison, <2 x i32> <i32 2, i32 2>
  %dr = shufflevector <8 x double> %mat, <8 x double> poison, <2 x i32> <i32 3, i32 3>
  %ai = shufflevector <8 x double> %mat, <8 x double> poison, <2 x i32> <i32 4, i32 4>
  %bi = shufflevector <8 x double> %mat, <8 x double> poison, <2 x i32> <i32 5, i32 5>
  %ci = shufflevector <8 x double> %mat, <8 x double> poison, <2 x i32> <i32 6, i32 6>
  %di = shufflevector <8 x double> %mat, <8 x double> poison, <2 x i32> <i32 7, i32 7>
  %ai_n = fmul <2 x double> %ai, <double 1.000000e+00, double -1.000000e+00>
  %bi_n = fmul <2 x double> %bi, <double 1.000000e+00, double -1.000000e+00>
  %ci_n = fmul <2 x double> %ci, <double 1.000000e+00, double -1.000000e+00>
  %di_n = fmul <2 x double> %di, <double 1.000000e+00, double -1.000000e+00>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idx_and_outer = and i64 %idx, -1
  %shl_outer = shl i64 %idx_and_outer, 2
  %idx_and_inner = and i64 %idx, 0
  %shl_inner = shl i64 %idx_and_inner, 1
  %alpha = add i64 %shl_outer, %shl_inner
  %beta = add i64 %alpha, 2
  %ptrLo = getelementptr double, ptr %psv, i64 %alpha
  %ptrHi = getelementptr double, ptr %psv, i64 %beta
  %Lo = load <2 x double>, ptr %ptrLo, align 16
  %Hi = load <2 x double>, ptr %ptrHi, align 16
  %LoRe = fmul <2 x double> %ar, %Lo
  %LoRe1 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %br, <2 x double> %Hi, <2 x double> %LoRe)
  %LoIm_s = fmul <2 x double> %ai_n, %Lo
  %LoIm_s2 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %bi_n, <2 x double> %Hi, <2 x double> %LoIm_s)
  %LoIm = shufflevector <2 x double> %LoIm_s2, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %newLo = fadd <2 x double> %LoRe1, %LoIm
  %HiRe = fmul <2 x double> %cr, %Lo
  %HiRe3 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %dr, <2 x double> %Hi, <2 x double> %HiRe)
  %HiIm_s = fmul <2 x double> %ci_n, %Lo
  %HiIm_s4 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %di_n, <2 x double> %Hi, <2 x double> %HiIm_s)
  %HiIm = shufflevector <2 x double> %HiIm_s4, <2 x double> poison, <2 x i32> <i32 1, i32 0>
  %newHi = fadd <2 x double> %HiRe3, %HiIm
  store <2 x double> %newLo, ptr %ptrLo, align 16
  store <2 x double> %newHi, ptr %ptrHi, align 16
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @u3_f64_alt_0200ffff(ptr %psv, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
  %ai_n = fmul <4 x double> %ai, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %bi_n = fmul <4 x double> %bi, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %ci_n = fmul <4 x double> %ci, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  %di_n = fmul <4 x double> %di, <double 1.000000e+00, double -1.000000e+00, double 1.000000e+00, double -1.000000e+00>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idx_and_outer = and i64 %idx, -2
  %shl_outer = shl i64 %idx_and_outer, 3
  %idx_and_inner = and i64 %idx, 1
  %shl_inner = shl i64 %idx_and_inner, 2
  %alpha = add i64 %shl_outer, %shl_inner
  %beta = add i64 %alpha, 8
  %ptrLo = getelementptr double, ptr %psv, i64 %alpha
  %ptrHi = getelementptr double, ptr %psv, i64 %beta
  %Lo = load <4 x double>, ptr %ptrLo, align 32
  %Hi = load <4 x double>, ptr %ptrHi, align 32
  %LoRe = fmul <4 x double> %ar, %Lo
  %LoRe1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Hi, <4 x double> %LoRe)
  %LoIm_s = fmul <4 x double> %ai_n, %Lo
  %LoIm_s2 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi_n, <4 x double> %Hi, <4 x double> %LoIm_s)
  %LoIm = shufflevector <4 x double> %LoIm_s2, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %newLo = fadd <4 x double> %LoRe1, %LoIm
  %HiRe = fmul <4 x double> %cr, %Lo
  %HiRe3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Hi, <4 x double> %HiRe)
  %HiIm_s = fmul <4 x double> %ci_n, %Lo
  %HiIm_s4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di_n, <4 x double> %Hi, <4 x double> %HiIm_s)
  %HiIm = shufflevector <4 x double> %HiIm_s4, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %newHi = fadd <4 x double> %HiRe3, %HiIm
  store <4 x double> %newLo, ptr %ptrLo, align 32
  store <4 x double> %newHi, ptr %ptrHi, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}


define void @u3_f64_sep_0200ffff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
  %idx_and_outer = and i64 %idx, -1
  %shl_outer = shl i64 %idx_and_outer, 3
  %idx_and_inner = and i64 %idx, 0
  %shl_inner = shl i64 %idx_and_inner, 2
  %alpha = add i64 %shl_outer, %shl_inner
  %beta = add i64 %alpha, 4
  %ptrAr = getelementptr double, ptr %preal, i64 %alpha
  %ptrAi = getelementptr double, ptr %pimag, i64 %alpha
  %ptrBr = getelementptr double, ptr %preal, i64 %beta
  %ptrBi = getelementptr double, ptr %pimag, i64 %beta
  %Ar = load <4 x double>, ptr %ptrAr, align 32
  %Ai = load <4 x double>, ptr %ptrAi, align 32
  %Br = load <4 x double>, ptr %ptrBr, align 32
  %Bi = load <4 x double>, ptr %ptrBi, align 32
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %aiAi = fmul <4 x double> %ai, %Ai
  %newAr2 = fsub <4 x double> %newAr1, %aiAi
  %biBi = fmul <4 x double> %bi, %Bi
  %newAr3 = fsub <4 x double> %newAr2, %biBi
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ai, <4 x double> %Ar, <4 x double> %newAi)
  %newAi5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi4)
  %newAi6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi, <4 x double> %Br, <4 x double> %newAi5)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci, %Ai
  %newBr8 = fsub <4 x double> %newBr7, %ciAi
  %diBi = fmul <4 x double> %di, %Bi
  %newBr9 = fsub <4 x double> %newBr8, %diBi
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci, <4 x double> %Ar, <4 x double> %newBi)
  %newBi11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di, <4 x double> %Br, <4 x double> %newBi10)
  %newBi12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi11)
  store <4 x double> %newAr3, ptr %ptrAr, align 32
  store <4 x double> %newAi6, ptr %ptrAi, align 32
  store <4 x double> %newBr9, ptr %ptrBr, align 32
  store <4 x double> %newBi12, ptr %ptrBi, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #0
; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #0
; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
