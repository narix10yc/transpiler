; ModuleID = 'myModule'
source_filename = "myModule"

define void @u3_0_02003fff(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) {
entry:
  %arElem = extractelement <8 x double> %mat, i64 0
  %brElem = extractelement <8 x double> %mat, i64 1
  %crElem = extractelement <8 x double> %mat, i64 2
  %drElem = extractelement <8 x double> %mat, i64 3
  %aiElem = extractelement <8 x double> %mat, i64 4
  %biElem = extractelement <8 x double> %mat, i64 5
  %ciElem = extractelement <8 x double> %mat, i64 6
  %diElem = extractelement <8 x double> %mat, i64 7
  %ar_insert_0 = insertelement <4 x double> undef, double %arElem, i64 0
  %ar_insert_1 = insertelement <4 x double> %ar_insert_0, double %arElem, i64 1
  %ar_insert_2 = insertelement <4 x double> %ar_insert_1, double %arElem, i64 2
  %ar_vec = insertelement <4 x double> %ar_insert_2, double %arElem, i64 3
  %br_insert_0 = insertelement <4 x double> undef, double %brElem, i64 0
  %br_insert_1 = insertelement <4 x double> %br_insert_0, double %brElem, i64 1
  %br_insert_2 = insertelement <4 x double> %br_insert_1, double %brElem, i64 2
  %br_vec = insertelement <4 x double> %br_insert_2, double %brElem, i64 3
  %cr_insert_0 = insertelement <4 x double> undef, double %crElem, i64 0
  %cr_insert_1 = insertelement <4 x double> %cr_insert_0, double %crElem, i64 1
  %cr_insert_2 = insertelement <4 x double> %cr_insert_1, double %crElem, i64 2
  %cr_vec = insertelement <4 x double> %cr_insert_2, double %crElem, i64 3
  %dr_insert_0 = insertelement <4 x double> undef, double %drElem, i64 0
  %dr_insert_1 = insertelement <4 x double> %dr_insert_0, double %drElem, i64 1
  %dr_insert_2 = insertelement <4 x double> %dr_insert_1, double %drElem, i64 2
  %dr_vec = insertelement <4 x double> %dr_insert_2, double %drElem, i64 3
  %ai_insert_0 = insertelement <4 x double> undef, double %aiElem, i64 0
  %ai_insert_1 = insertelement <4 x double> %ai_insert_0, double %aiElem, i64 1
  %ai_insert_2 = insertelement <4 x double> %ai_insert_1, double %aiElem, i64 2
  %ai_vec = insertelement <4 x double> %ai_insert_2, double %aiElem, i64 3
  %bi_insert_0 = insertelement <4 x double> undef, double %biElem, i64 0
  %bi_insert_1 = insertelement <4 x double> %bi_insert_0, double %biElem, i64 1
  %bi_insert_2 = insertelement <4 x double> %bi_insert_1, double %biElem, i64 2
  %bi_vec = insertelement <4 x double> %bi_insert_2, double %biElem, i64 3
  %ci_insert_0 = insertelement <4 x double> undef, double %ciElem, i64 0
  %ci_insert_1 = insertelement <4 x double> %ci_insert_0, double %ciElem, i64 1
  %ci_insert_2 = insertelement <4 x double> %ci_insert_1, double %ciElem, i64 2
  %ci_vec = insertelement <4 x double> %ci_insert_2, double %ciElem, i64 3
  %di_insert_0 = insertelement <4 x double> undef, double %diElem, i64 0
  %di_insert_1 = insertelement <4 x double> %di_insert_0, double %diElem, i64 1
  %di_insert_2 = insertelement <4 x double> %di_insert_1, double %diElem, i64 2
  %di_vec = insertelement <4 x double> %di_insert_2, double %diElem, i64 3
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
  %newAr = fmul <4 x double> %ar_vec, %Ar
  %newAr1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Br, <4 x double> %newAr)
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr2 = fsub <4 x double> %newAr1, %aiAi
  %biBi = fmul <4 x double> %bi_vec, %Bi
  %newAr3 = fsub <4 x double> %newAr2, %biBi
  %newAi = fmul <4 x double> %ar_vec, %Ai
  %newAi4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ai_vec, <4 x double> %Ar, <4 x double> %newAi)
  %newAi5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Bi, <4 x double> %newAi4)
  %newAi6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi_vec, <4 x double> %Br, <4 x double> %newAi5)
  %newBr = fmul <4 x double> %cr_vec, %Ar
  %newBr7 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci_vec, %Ai
  %newBr8 = fsub <4 x double> %newBr7, %ciAi
  %diBi = fmul <4 x double> %di_vec, %Bi
  %newBr9 = fsub <4 x double> %newBr8, %diBi
  %newBi = fmul <4 x double> %cr_vec, %Ai
  %newBi10 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci_vec, <4 x double> %Ar, <4 x double> %newBi)
  %newBi11 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di_vec, <4 x double> %Br, <4 x double> %newBi10)
  %newBi12 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Bi, <4 x double> %newBi11)
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
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #0

define void @u3_1_02001080(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) {
entry:
  %arElem = extractelement <8 x double> %mat, i64 0
  %brElem = extractelement <8 x double> %mat, i64 1
  %crElem = extractelement <8 x double> %mat, i64 2
  %drElem = extractelement <8 x double> %mat, i64 3
  %aiElem = extractelement <8 x double> %mat, i64 4
  %biElem = extractelement <8 x double> %mat, i64 5
  %ciElem = extractelement <8 x double> %mat, i64 6
  %diElem = extractelement <8 x double> %mat, i64 7
  %ar_insert_0 = insertelement <4 x double> undef, double %arElem, i64 0
  %ar_insert_1 = insertelement <4 x double> %ar_insert_0, double %arElem, i64 1
  %ar_insert_2 = insertelement <4 x double> %ar_insert_1, double %arElem, i64 2
  %ar_vec = insertelement <4 x double> %ar_insert_2, double %arElem, i64 3
  %br_insert_0 = insertelement <4 x double> undef, double %brElem, i64 0
  %br_insert_1 = insertelement <4 x double> %br_insert_0, double %brElem, i64 1
  %br_insert_2 = insertelement <4 x double> %br_insert_1, double %brElem, i64 2
  %br_vec = insertelement <4 x double> %br_insert_2, double %brElem, i64 3
  %cr_insert_0 = insertelement <4 x double> undef, double %crElem, i64 0
  %cr_insert_1 = insertelement <4 x double> %cr_insert_0, double %crElem, i64 1
  %cr_insert_2 = insertelement <4 x double> %cr_insert_1, double %crElem, i64 2
  %cr_vec = insertelement <4 x double> %cr_insert_2, double %crElem, i64 3
  %dr_insert_0 = insertelement <4 x double> undef, double %drElem, i64 0
  %dr_insert_1 = insertelement <4 x double> %dr_insert_0, double %drElem, i64 1
  %dr_insert_2 = insertelement <4 x double> %dr_insert_1, double %drElem, i64 2
  %dr_vec = insertelement <4 x double> %dr_insert_2, double %drElem, i64 3
  %ai_insert_0 = insertelement <4 x double> undef, double %aiElem, i64 0
  %ai_insert_1 = insertelement <4 x double> %ai_insert_0, double %aiElem, i64 1
  %ai_insert_2 = insertelement <4 x double> %ai_insert_1, double %aiElem, i64 2
  %ai_vec = insertelement <4 x double> %ai_insert_2, double %aiElem, i64 3
  %bi_insert_0 = insertelement <4 x double> undef, double %biElem, i64 0
  %bi_insert_1 = insertelement <4 x double> %bi_insert_0, double %biElem, i64 1
  %bi_insert_2 = insertelement <4 x double> %bi_insert_1, double %biElem, i64 2
  %bi_vec = insertelement <4 x double> %bi_insert_2, double %biElem, i64 3
  %ci_insert_0 = insertelement <4 x double> undef, double %ciElem, i64 0
  %ci_insert_1 = insertelement <4 x double> %ci_insert_0, double %ciElem, i64 1
  %ci_insert_2 = insertelement <4 x double> %ci_insert_1, double %ciElem, i64 2
  %ci_vec = insertelement <4 x double> %ci_insert_2, double %ciElem, i64 3
  %di_insert_0 = insertelement <4 x double> undef, double %diElem, i64 0
  %di_insert_1 = insertelement <4 x double> %di_insert_0, double %diElem, i64 1
  %di_insert_2 = insertelement <4 x double> %di_insert_1, double %diElem, i64 2
  %di_vec = insertelement <4 x double> %di_insert_2, double %diElem, i64 3
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
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr = fsub <4 x double> %Ar, %aiAi
  %newAi = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ai_vec, <4 x double> %Ar, <4 x double> %Ai)
  %newBr = fneg <4 x double> %Br
  %newBi = fneg <4 x double> %Bi
  store <4 x double> %newAr, ptr %ptrAr, align 32
  store <4 x double> %newAi, ptr %ptrAi, align 32
  store <4 x double> %newBr, ptr %ptrBr, align 32
  store <4 x double> %newBi, ptr %ptrBi, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @u3_2_02003fc0(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) {
entry:
  %arElem = extractelement <8 x double> %mat, i64 0
  %brElem = extractelement <8 x double> %mat, i64 1
  %crElem = extractelement <8 x double> %mat, i64 2
  %drElem = extractelement <8 x double> %mat, i64 3
  %aiElem = extractelement <8 x double> %mat, i64 4
  %biElem = extractelement <8 x double> %mat, i64 5
  %ciElem = extractelement <8 x double> %mat, i64 6
  %diElem = extractelement <8 x double> %mat, i64 7
  %ar_insert_0 = insertelement <4 x double> undef, double %arElem, i64 0
  %ar_insert_1 = insertelement <4 x double> %ar_insert_0, double %arElem, i64 1
  %ar_insert_2 = insertelement <4 x double> %ar_insert_1, double %arElem, i64 2
  %ar_vec = insertelement <4 x double> %ar_insert_2, double %arElem, i64 3
  %br_insert_0 = insertelement <4 x double> undef, double %brElem, i64 0
  %br_insert_1 = insertelement <4 x double> %br_insert_0, double %brElem, i64 1
  %br_insert_2 = insertelement <4 x double> %br_insert_1, double %brElem, i64 2
  %br_vec = insertelement <4 x double> %br_insert_2, double %brElem, i64 3
  %cr_insert_0 = insertelement <4 x double> undef, double %crElem, i64 0
  %cr_insert_1 = insertelement <4 x double> %cr_insert_0, double %crElem, i64 1
  %cr_insert_2 = insertelement <4 x double> %cr_insert_1, double %crElem, i64 2
  %cr_vec = insertelement <4 x double> %cr_insert_2, double %crElem, i64 3
  %dr_insert_0 = insertelement <4 x double> undef, double %drElem, i64 0
  %dr_insert_1 = insertelement <4 x double> %dr_insert_0, double %drElem, i64 1
  %dr_insert_2 = insertelement <4 x double> %dr_insert_1, double %drElem, i64 2
  %dr_vec = insertelement <4 x double> %dr_insert_2, double %drElem, i64 3
  %ai_insert_0 = insertelement <4 x double> undef, double %aiElem, i64 0
  %ai_insert_1 = insertelement <4 x double> %ai_insert_0, double %aiElem, i64 1
  %ai_insert_2 = insertelement <4 x double> %ai_insert_1, double %aiElem, i64 2
  %ai_vec = insertelement <4 x double> %ai_insert_2, double %aiElem, i64 3
  %bi_insert_0 = insertelement <4 x double> undef, double %biElem, i64 0
  %bi_insert_1 = insertelement <4 x double> %bi_insert_0, double %biElem, i64 1
  %bi_insert_2 = insertelement <4 x double> %bi_insert_1, double %biElem, i64 2
  %bi_vec = insertelement <4 x double> %bi_insert_2, double %biElem, i64 3
  %ci_insert_0 = insertelement <4 x double> undef, double %ciElem, i64 0
  %ci_insert_1 = insertelement <4 x double> %ci_insert_0, double %ciElem, i64 1
  %ci_insert_2 = insertelement <4 x double> %ci_insert_1, double %ciElem, i64 2
  %ci_vec = insertelement <4 x double> %ci_insert_2, double %ciElem, i64 3
  %di_insert_0 = insertelement <4 x double> undef, double %diElem, i64 0
  %di_insert_1 = insertelement <4 x double> %di_insert_0, double %diElem, i64 1
  %di_insert_2 = insertelement <4 x double> %di_insert_1, double %diElem, i64 2
  %di_vec = insertelement <4 x double> %di_insert_2, double %diElem, i64 3
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
  %newAr = fmul <4 x double> %ar_vec, %Ar
  %newAr1 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Br, <4 x double> %newAr)
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr2 = fsub <4 x double> %newAr1, %aiAi
  %newAi = fmul <4 x double> %ar_vec, %Ai
  %newAi3 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ai_vec, <4 x double> %Ar, <4 x double> %newAi)
  %newAi4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Bi, <4 x double> %newAi3)
  %newBr = fmul <4 x double> %cr_vec, %Ar
  %newBr5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Br, <4 x double> %newBr)
  %newBi = fmul <4 x double> %cr_vec, %Ai
  %newBi6 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Bi, <4 x double> %newBi)
  store <4 x double> %newAr2, ptr %ptrAr, align 32
  store <4 x double> %newAi4, ptr %ptrAi, align 32
  store <4 x double> %newBr5, ptr %ptrBr, align 32
  store <4 x double> %newBi6, ptr %ptrBi, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
