; ModuleID = '../performance/gen_file.ll'
source_filename = "myModule"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @u3_f64_0_00003fff(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) local_unnamed_addr #0 {
entry:
  %ar_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> zeroinitializer
  %br_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %ai_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %bi_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %idx_and_inner = shl i64 %idx2, 2
  %beta = or i64 %idx_and_inner, 1
  %ptrAr = getelementptr double, ptr %preal, i64 %idx_and_inner
  %ptrAi = getelementptr double, ptr %pimag, i64 %idx_and_inner
  %ptrBr = getelementptr double, ptr %preal, i64 %beta
  %ptrBi = getelementptr double, ptr %pimag, i64 %beta
  %Ar = load <4 x double>, ptr %ptrAr, align 32
  %Ai = load <4 x double>, ptr %ptrAi, align 32
  %Br = load <4 x double>, ptr %ptrBr, align 32
  %Bi = load <4 x double>, ptr %ptrBi, align 32
  %newAr = fmul <4 x double> %ar_vec, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Br, <4 x double> %newAr)
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr2 = fsub <4 x double> %newAr1, %aiAi
  %biBi = fmul <4 x double> %bi_vec, %Bi
  %newAr3 = fsub <4 x double> %newAr2, %biBi
  %newAi = fmul <4 x double> %ar_vec, %Ai
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ai_vec, <4 x double> %Ar, <4 x double> %newAi)
  %newAi5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Bi, <4 x double> %newAi4)
  %newAi6 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi_vec, <4 x double> %Br, <4 x double> %newAi5)
  %newBr = fmul <4 x double> %cr_vec, %Ar
  %newBr7 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci_vec, %Ai
  %newBr8 = fsub <4 x double> %newBr7, %ciAi
  %diBi = fmul <4 x double> %di_vec, %Bi
  %newBr9 = fsub <4 x double> %newBr8, %diBi
  %newBi = fmul <4 x double> %cr_vec, %Ai
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci_vec, <4 x double> %Ar, <4 x double> %newBi)
  %newBi11 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di_vec, <4 x double> %Br, <4 x double> %newBi10)
  %newBi12 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Bi, <4 x double> %newBi11)
  store <4 x double> %newAr3, ptr %ptrAr, align 32
  store <4 x double> %newAi6, ptr %ptrAi, align 32
  store <4 x double> %newBr9, ptr %ptrBr, align 32
  store <4 x double> %newBi12, ptr %ptrBi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #1

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @u3_f64_1_01003fff(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) local_unnamed_addr #0 {
entry:
  %ar_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> zeroinitializer
  %br_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %ai_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %bi_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %shl_inner = shl i64 %idx2, 2
  %beta = or i64 %shl_inner, 2
  %ptrAr = getelementptr double, ptr %preal, i64 %shl_inner
  %ptrAi = getelementptr double, ptr %pimag, i64 %shl_inner
  %ptrBr = getelementptr double, ptr %preal, i64 %beta
  %ptrBi = getelementptr double, ptr %pimag, i64 %beta
  %Ar = load <4 x double>, ptr %ptrAr, align 32
  %Ai = load <4 x double>, ptr %ptrAi, align 32
  %Br = load <4 x double>, ptr %ptrBr, align 32
  %Bi = load <4 x double>, ptr %ptrBi, align 32
  %newAr = fmul <4 x double> %ar_vec, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Br, <4 x double> %newAr)
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr2 = fsub <4 x double> %newAr1, %aiAi
  %biBi = fmul <4 x double> %bi_vec, %Bi
  %newAr3 = fsub <4 x double> %newAr2, %biBi
  %newAi = fmul <4 x double> %ar_vec, %Ai
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ai_vec, <4 x double> %Ar, <4 x double> %newAi)
  %newAi5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Bi, <4 x double> %newAi4)
  %newAi6 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi_vec, <4 x double> %Br, <4 x double> %newAi5)
  %newBr = fmul <4 x double> %cr_vec, %Ar
  %newBr7 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci_vec, %Ai
  %newBr8 = fsub <4 x double> %newBr7, %ciAi
  %diBi = fmul <4 x double> %di_vec, %Bi
  %newBr9 = fsub <4 x double> %newBr8, %diBi
  %newBi = fmul <4 x double> %cr_vec, %Ai
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci_vec, <4 x double> %Ar, <4 x double> %newBi)
  %newBi11 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di_vec, <4 x double> %Br, <4 x double> %newBi10)
  %newBi12 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Bi, <4 x double> %newBi11)
  store <4 x double> %newAr3, ptr %ptrAr, align 32
  store <4 x double> %newAi6, ptr %ptrAi, align 32
  store <4 x double> %newBr9, ptr %ptrBr, align 32
  store <4 x double> %newBi12, ptr %ptrBi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @u3_f64_2_02003fff(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) local_unnamed_addr #0 {
entry:
  %ar_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> zeroinitializer
  %br_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %ai_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %bi_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %shl_outer = shl i64 %idx2, 3
  %beta = or i64 %shl_outer, 4
  %ptrAr = getelementptr double, ptr %preal, i64 %shl_outer
  %ptrAi = getelementptr double, ptr %pimag, i64 %shl_outer
  %ptrBr = getelementptr double, ptr %preal, i64 %beta
  %ptrBi = getelementptr double, ptr %pimag, i64 %beta
  %Ar = load <4 x double>, ptr %ptrAr, align 32
  %Ai = load <4 x double>, ptr %ptrAi, align 32
  %Br = load <4 x double>, ptr %ptrBr, align 32
  %Bi = load <4 x double>, ptr %ptrBi, align 32
  %newAr = fmul <4 x double> %ar_vec, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Br, <4 x double> %newAr)
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr2 = fsub <4 x double> %newAr1, %aiAi
  %biBi = fmul <4 x double> %bi_vec, %Bi
  %newAr3 = fsub <4 x double> %newAr2, %biBi
  %newAi = fmul <4 x double> %ar_vec, %Ai
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ai_vec, <4 x double> %Ar, <4 x double> %newAi)
  %newAi5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Bi, <4 x double> %newAi4)
  %newAi6 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi_vec, <4 x double> %Br, <4 x double> %newAi5)
  %newBr = fmul <4 x double> %cr_vec, %Ar
  %newBr7 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci_vec, %Ai
  %newBr8 = fsub <4 x double> %newBr7, %ciAi
  %diBi = fmul <4 x double> %di_vec, %Bi
  %newBr9 = fsub <4 x double> %newBr8, %diBi
  %newBi = fmul <4 x double> %cr_vec, %Ai
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci_vec, <4 x double> %Ar, <4 x double> %newBi)
  %newBi11 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di_vec, <4 x double> %Br, <4 x double> %newBi10)
  %newBi12 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Bi, <4 x double> %newBi11)
  store <4 x double> %newAr3, ptr %ptrAr, align 32
  store <4 x double> %newAi6, ptr %ptrAi, align 32
  store <4 x double> %newBr9, ptr %ptrBr, align 32
  store <4 x double> %newBi12, ptr %ptrBi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @u3_f64_3_03003fff(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) local_unnamed_addr #0 {
entry:
  %ar_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> zeroinitializer
  %br_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %ai_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
  %bi_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  %ci_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 6, i32 6, i32 6, i32 6>
  %di_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %idx_and_outer = shl i64 %idx2, 3
  %shl_outer = and i64 %idx_and_outer, -16
  %idx_and_inner = shl i64 %idx2, 2
  %shl_inner = and i64 %idx_and_inner, 4
  %alpha = or i64 %shl_outer, %shl_inner
  %beta = or i64 %alpha, 8
  %ptrAr = getelementptr double, ptr %preal, i64 %alpha
  %ptrAi = getelementptr double, ptr %pimag, i64 %alpha
  %ptrBr = getelementptr double, ptr %preal, i64 %beta
  %ptrBi = getelementptr double, ptr %pimag, i64 %beta
  %Ar = load <4 x double>, ptr %ptrAr, align 32
  %Ai = load <4 x double>, ptr %ptrAi, align 32
  %Br = load <4 x double>, ptr %ptrBr, align 32
  %Bi = load <4 x double>, ptr %ptrBi, align 32
  %newAr = fmul <4 x double> %ar_vec, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Br, <4 x double> %newAr)
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr2 = fsub <4 x double> %newAr1, %aiAi
  %biBi = fmul <4 x double> %bi_vec, %Bi
  %newAr3 = fsub <4 x double> %newAr2, %biBi
  %newAi = fmul <4 x double> %ar_vec, %Ai
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ai_vec, <4 x double> %Ar, <4 x double> %newAi)
  %newAi5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br_vec, <4 x double> %Bi, <4 x double> %newAi4)
  %newAi6 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi_vec, <4 x double> %Br, <4 x double> %newAi5)
  %newBr = fmul <4 x double> %cr_vec, %Ar
  %newBr7 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci_vec, %Ai
  %newBr8 = fsub <4 x double> %newBr7, %ciAi
  %diBi = fmul <4 x double> %di_vec, %Bi
  %newBr9 = fsub <4 x double> %newBr8, %diBi
  %newBi = fmul <4 x double> %cr_vec, %Ai
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci_vec, <4 x double> %Ar, <4 x double> %newBi)
  %newBi11 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di_vec, <4 x double> %Br, <4 x double> %newBi10)
  %newBi12 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr_vec, <4 x double> %Bi, <4 x double> %newBi11)
  store <4 x double> %newAr3, ptr %ptrAr, align 32
  store <4 x double> %newAi6, ptr %ptrAi, align 32
  store <4 x double> %newBr9, ptr %ptrBr, align 32
  store <4 x double> %newBi12, ptr %ptrBi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

attributes #0 = { nofree nosync nounwind memory(argmem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
