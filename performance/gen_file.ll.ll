; ModuleID = '../performance/gen_file.ll'
source_filename = "myModule"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define void @u3_0_02003fff(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) local_unnamed_addr #0 {
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
  %arAr = fmul <4 x double> %ar_vec, %Ar
  %brBr = fmul <4 x double> %br_vec, %Br
  %newAr = fadd <4 x double> %arAr, %brBr
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr1 = fsub <4 x double> %newAr, %aiAi
  %biBi = fmul <4 x double> %bi_vec, %Bi
  %newAr2 = fsub <4 x double> %newAr1, %biBi
  %arAi = fmul <4 x double> %ar_vec, %Ai
  %aiAr = fmul <4 x double> %ai_vec, %Ar
  %newAi = fadd <4 x double> %aiAr, %arAi
  %brBi = fmul <4 x double> %br_vec, %Bi
  %newAi3 = fadd <4 x double> %newAi, %brBi
  %biBr = fmul <4 x double> %bi_vec, %Br
  %newAi4 = fadd <4 x double> %biBr, %newAi3
  %crAr = fmul <4 x double> %cr_vec, %Ar
  %drBr = fmul <4 x double> %dr_vec, %Br
  %newBr = fadd <4 x double> %crAr, %drBr
  %ciAi = fmul <4 x double> %ci_vec, %Ai
  %newBr5 = fsub <4 x double> %newBr, %ciAi
  %diBi = fmul <4 x double> %di_vec, %Bi
  %newBr6 = fsub <4 x double> %newBr5, %diBi
  %crAi = fmul <4 x double> %cr_vec, %Ai
  %ciAr = fmul <4 x double> %ci_vec, %Ar
  %newBi = fadd <4 x double> %ciAr, %crAi
  %diBr = fmul <4 x double> %di_vec, %Br
  %newBi7 = fadd <4 x double> %newBi, %diBr
  %drBi = fmul <4 x double> %dr_vec, %Bi
  %newBi8 = fadd <4 x double> %newBi7, %drBi
  store <4 x double> %newAr2, ptr %ptrAr, align 32
  store <4 x double> %newAi4, ptr %ptrAi, align 32
  store <4 x double> %newBr6, ptr %ptrBr, align 32
  store <4 x double> %newBi8, ptr %ptrBi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define void @u3_1_02001080(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) local_unnamed_addr #0 {
entry:
  %ai_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
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
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr = fsub <4 x double> %Ar, %aiAi
  %aiAr = fmul <4 x double> %ai_vec, %Ar
  %newAi = fadd <4 x double> %Ai, %aiAr
  %newBr = fneg <4 x double> %Br
  %newBi = fneg <4 x double> %Bi
  store <4 x double> %newAr, ptr %ptrAr, align 32
  store <4 x double> %newAi, ptr %ptrAi, align 32
  store <4 x double> %newBr, ptr %ptrBr, align 32
  store <4 x double> %newBi, ptr %ptrBi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define void @u3_2_02003fc0(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) local_unnamed_addr #0 {
entry:
  %ar_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> zeroinitializer
  %br_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %cr_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 2, i32 2, i32 2, i32 2>
  %dr_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  %ai_vec = shufflevector <8 x double> %mat, <8 x double> undef, <4 x i32> <i32 4, i32 4, i32 4, i32 4>
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
  %arAr = fmul <4 x double> %ar_vec, %Ar
  %brBr = fmul <4 x double> %br_vec, %Br
  %newAr = fadd <4 x double> %arAr, %brBr
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr1 = fsub <4 x double> %newAr, %aiAi
  %arAi = fmul <4 x double> %ar_vec, %Ai
  %aiAr = fmul <4 x double> %ai_vec, %Ar
  %newAi = fadd <4 x double> %aiAr, %arAi
  %brBi = fmul <4 x double> %br_vec, %Bi
  %newAi2 = fadd <4 x double> %newAi, %brBi
  %crAr = fmul <4 x double> %cr_vec, %Ar
  %drBr = fmul <4 x double> %dr_vec, %Br
  %newBr = fadd <4 x double> %crAr, %drBr
  %crAi = fmul <4 x double> %cr_vec, %Ai
  %drBi = fmul <4 x double> %dr_vec, %Bi
  %newBi = fadd <4 x double> %crAi, %drBi
  store <4 x double> %newAr1, ptr %ptrAr, align 32
  store <4 x double> %newAi2, ptr %ptrAi, align 32
  store <4 x double> %newBr, ptr %ptrBr, align 32
  store <4 x double> %newBi, ptr %ptrBi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) }
