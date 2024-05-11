; ModuleID = '../tests/ir.ll'
source_filename = "myModule"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"


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

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @u3_f64_alt_0200ffff(ptr nocapture %psv, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
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
  %ptrLo = getelementptr double, ptr %psv, i64 %alpha
  %ptrHi = getelementptr double, ptr %psv, i64 %beta
  %Lo = load <4 x double>, ptr %ptrLo, align 32
  %Hi = load <4 x double>, ptr %ptrHi, align 32
  %LoRe = fmul <4 x double> %ar, %Lo
  %LoRe1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Hi, <4 x double> %LoRe)
  %LoIm_s = fmul <4 x double> %ai_n, %Lo
  %LoIm_s2 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi_n, <4 x double> %Hi, <4 x double> %LoIm_s)
  %LoIm = shufflevector <4 x double> %LoIm_s2, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %newLo = fadd <4 x double> %LoRe1, %LoIm
  %HiRe = fmul <4 x double> %cr, %Hi
  %HiRe3 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Lo, <4 x double> %HiRe)
  %HiIm_s = fmul <4 x double> %ci_n, %Hi
  %HiIm_s4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di_n, <4 x double> %Lo, <4 x double> %HiIm_s)
  %HiIm = shufflevector <4 x double> %HiIm_s4, <4 x double> poison, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %newHi = fadd <4 x double> %HiRe3, %HiIm
  store <4 x double> %newLo, ptr %ptrLo, align 32
  store <4 x double> %newHi, ptr %ptrHi, align 32
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @u3_f64_sep_0200ffff(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
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
  %newAr = fmul <4 x double> %ar, %Ar
  %newAr1 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Br, <4 x double> %newAr)
  %aiAi = fmul <4 x double> %ai, %Ai
  %newAr2 = fsub <4 x double> %newAr1, %aiAi
  %biBi = fmul <4 x double> %bi, %Bi
  %newAr3 = fsub <4 x double> %newAr2, %biBi
  %newAi = fmul <4 x double> %ar, %Ai
  %newAi4 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ai, <4 x double> %Ar, <4 x double> %newAi)
  %newAi5 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %br, <4 x double> %Bi, <4 x double> %newAi4)
  %newAi6 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %bi, <4 x double> %Br, <4 x double> %newAi5)
  %newBr = fmul <4 x double> %cr, %Ar
  %newBr7 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Br, <4 x double> %newBr)
  %ciAi = fmul <4 x double> %ci, %Ai
  %newBr8 = fsub <4 x double> %newBr7, %ciAi
  %diBi = fmul <4 x double> %di, %Bi
  %newBr9 = fsub <4 x double> %newBr8, %diBi
  %newBi = fmul <4 x double> %cr, %Ai
  %newBi10 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %ci, <4 x double> %Ar, <4 x double> %newBi)
  %newBi11 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %di, <4 x double> %Br, <4 x double> %newBi10)
  %newBi12 = tail call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %dr, <4 x double> %Bi, <4 x double> %newBi11)
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
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #1

attributes #0 = { nofree nosync nounwind memory(argmem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }