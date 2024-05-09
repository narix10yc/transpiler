; ModuleID = '../performance/gen_file.ll'
source_filename = "myModule"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite)
define void @f32_s3_sep_u3_k0_33330333(ptr nocapture %preal, ptr nocapture %pimag, i64 %idx_start, i64 %idx_end, ptr nocapture readonly %pmat) local_unnamed_addr #0 {
entry:
  %mat = load <8 x float>, ptr %pmat, align 32
  %ar = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> zeroinitializer
  %br = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %bi = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  %cond1 = icmp slt i64 %idx_start, %idx_end
  br i1 %cond1, label %loopBody, label %ret

loopBody:                                         ; preds = %entry, %loopBody
  %idx2 = phi i64 [ %idx_next, %loopBody ], [ %idx_start, %entry ]
  %pRe = getelementptr <16 x float>, ptr %preal, i64 %idx2
  %pIm = getelementptr <16 x float>, ptr %pimag, i64 %idx2
  %Re = load <16 x float>, ptr %pRe, align 64
  %Im = load <16 x float>, ptr %pIm, align 64
  %Ar = shufflevector <16 x float> %Re, <16 x float> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %Ai = shufflevector <16 x float> %Im, <16 x float> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %Br = shufflevector <16 x float> %Re, <16 x float> poison, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %Bi = shufflevector <16 x float> %Im, <16 x float> poison, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %newAr = fmul <8 x float> %ar, %Ar
  %newAr1 = tail call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %br, <8 x float> %Br, <8 x float> %newAr)
  %biBi = fmul <8 x float> %bi, %Bi
  %newAr2 = fsub <8 x float> %newAr1, %biBi
  %newAi = fmul <8 x float> %ar, %Ai
  %newAi3 = tail call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %br, <8 x float> %Bi, <8 x float> %newAi)
  %newAi4 = tail call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %bi, <8 x float> %Br, <8 x float> %newAi3)
  %newBr = fmul <8 x float> %cr, %Ar
  %newBr5 = tail call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %dr, <8 x float> %Br, <8 x float> %newBr)
  %ciAi = fmul <8 x float> %ci, %Ai
  %newBr6 = fsub <8 x float> %newBr5, %ciAi
  %diBi = fmul <8 x float> %di, %Bi
  %newBr7 = fsub <8 x float> %newBr6, %diBi
  %newBi = fmul <8 x float> %cr, %Ai
  %newBi8 = tail call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %ci, <8 x float> %Ar, <8 x float> %newBi)
  %newBi9 = tail call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %di, <8 x float> %Br, <8 x float> %newBi8)
  %newBi10 = tail call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %dr, <8 x float> %Bi, <8 x float> %newBi9)
  %newRe = shufflevector <8 x float> %newAr2, <8 x float> %newBr7, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %newIm = shufflevector <8 x float> %newAi4, <8 x float> %newBi10, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  store <16 x float> %newRe, ptr %pRe, align 64
  store <16 x float> %newIm, ptr %pIm, align 64
  %idx_next = add nsw i64 %idx2, 1
  %exitcond.not = icmp eq i64 %idx_next, %idx_end
  br i1 %exitcond.not, label %ret, label %loopBody

ret:                                              ; preds = %loopBody, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>) #1

attributes #0 = { nofree nosync nounwind memory(argmem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
