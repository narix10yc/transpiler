; ModuleID = 'myModule'
source_filename = "myModule"

define void @f64_s2_sep_u3_k1_33330333(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
define void @f64_s2_sep_u3_k0_33330333(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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

define void @f32_s3_sep_u3_k3_33330333(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <8 x float>, ptr %pmat, align 32
  %ar = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> zeroinitializer
  %br = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %ai = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %bi = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %idx_and_outer = and i64 %idx, -1
  %shl_outer = shl i64 %idx_and_outer, 4
  %idx_and_inner = and i64 %idx, 0
  %shl_inner = shl i64 %idx_and_inner, 3
  %idxA = add i64 %shl_outer, %shl_inner
  %idxB = add i64 %idxA, 8
  %pAr = getelementptr float, ptr %preal, i64 %idxA
  %pAi = getelementptr float, ptr %pimag, i64 %idxA
  %pBr = getelementptr float, ptr %preal, i64 %idxB
  %pBi = getelementptr float, ptr %pimag, i64 %idxB
  %Ar = load <8 x float>, ptr %pAr, align 32
  %Ai = load <8 x float>, ptr %pAi, align 32
  %Br = load <8 x float>, ptr %pBr, align 32
  %Bi = load <8 x float>, ptr %pBi, align 32
  %newAr = fmul <8 x float> %ar, %Ar
  %newAr1 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %br, <8 x float> %Br, <8 x float> %newAr)
  %biBi = fmul <8 x float> %bi, %Bi
  %newAr2 = fsub <8 x float> %newAr1, %biBi
  %newAi = fmul <8 x float> %ar, %Ai
  %newAi3 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %br, <8 x float> %Bi, <8 x float> %newAi)
  %newAi4 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %bi, <8 x float> %Br, <8 x float> %newAi3)
  %newBr = fmul <8 x float> %cr, %Ar
  %newBr5 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %dr, <8 x float> %Br, <8 x float> %newBr)
  %ciAi = fmul <8 x float> %ci, %Ai
  %newBr6 = fsub <8 x float> %newBr5, %ciAi
  %diBi = fmul <8 x float> %di, %Bi
  %newBr7 = fsub <8 x float> %newBr6, %diBi
  %newBi = fmul <8 x float> %cr, %Ai
  %newBi8 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %ci, <8 x float> %Ar, <8 x float> %newBi)
  %newBi9 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %di, <8 x float> %Br, <8 x float> %newBi8)
  %newBi10 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %dr, <8 x float> %Bi, <8 x float> %newBi9)
  store <8 x float> %newAr2, ptr %pAr, align 32
  store <8 x float> %newAi4, ptr %pAi, align 32
  store <8 x float> %newBr7, ptr %pBr, align 32
  store <8 x float> %newBi10, ptr %pBi, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}
define void @f32_s3_sep_u3_k2_33330333(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <8 x float>, ptr %pmat, align 32
  %ar = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> zeroinitializer
  %br = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %ai = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %bi = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pRe = getelementptr <16 x float>, ptr %preal, i64 %idx
  %pIm = getelementptr <16 x float>, ptr %pimag, i64 %idx
  %Re = load <16 x float>, ptr %pRe, align 64
  %Im = load <16 x float>, ptr %pIm, align 64
  %Ar = shufflevector <16 x float> %Re, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %Ai = shufflevector <16 x float> %Im, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11>
  %Br = shufflevector <16 x float> %Re, <16 x float> poison, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %Bi = shufflevector <16 x float> %Im, <16 x float> poison, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %newAr = fmul <8 x float> %ar, %Ar
  %newAr1 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %br, <8 x float> %Br, <8 x float> %newAr)
  %biBi = fmul <8 x float> %bi, %Bi
  %newAr2 = fsub <8 x float> %newAr1, %biBi
  %newAi = fmul <8 x float> %ar, %Ai
  %newAi3 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %br, <8 x float> %Bi, <8 x float> %newAi)
  %newAi4 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %bi, <8 x float> %Br, <8 x float> %newAi3)
  %newBr = fmul <8 x float> %cr, %Ar
  %newBr5 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %dr, <8 x float> %Br, <8 x float> %newBr)
  %ciAi = fmul <8 x float> %ci, %Ai
  %newBr6 = fsub <8 x float> %newBr5, %ciAi
  %diBi = fmul <8 x float> %di, %Bi
  %newBr7 = fsub <8 x float> %newBr6, %diBi
  %newBi = fmul <8 x float> %cr, %Ai
  %newBi8 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %ci, <8 x float> %Ar, <8 x float> %newBi)
  %newBi9 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %di, <8 x float> %Br, <8 x float> %newBi8)
  %newBi10 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %dr, <8 x float> %Bi, <8 x float> %newBi9)
  %newRe = shufflevector <8 x float> %newAr2, <8 x float> %newBr7, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  %newIm = shufflevector <8 x float> %newAi4, <8 x float> %newBi10, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 10, i32 11, i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15>
  store <16 x float> %newRe, ptr %pRe, align 64
  store <16 x float> %newIm, ptr %pIm, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}


define void @f32_s3_sep_u3_k1_33330333(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <8 x float>, ptr %pmat, align 32
  %ar = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> zeroinitializer
  %br = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %ai = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %bi = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pRe = getelementptr <16 x float>, ptr %preal, i64 %idx
  %pIm = getelementptr <16 x float>, ptr %pimag, i64 %idx
  %Re = load <16 x float>, ptr %pRe, align 64
  %Im = load <16 x float>, ptr %pIm, align 64
  %Ar = shufflevector <16 x float> %Re, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13>
  %Ai = shufflevector <16 x float> %Im, <16 x float> poison, <8 x i32> <i32 0, i32 1, i32 4, i32 5, i32 8, i32 9, i32 12, i32 13>
  %Br = shufflevector <16 x float> %Re, <16 x float> poison, <8 x i32> <i32 2, i32 3, i32 6, i32 7, i32 10, i32 11, i32 14, i32 15>
  %Bi = shufflevector <16 x float> %Im, <16 x float> poison, <8 x i32> <i32 2, i32 3, i32 6, i32 7, i32 10, i32 11, i32 14, i32 15>
  %newAr = fmul <8 x float> %ar, %Ar
  %newAr1 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %br, <8 x float> %Br, <8 x float> %newAr)
  %biBi = fmul <8 x float> %bi, %Bi
  %newAr2 = fsub <8 x float> %newAr1, %biBi
  %newAi = fmul <8 x float> %ar, %Ai
  %newAi3 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %br, <8 x float> %Bi, <8 x float> %newAi)
  %newAi4 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %bi, <8 x float> %Br, <8 x float> %newAi3)
  %newBr = fmul <8 x float> %cr, %Ar
  %newBr5 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %dr, <8 x float> %Br, <8 x float> %newBr)
  %ciAi = fmul <8 x float> %ci, %Ai
  %newBr6 = fsub <8 x float> %newBr5, %ciAi
  %diBi = fmul <8 x float> %di, %Bi
  %newBr7 = fsub <8 x float> %newBr6, %diBi
  %newBi = fmul <8 x float> %cr, %Ai
  %newBi8 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %ci, <8 x float> %Ar, <8 x float> %newBi)
  %newBi9 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %di, <8 x float> %Br, <8 x float> %newBi8)
  %newBi10 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %dr, <8 x float> %Bi, <8 x float> %newBi9)
  %newRe = shufflevector <8 x float> %newAr2, <8 x float> %newBr7, <16 x i32> <i32 0, i32 1, i32 8, i32 9, i32 2, i32 3, i32 10, i32 11, i32 4, i32 5, i32 12, i32 13, i32 6, i32 7, i32 14, i32 15>
  %newIm = shufflevector <8 x float> %newAi4, <8 x float> %newBi10, <16 x i32> <i32 0, i32 1, i32 8, i32 9, i32 2, i32 3, i32 10, i32 11, i32 4, i32 5, i32 12, i32 13, i32 6, i32 7, i32 14, i32 15>
  store <16 x float> %newRe, ptr %pRe, align 64
  store <16 x float> %newIm, ptr %pIm, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

define void @f32_s3_sep_u3_k0_33330333(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
entry:
  %mat = load <8 x float>, ptr %pmat, align 32
  %ar = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> zeroinitializer
  %br = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %cr = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2>
  %dr = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3, i32 3>
  %ai = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4, i32 4>
  %bi = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  %ci = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6, i32 6>
  %di = shufflevector <8 x float> %mat, <8 x float> poison, <8 x i32> <i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7, i32 7>
  br label %loop

loop:                                             ; preds = %loopBody, %entry
  %idx = phi i64 [ %idx_start, %entry ], [ %idx_next, %loopBody ]
  %cond = icmp slt i64 %idx, %idx_end
  br i1 %cond, label %loopBody, label %ret

loopBody:                                         ; preds = %loop
  %pRe = getelementptr <16 x float>, ptr %preal, i64 %idx
  %pIm = getelementptr <16 x float>, ptr %pimag, i64 %idx
  %Re = load <16 x float>, ptr %pRe, align 64
  %Im = load <16 x float>, ptr %pIm, align 64
  %Ar = shufflevector <16 x float> %Re, <16 x float> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %Ai = shufflevector <16 x float> %Im, <16 x float> poison, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>
  %Br = shufflevector <16 x float> %Re, <16 x float> poison, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %Bi = shufflevector <16 x float> %Im, <16 x float> poison, <8 x i32> <i32 1, i32 3, i32 5, i32 7, i32 9, i32 11, i32 13, i32 15>
  %newAr = fmul <8 x float> %ar, %Ar
  %newAr1 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %br, <8 x float> %Br, <8 x float> %newAr)
  %biBi = fmul <8 x float> %bi, %Bi
  %newAr2 = fsub <8 x float> %newAr1, %biBi
  %newAi = fmul <8 x float> %ar, %Ai
  %newAi3 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %br, <8 x float> %Bi, <8 x float> %newAi)
  %newAi4 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %bi, <8 x float> %Br, <8 x float> %newAi3)
  %newBr = fmul <8 x float> %cr, %Ar
  %newBr5 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %dr, <8 x float> %Br, <8 x float> %newBr)
  %ciAi = fmul <8 x float> %ci, %Ai
  %newBr6 = fsub <8 x float> %newBr5, %ciAi
  %diBi = fmul <8 x float> %di, %Bi
  %newBr7 = fsub <8 x float> %newBr6, %diBi
  %newBi = fmul <8 x float> %cr, %Ai
  %newBi8 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %ci, <8 x float> %Ar, <8 x float> %newBi)
  %newBi9 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %di, <8 x float> %Br, <8 x float> %newBi8)
  %newBi10 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %dr, <8 x float> %Bi, <8 x float> %newBi9)
  %newRe = shufflevector <8 x float> %newAr2, <8 x float> %newBr7, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  %newIm = shufflevector <8 x float> %newAi4, <8 x float> %newBi10, <16 x i32> <i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11, i32 4, i32 12, i32 5, i32 13, i32 6, i32 14, i32 7, i32 15>
  store <16 x float> %newRe, ptr %pRe, align 64
  store <16 x float> %newIm, ptr %pIm, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>) #0
; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
