; ModuleID = 'myModule'
source_filename = "myModule"

define void @f64_s1_sep_u2q_k1l0_0000000004104001(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, ptr %pmat) {
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
  %vecRe0 = shufflevector <2 x double> %Re0, <2 x double> %Re3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %vecRe1 = shufflevector <2 x double> %Re2, <2 x double> %Re1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %newReal = shufflevector <4 x double> %vecRe0, <4 x double> %vecRe1, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 1, i32 3, i32 5, i32 7>
  store <8 x double> %newReal, ptr %pReal, align 64
  %vecIm0 = shufflevector <2 x double> %Im0, <2 x double> %Im3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %vecIm1 = shufflevector <2 x double> %Im2, <2 x double> %Im1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %newImag = shufflevector <4 x double> %vecIm0, <4 x double> %vecIm1, <8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 1, i32 3, i32 5, i32 7>
  store <8 x double> %newImag, ptr %pImag, align 64
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}
