; ModuleID = 'myModule'
source_filename = "myModule"

define void @u3_0_a0020c3(ptr %preal, ptr %pimag, i64 %idx_start, i64 %idx_end, <8 x double> %mat) {
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
  %idx_and_outer = and i64 %idx, -256
  %shl_outer = shl i64 %idx_and_outer, 3
  %idx_and_inner = and i64 %idx, 255
  %shl_inner = shl i64 %idx_and_inner, 2
  %alpha = add i64 %shl_outer, %shl_inner
  %beta = add i64 %alpha, 1024
  %ptrAr = getelementptr double, ptr %preal, i64 %alpha
  %ptrAi = getelementptr double, ptr %pimag, i64 %alpha
  %ptrBr = getelementptr double, ptr %preal, i64 %beta
  %ptrBi = getelementptr double, ptr %pimag, i64 %beta
  %Ar = load <4 x double>, ptr %ptrAr, align 32
  %Ai = load <4 x double>, ptr %ptrAi, align 32
  %Br = load <4 x double>, ptr %ptrBr, align 32
  %Bi = load <4 x double>, ptr %ptrBi, align 32
  %newAr = fneg <4 x double> %Ar
  %aiAi = fmul <4 x double> %ai_vec, %Ai
  %newAr1 = fsub <4 x double> %newAr, %aiAi
  %newAi = fneg <4 x double> %Ai
  %aiAr = fmul <4 x double> %ai_vec, %Ar
  %newAi2 = fadd <4 x double> %newAi, %aiAr
  %drBr = fmul <4 x double> %dr_vec, %Br
  %diBi = fmul <4 x double> %di_vec, %Bi
  %newBr = fsub <4 x double> %drBr, %diBi
  %diBr = fmul <4 x double> %di_vec, %Br
  %drBi = fmul <4 x double> %dr_vec, %Bi
  %newBi = fadd <4 x double> %diBr, %drBi
  store <4 x double> %newAr1, ptr %ptrAr, align 32
  store <4 x double> %newAi2, ptr %ptrAi, align 32
  store <4 x double> %newBr, ptr %ptrBr, align 32
  store <4 x double> %newBi, ptr %ptrBi, align 32
  %idx_next = add i64 %idx, 1
  br label %loop

ret:                                              ; preds = %loop
  ret void
}
