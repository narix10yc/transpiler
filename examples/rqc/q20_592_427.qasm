OPENQASM 2.0;
qreg q[20];
u3(0,0,-pi/4) q[0];
u3(0,0,1.2855123337519458) q[1];
u3(0,0,-0.4125049863105691) q[3];
cx q[1],q[3];
u3(-0.11338121498674729,0,-1.2855123337519458) q[3];
cx q[1],q[3];
u3(pi/2,pi/4,-pi/2) q[1];
u3(1.4583353213678008,-1.585242875855311,-3.013558452255797) q[3];
u3(pi/2,1.7125798473339575,0) q[4];
u3(0,1.4065829705916304,-1.4065829705916302) q[5];
u3(pi,pi/2,pi/2) q[6];
u3(0,0,pi/4) q[7];
cx q[9],q[5];
u3(0,0,-pi/4) q[5];
u3(0.029150788531760393,0,0) q[10];
u3(pi,pi/4,-3*pi/4) q[11];
u3(0,1.4065829705916304,-1.4065829705916302) q[12];
cx q[13],q[10];
u3(-0.029150788531760393,0,0) q[10];
cx q[13],q[10];
u3(0,1.4065829705916304,-1.4065829705916302) q[10];
u3(pi/2,-pi/2,-1.3745875108326402) q[13];
u3(pi/2,0,0) q[14];
u3(0,1.4065829705916304,-1.4065829705916302) q[15];
cx q[7],q[15];
u3(0,0,-pi/4) q[15];
cx q[7],q[15];
u3(0.5761819871170232,pi/2,-2.694983321971657) q[7];
cx q[11],q[7];
u3(0,0,-pi/4) q[7];
cx q[11],q[7];
u3(0,1.4065829705916295,-0.6211848071941821) q[7];
u3(0,-0.1642133562032666,-0.6211848071941821) q[15];
u3(0,0,-2.1258644408790843) q[16];
cx q[14],q[16];
u3(1.2694095035181723,0,0) q[14];
u3(-1.2694095035181723,0,0) q[16];
cx q[14],q[16];
u3(pi/2,-pi,-pi) q[14];
cx q[14],q[1];
u3(pi/2,-pi/2,pi/4) q[1];
u3(2.157590186922424,4.112727032961092,3.7862274291496516) q[14];
cx q[14],q[7];
u3(0,0,2.756021715145907) q[7];
cx q[14],q[7];
u3(pi/2,0,pi) q[7];
u3(0,0,2.236977458900661) q[14];
u3(0,0,2.9112626042765326) q[16];
cx q[16],q[12];
u3(0,0,-pi/4) q[12];
cx q[16],q[12];
u3(pi/2,0,-3*pi/4) q[12];
u3(0,0,2.5513038752535597) q[16];
cx q[17],q[5];
u3(0,0,pi/4) q[5];
cx q[9],q[5];
u3(0,0,-pi/4) q[5];
u3(0,0,pi/4) q[9];
cx q[17],q[5];
u3(0,1.4065829705916295,-0.6211848071941821) q[5];
cx q[5],q[10];
u3(0,0,-pi/4) q[10];
cx q[17],q[9];
u3(0,0,-pi/4) q[9];
u3(0,0,pi/4) q[17];
cx q[17],q[9];
cx q[9],q[10];
u3(0,0,pi/4) q[10];
cx q[5],q[10];
u3(0,0,pi/4) q[5];
u3(0,0,-pi/4) q[10];
cx q[9],q[10];
cx q[9],q[5];
u3(0,0,-pi/4) q[5];
u3(0,0,pi/4) q[9];
cx q[9],q[5];
u3(pi/2,-1.6855064389836476,-pi/2) q[9];
u3(0,1.4065829705916295,-0.6211848071941821) q[10];
cx q[16],q[5];
u3(0,0,-2.5513038752535597) q[5];
cx q[16],q[5];
u3(0,0,2.5513038752535597) q[5];
u3(0,0,-pi/2) q[17];
cx q[6],q[17];
cx q[6],q[15];
cx q[15],q[6];
cx q[6],q[16];
u3(0,0,2.153521638374932) q[15];
cx q[16],q[6];
cx q[6],q[16];
u3(0,0,pi/2) q[6];
u3(0,0,-pi/2) q[16];
u3(pi/2,0,0) q[17];
u3(pi/2,pi/4,-pi) q[18];
cx q[8],q[18];
u3(pi/2,0,3*pi/4) q[18];
cx q[2],q[18];
u3(0,0,pi/4) q[18];
cx q[19],q[18];
u3(0,0,-pi/4) q[18];
cx q[2],q[18];
u3(0,0,2.220319221952373) q[2];
cx q[4],q[2];
u3(-2.7832895869824825,0,0) q[2];
u3(2.7832895869824825,0,0) q[4];
cx q[4],q[2];
u3(pi/2,0,0.9212734316374203) q[2];
cx q[2],q[12];
u3(pi/2,-pi,pi/2) q[4];
cx q[4],q[13];
u3(0,0,0.6820697579031797) q[12];
cx q[2],q[12];
u3(pi/2,-pi/2,-pi) q[2];
u3(pi/2,0,-pi) q[12];
u3(0,0,3.33240711173249) q[13];
cx q[4],q[13];
u3(-pi/2,-pi/2,pi/2) q[4];
cx q[4],q[2];
u3(pi,pi/2,-pi) q[2];
u3(0,0,pi/2) q[4];
cx q[7],q[4];
u3(-2.2096319169386405,0,0) q[4];
cx q[7],q[4];
u3(2.2096319169386405,-pi/2,0) q[4];
u3(0,0,pi/2) q[7];
u3(-pi/2,-pi/2,pi/2) q[13];
u3(0,0,pi/4) q[18];
cx q[19],q[18];
u3(pi/2,pi/4,3*pi/4) q[18];
cx q[8],q[18];
u3(pi,pi/2,pi/2) q[8];
cx q[8],q[0];
u3(0,0,2.2983394420779013) q[0];
cx q[8],q[0];
cx q[0],q[13];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,-pi/2) q[8];
cx q[13],q[0];
u3(0,0,-pi/4) q[0];
u3(pi/2,-pi/2,3*pi/4) q[18];
cx q[10],q[18];
u3(-0.18909942768413354,0,0) q[18];
cx q[10],q[18];
u3(pi,-0.142979852960873,pi/2) q[10];
u3(0.18909942768413354,-pi/2,0) q[18];
cx q[18],q[0];
u3(0,0,pi/4) q[0];
cx q[13],q[0];
u3(0,0,-pi/4) q[0];
u3(0,0,pi/4) q[13];
cx q[18],q[0];
u3(pi/2,0,-3*pi/4) q[0];
cx q[18],q[13];
u3(0,0,-pi/4) q[13];
u3(0,0,pi/4) q[18];
cx q[18],q[13];
cx q[0],q[13];
u3(0,0,-0.24879721119746367) q[0];
u3(0,0,-1.0746325114760582) q[13];
cx q[0],q[13];
u3(-1.7130493110247542,0,-2.170784129929007) q[13];
cx q[0],q[13];
u3(2.8094146486363365,-pi/2,2.838732149589763) q[0];
u3(1.7130493110247542,3.245416641405065,0) q[13];
u3(pi/2,0,pi) q[18];
u3(pi/2,-pi/2,pi/2) q[19];
cx q[19],q[3];
u3(0,0,5.170597713704947) q[3];
cx q[19],q[3];
u3(pi/2,1.8194937489255452,-pi/2) q[3];
cx q[1],q[3];
u3(-0.24639139478678623,0,0) q[1];
u3(-0.24639139478678623,0,0) q[3];
cx q[1],q[3];
u3(pi/2,-pi/2,-pi) q[1];
u3(0,0,-0.24869742213064816) q[3];
cx q[3],q[12];
cx q[8],q[1];
cx q[1],q[8];
u3(0,1.4065829705916304,-1.4065829705916302) q[1];
cx q[1],q[18];
u3(pi/2,pi/4,-pi) q[8];
u3(0,0,-pi/4) q[12];
cx q[11],q[12];
u3(0,0,pi/4) q[12];
cx q[3],q[12];
u3(0,0,pi/4) q[3];
u3(0,0,-pi/4) q[12];
cx q[11],q[12];
cx q[11],q[3];
u3(0,0,-pi/4) q[3];
u3(0,0,pi/4) q[11];
cx q[11],q[3];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[11],q[8];
u3(0,0,-pi/4) q[8];
u3(0,2.191981133989078,-0.6211848071941821) q[12];
cx q[15],q[3];
u3(pi/2,pi/2,-pi/2) q[15];
cx q[7],q[15];
u3(0,1.4065829705916304,-1.4065829705916302) q[7];
u3(0,0,5.714087545357307) q[18];
cx q[1],q[18];
u3(pi/2,0,2.850806209712708) q[1];
u3(pi/2,0,pi/2) q[18];
u3(pi/2,-pi,-pi/2) q[19];
cx q[17],q[19];
cx q[19],q[17];
cx q[17],q[9];
u3(0,0,-3.0268825414010423) q[9];
cx q[17],q[9];
cx q[9],q[8];
u3(0,0,pi/4) q[8];
cx q[11],q[8];
u3(pi/2,0,-3*pi/4) q[8];
u3(pi/2,0,pi) q[11];
cx q[13],q[9];
u3(pi/2,0,pi) q[13];
cx q[9],q[13];
u3(0,0,-pi/4) q[13];
cx q[16],q[13];
u3(0,0,pi/4) q[13];
cx q[9],q[13];
u3(0,0,pi/4) q[9];
u3(0,0,-pi/4) q[13];
cx q[16],q[13];
u3(pi/2,0,-3*pi/4) q[13];
cx q[16],q[9];
u3(0,0,-pi/4) q[9];
u3(0,0,pi/4) q[16];
cx q[16],q[9];
cx q[13],q[9];
u3(pi/2,3.977179638177784,0.44010251595193955) q[13];
u3(pi/2,0,pi) q[17];
cx q[12],q[17];
u3(0,0,-pi/4) q[17];
cx q[12],q[17];
u3(0,0,pi/2) q[12];
cx q[12],q[8];
cx q[9],q[8];
u3(pi/2,0,pi) q[9];
cx q[8],q[9];
u3(0,0,-pi/4) q[9];
u3(pi/2,pi/2,-pi/2) q[12];
cx q[1],q[12];
u3(0,0,0.024770899999468004) q[12];
cx q[1],q[12];
u3(pi/2,2.966129862229282,-pi) q[1];
u3(pi/2,-pi/2,-pi) q[12];
u3(pi/2,0,-3*pi/4) q[17];
cx q[17],q[0];
u3(pi/2,0,pi) q[0];
u3(2.4324046727349935,0,-pi) q[19];
cx q[5],q[19];
u3(-0.861608345940097,0,0) q[19];
cx q[5],q[19];
u3(pi/2,0,-pi/2) q[5];
cx q[5],q[6];
cx q[6],q[5];
u3(0,0,pi/4) q[5];
cx q[5],q[11];
u3(pi/2,0,pi) q[6];
cx q[6],q[2];
u3(pi/2,0,pi) q[2];
u3(0,0,pi/4) q[6];
cx q[6],q[18];
u3(0,0,-pi/4) q[11];
cx q[5],q[11];
u3(pi/2,0,pi) q[5];
u3(pi/2,pi/2,-3*pi/4) q[11];
cx q[14],q[19];
cx q[15],q[2];
u3(0,0,6.043287438608912) q[2];
cx q[15],q[2];
u3(0.7790830110014589,2.5535354582185654,2.5506088113656293) q[2];
u3(pi/2,-pi,-pi) q[15];
cx q[10],q[15];
u3(0,0,pi/4) q[10];
u3(pi/2,0,pi) q[15];
cx q[10],q[15];
u3(0,0,-pi/4) q[15];
cx q[10],q[15];
u3(pi/2,-pi/2,pi/2) q[10];
u3(0,-1.7662842897866815,-0.6211848071941821) q[15];
u3(0,0,-pi/4) q[18];
cx q[6],q[18];
u3(0,-2.2864569352029425,-0.6211848071941821) q[18];
cx q[18],q[6];
u3(0,0,-2.590145401385014) q[6];
cx q[18],q[6];
u3(0,0,2.590145401385014) q[6];
u3(pi/2,0,3*pi/4) q[18];
u3(0,0,-2.236977458900661) q[19];
cx q[14],q[19];
u3(0,1.4065829705916304,-1.4065829705916302) q[14];
cx q[3],q[14];
u3(0,0,-pi/4) q[14];
cx q[4],q[14];
u3(0,0,pi/4) q[14];
cx q[3],q[14];
u3(0,0,pi/4) q[3];
u3(0,0,-pi/4) q[14];
cx q[4],q[14];
cx q[4],q[3];
u3(0,0,-pi/4) q[3];
u3(0,0,pi/4) q[4];
cx q[4],q[3];
cx q[3],q[5];
u3(0.9315849811526642,1.0935433189691324,0.2523517311476726) q[4];
u3(0,0,-pi/4) q[5];
u3(pi/2,-pi/2,3*pi/4) q[14];
cx q[11],q[14];
u3(pi,0,pi) q[11];
u3(1.0406210126321873,-0.8269760347705106,1.166015322217281) q[14];
cx q[16],q[5];
u3(0,0,pi/4) q[5];
cx q[3],q[5];
u3(0,0,pi/4) q[3];
u3(0,0,-pi/4) q[5];
cx q[16],q[5];
u3(pi/2,0,-3*pi/4) q[5];
cx q[5],q[12];
u3(0,0,-pi/4) q[12];
cx q[16],q[3];
u3(0,0,-pi/4) q[3];
u3(0,0,pi/4) q[16];
cx q[16],q[3];
u3(pi/2,-1.3829880446409617,-1.0041900149686311) q[3];
cx q[14],q[3];
cx q[3],q[14];
u3(0,1.4065829705916304,-1.4065829705916302) q[3];
cx q[14],q[18];
u3(0,0,pi/4) q[14];
u3(pi/2,0,pi) q[16];
cx q[11],q[16];
u3(0,0,pi/4) q[11];
u3(pi/2,0,pi) q[16];
cx q[11],q[16];
u3(0,0,-pi/4) q[16];
cx q[11],q[16];
u3(0,0,3.3684231870313366) q[11];
cx q[11],q[3];
u3(pi/2,0,pi) q[3];
u3(0,1.4065829705916295,-0.6211848071941821) q[16];
u3(pi/2,0,pi) q[18];
cx q[14],q[18];
u3(0,0,-pi/4) q[18];
cx q[14],q[18];
u3(0,0,pi) q[14];
u3(0.9451785664057596,0,pi/4) q[18];
u3(pi/2,-2.8456735605387333,-3.084937333849605) q[19];
cx q[19],q[9];
u3(0,0,pi/4) q[9];
cx q[8],q[9];
u3(0,0,pi/4) q[8];
u3(0,0,-pi/4) q[9];
cx q[19],q[9];
u3(pi/2,0,-3*pi/4) q[9];
cx q[19],q[8];
u3(0,0,-pi/4) q[8];
u3(0,0,pi/4) q[19];
cx q[19],q[8];
cx q[9],q[8];
cx q[8],q[12];
u3(0,0,pi/4) q[9];
cx q[9],q[7];
u3(0,0,-pi/4) q[7];
cx q[9],q[7];
u3(pi/2,0,pi/4) q[7];
cx q[7],q[1];
u3(-2.3226783546927483,0,0) q[1];
u3(-2.3226783546927483,0,0) q[7];
cx q[7],q[1];
u3(0,0,-1.3953335354343857) q[1];
cx q[1],q[18];
u3(pi/2,-3*pi/4,-pi) q[7];
u3(0,0,1.3536180267538853) q[9];
cx q[9],q[0];
u3(0,0,-1.3536180267538853) q[0];
cx q[9],q[0];
u3(0,0,1.3536180267538853) q[0];
u3(pi/2,-pi/2,pi/2) q[9];
u3(0,0,pi/4) q[12];
cx q[5],q[12];
u3(0,0,pi/4) q[5];
u3(0,0,-pi/4) q[12];
cx q[8],q[12];
cx q[8],q[5];
u3(0,0,-pi/4) q[5];
u3(0,0,pi/4) q[8];
cx q[8],q[5];
u3(0,0,pi/4) q[5];
u3(0,0,1.7199207581579519) q[8];
cx q[8],q[17];
u3(pi/2,2.4086626797782964,-3*pi/4) q[12];
cx q[12],q[6];
u3(0,0,-2.4086626797782964) q[6];
cx q[12],q[6];
u3(pi/2,pi/2,-0.7329299738114967) q[6];
u3(0,0,1.495007407084527) q[12];
cx q[0],q[12];
u3(0,0,-1.495007407084527) q[12];
cx q[0],q[12];
u3(-pi/2,-pi/2,pi/2) q[0];
u3(0,0,-1.7199207581579519) q[17];
cx q[8],q[17];
u3(pi/2,-pi/2,pi/2) q[8];
cx q[9],q[8];
u3(0,0,4.574991018060457) q[8];
cx q[9],q[8];
u3(0,-pi/2,pi/2) q[8];
u3(pi/2,-pi/2,-pi) q[9];
cx q[7],q[9];
u3(0,0,4.914195846490501) q[9];
cx q[7],q[9];
u3(pi/2,2.1913631698914724,-pi) q[9];
u3(0,1.2418439922837017,-2.6635158877155427) q[17];
u3(-0.9451785664057596,0,0) q[18];
cx q[1],q[18];
cx q[7],q[1];
u3(0,0,6.219067682036876) q[1];
cx q[7],q[1];
u3(pi/2,0,0) q[1];
u3(0.6794119270604229,-1.6113010146300024,-1.6113010146300017) q[7];
u3(1.0087482692781182,-2.3245875598290837,-3.025923734739406) q[18];
u3(pi/2,0,pi) q[19];
cx q[13],q[19];
cx q[13],q[16];
u3(0,0,1.9921350409298872) q[16];
cx q[13],q[16];
u3(0,0,-2.1643732114520877) q[13];
cx q[15],q[13];
u3(-0.953165217491557,0,-3.1103180468012748) q[13];
cx q[15],q[13];
u3(0.953165217491557,0.8357187639622485,0) q[13];
cx q[13],q[2];
u3(0,0,-0.40145053601658043) q[2];
cx q[13],q[2];
u3(0,0,0.40145053601658043) q[2];
u3(0,0,1.2450424134350557) q[13];
u3(0,0,1.7902425029069833) q[15];
cx q[15],q[11];
u3(0,0,-1.7902425029069833) q[11];
cx q[15],q[11];
u3(pi/2,-pi/2,-2.922146477477707) q[11];
u3(pi,0,pi) q[15];
u3(pi/2,-pi/4,-pi) q[16];
cx q[12],q[16];
u3(0,0,-pi/4) q[16];
cx q[3],q[16];
u3(pi/2,-pi/2,pi/2) q[3];
cx q[3],q[11];
u3(0,0,6.18907342889127) q[11];
cx q[3],q[11];
u3(-pi/2,-pi/2,pi/2) q[3];
u3(pi/2,pi/2,0) q[11];
cx q[11],q[13];
u3(1.9651611083256304,0,0) q[11];
u3(-1.9651611083256304,0,0) q[13];
cx q[11],q[13];
u3(1.0362750884975305,1.5552145619396107,pi/2) q[11];
u3(0,0,-1.2450424134350557) q[13];
u3(0,0,pi/4) q[16];
cx q[12],q[16];
cx q[12],q[0];
u3(0,0,pi/2) q[12];
u3(pi/2,-1.6669436982489452,3*pi/4) q[16];
cx q[9],q[16];
u3(-2.645729885056441,0,-4.148497761441564) q[16];
cx q[9],q[16];
u3(pi/2,0,pi) q[9];
cx q[0],q[9];
u3(0,0,-pi/4) q[9];
cx q[2],q[9];
u3(0,0,pi/4) q[9];
cx q[0],q[9];
u3(0,0,pi/4) q[0];
u3(0,0,-pi/4) q[9];
cx q[2],q[9];
cx q[2],q[0];
u3(0,0,-pi/4) q[0];
u3(0,0,pi/4) q[2];
cx q[2],q[0];
u3(pi/2,pi/2,-pi) q[0];
u3(0,0,-pi/2) q[2];
u3(pi/2,1.0393255190658497,-3*pi/4) q[9];
u3(2.645729885056441,1.2553635239117753,0) q[16];
cx q[16],q[15];
u3(0,0,-1.7231073714008522) q[15];
cx q[16],q[15];
u3(0,0,2.7621476897665787) q[15];
u3(pi/2,0,0) q[16];
u3(0,1.4065829705916304,-1.4065829705916302) q[19];
cx q[4],q[19];
u3(0,0,1.7768032393979634) q[19];
cx q[4],q[19];
u3(0,1.4065829705916304,-1.4065829705916302) q[4];
cx q[5],q[4];
u3(pi/2,0,pi) q[4];
u3(0,0,pi/4) q[5];
cx q[5],q[4];
u3(0,0,-pi/4) q[4];
cx q[5],q[4];
u3(2.040908279656696,-pi/4,-0.9302339733610063) q[4];
u3(0,0,-pi) q[5];
cx q[5],q[17];
u3(pi,-0.6949041229592967,-pi) q[5];
cx q[6],q[4];
u3(pi/2,0,3*pi/4) q[4];
cx q[8],q[4];
u3(0,0,pi/4) q[4];
u3(pi/2,0,-pi/2) q[19];
cx q[10],q[19];
u3(0,0,1.1306113859072064) q[19];
cx q[10],q[19];
u3(-pi/2,-pi/2,pi/2) q[10];
u3(pi/2,-pi,-pi/2) q[19];
cx q[10],q[19];
u3(-1.5711829194062958,0,0) q[19];
cx q[10],q[19];
u3(pi/2,0,pi) q[10];
cx q[14],q[10];
u3(0,0,4.174362439052028) q[10];
cx q[14],q[10];
u3(2.2307670492670004,-pi,0) q[10];
u3(pi/2,0,-3*pi/4) q[14];
cx q[17],q[10];
u3(-2.4816219311176897,0,0) q[10];
cx q[17],q[10];
cx q[10],q[2];
u3(0,0,pi/2) q[2];
cx q[2],q[0];
u3(-1.815384321692164,0,0) q[0];
cx q[2],q[0];
u3(1.7026640803324642,2.934995028865468,-1.01050604386249) q[0];
u3(pi/2,0,pi) q[2];
u3(2.055038555799429,2.264345048014418,1.8985712887277266) q[10];
u3(0,1.4065829705916304,-1.4065829705916302) q[17];
cx q[18],q[17];
u3(0,0,-pi/4) q[17];
cx q[18],q[17];
u3(0,1.4065829705916295,-0.6211848071941821) q[17];
cx q[9],q[17];
u3(0,0,-0.2539273556684011) q[17];
cx q[9],q[17];
u3(pi/2,-pi/2,1.824723682463297) q[17];
cx q[18],q[7];
u3(0,0,-pi/4) q[7];
cx q[11],q[7];
u3(0,0,pi/4) q[7];
cx q[18],q[7];
u3(0,0,-pi/4) q[7];
cx q[11],q[7];
u3(pi/2,1.2457714746769177,-3*pi/4) q[7];
u3(0,0,pi/4) q[18];
cx q[11],q[18];
u3(0,0,pi/4) q[11];
u3(0,0,-pi/4) q[18];
cx q[11],q[18];
cx q[11],q[2];
u3(0,0,1.1396916126131589) q[2];
cx q[11],q[2];
u3(pi/2,-pi/2,-pi) q[2];
u3(pi/2,0,pi) q[11];
u3(0,0,1.073255987446521) q[18];
u3(1.5711829194062958,-pi/2,0) q[19];
cx q[19],q[4];
u3(0,0,-pi/4) q[4];
cx q[8],q[4];
u3(0,0,pi/4) q[4];
u3(0,0,1.8031868574007799) q[8];
cx q[1],q[8];
u3(-0.0991291356102213,0,0) q[1];
u3(-0.0991291356102213,0,0) q[8];
cx q[1],q[8];
u3(pi/2,-pi,-pi) q[1];
cx q[1],q[14];
u3(0,0,-1.4726498107654589) q[8];
u3(0,0,-pi/4) q[14];
cx q[13],q[14];
u3(0,0,pi/4) q[14];
cx q[1],q[14];
u3(0,0,pi/4) q[1];
u3(0,0,-pi/4) q[14];
cx q[13],q[14];
cx q[13],q[1];
u3(0,0,-pi/4) q[1];
u3(0,0,pi/4) q[13];
cx q[13],q[1];
u3(pi,1.6800081056668619,-pi) q[13];
cx q[13],q[18];
u3(pi/2,0,-3*pi/4) q[14];
cx q[16],q[8];
u3(-1.7897334084369376,0,0) q[8];
u3(-1.7897334084369376,0,0) q[16];
cx q[16],q[8];
u3(pi/2,0,2.811055606954472) q[8];
cx q[14],q[8];
u3(0,0,-pi/4) q[8];
cx q[1],q[8];
u3(0,0,pi/4) q[8];
cx q[14],q[8];
u3(0,0,-pi/4) q[8];
cx q[1],q[8];
u3(pi/2,0,-3*pi/4) q[8];
u3(0,0,pi/4) q[14];
cx q[1],q[14];
u3(0,0,pi/4) q[1];
u3(0,0,-pi/4) q[14];
cx q[1],q[14];
u3(1.4331823736549545,0.2603082385111435,1.203645367440009) q[1];
u3(0,0,-0.5515930714665309) q[14];
cx q[7],q[14];
u3(-0.6953920624363693,0,-1.2457714746769173) q[14];
cx q[7],q[14];
u3(0.8228126411852205,-1.7728340126150268,0.5454010260545541) q[7];
u3(0.6953920624363693,1.7973645461434482,0) q[14];
cx q[14],q[2];
u3(pi/2,-pi/2,-pi) q[2];
u3(pi/2,-pi/2,-pi) q[16];
u3(-2.1180113025688394,0,-1.6800081056668619) q[18];
cx q[13],q[18];
u3(0,0,-3*pi/16) q[13];
u3(2.1180113025688394,0.606752118220341,0) q[18];
cx q[18],q[0];
u3(0,0,pi/2) q[0];
u3(pi/2,1.5520826773217973,pi/2) q[18];
cx q[19],q[4];
u3(pi/2,pi/4,3*pi/4) q[4];
cx q[6],q[4];
cx q[3],q[6];
u3(0,0,1.6436396247055174) q[3];
u3(pi,0.639985212896022,2.996179703088367) q[4];
cx q[4],q[5];
u3(2.145254923511506,0,0) q[4];
u3(-2.145254923511506,0,0) q[5];
cx q[4],q[5];
u3(pi/2,-pi,pi/2) q[4];
u3(0,0,0.6949041229592967) q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
u3(2.2551717305056775,1.0909609216795748,0.6551713761708293) q[6];
cx q[9],q[6];
u3(pi/2,-pi/2,pi/2) q[6];
cx q[15],q[3];
u3(-0.4803413887891855,0,-3.8281121869614343) q[3];
cx q[15],q[3];
u3(0.4803413887891855,2.184472562255917,0) q[3];
cx q[5],q[3];
cx q[3],q[5];
u3(0.13389694142663436,-0.4698658506940703,1.20653986610367) q[15];
u3(0,0,pi/2) q[19];
cx q[19],q[12];
u3(-2.430877990280737,0,0) q[12];
cx q[19],q[12];
u3(2.281510990103953,pi/2,-pi) q[12];
cx q[16],q[12];
u3(1.6875971286389222,0,0) q[12];
cx q[5],q[12];
u3(-1.6875971286389222,0,0) q[12];
cx q[5],q[12];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[12];
cx q[12],q[7];
u3(0,0,pi/4) q[7];
u3(4.54825544749003,2.0278345654774785,0.7662870586346435) q[12];
u3(pi,0,pi) q[16];
cx q[16],q[15];
u3(0,0,-pi/4) q[15];
cx q[8],q[15];
u3(0,0,pi/4) q[15];
cx q[16],q[15];
u3(0,0,-pi/4) q[15];
cx q[8],q[15];
u3(pi/2,pi/2,-pi/4) q[15];
u3(0,0,pi/4) q[16];
cx q[8],q[16];
u3(0,0,pi/4) q[8];
u3(0,0,-pi/4) q[16];
cx q[8],q[16];
cx q[8],q[5];
u3(0,0,-pi/4) q[5];
cx q[16],q[5];
u3(0,0,pi/4) q[5];
cx q[8],q[5];
u3(0,0,-pi/4) q[5];
u3(0,0,pi/4) q[8];
cx q[16],q[5];
u3(pi/2,-pi/2,3*pi/4) q[5];
cx q[16],q[8];
u3(0,0,-pi/4) q[8];
u3(0,0,pi/4) q[16];
cx q[16],q[8];
u3(0,0,pi/2) q[16];
u3(0,0,pi/4) q[19];
cx q[19],q[4];
u3(0,0,-pi/4) q[4];
cx q[19],q[4];
u3(0,2.977379297386526,-0.6211848071941817) q[4];
cx q[3],q[4];
u3(-2.8368562325709457,0,0) q[4];
cx q[3],q[4];
u3(0.22495803614500048,0,pi/2) q[3];
cx q[3],q[2];
u3(pi/2,-pi/2,pi/2) q[3];
u3(2.8368562325709457,pi/4,0) q[4];
cx q[5],q[4];
cx q[4],q[5];
u3(pi,0,-pi) q[4];
cx q[4],q[18];
u3(-0.8627900333037687,0,0) q[4];
cx q[7],q[3];
u3(0,0,-pi/4) q[3];
cx q[7],q[3];
u3(pi/2,pi/2,pi/4) q[3];
u3(pi/2,4.668303356990303,3.8764413905341115) q[7];
u3(-0.8627900333037687,0,0) q[18];
cx q[4],q[18];
u3(pi/2,-pi,pi/2) q[4];
u3(0,0,-3.122879004116694) q[18];
u3(pi/2,-pi/2,pi/2) q[19];
cx q[19],q[17];
u3(0,0,5.2048123432521125) q[17];
cx q[19],q[17];
u3(-pi/2,-pi/2,pi/2) q[17];
cx q[17],q[9];
cx q[8],q[9];
u3(0,0,5.102224286605419) q[9];
cx q[8],q[9];
u3(0,1.4065829705916304,-1.4065829705916302) q[8];
u3(0,0,-pi/2) q[9];
cx q[15],q[8];
u3(0,0,-pi/4) q[8];
u3(pi/2,0,pi) q[17];
cx q[13],q[17];
u3(0,0,-pi/16) q[17];
cx q[13],q[17];
cx q[13],q[6];
u3(0,0,-pi/16) q[6];
u3(0,1.4065829705916304,-1.210233429742268) q[17];
cx q[6],q[17];
u3(0,0,pi/16) q[17];
cx q[6],q[17];
cx q[13],q[6];
u3(0,0,pi/16) q[6];
u3(0,1.4065829705916304,-1.6029325114409922) q[17];
cx q[6],q[17];
u3(0,0,-pi/16) q[17];
cx q[6],q[17];
cx q[6],q[0];
u3(0,0,-pi/16) q[0];
u3(0,1.4065829705916304,-1.210233429742268) q[17];
cx q[0],q[17];
u3(0,0,pi/16) q[17];
cx q[0],q[17];
cx q[13],q[0];
u3(0,0,pi/16) q[0];
u3(0,1.4065829705916304,-1.6029325114409922) q[17];
cx q[0],q[17];
u3(0,0,-pi/16) q[17];
cx q[0],q[17];
cx q[6],q[0];
u3(0,0,-pi/16) q[0];
u3(pi/2,-2.6971082585225794,-pi/2) q[6];
cx q[6],q[18];
u3(0,1.4065829705916304,-1.210233429742268) q[17];
cx q[0],q[17];
u3(0,0,pi/16) q[17];
cx q[0],q[17];
cx q[13],q[0];
u3(0,0,pi/16) q[0];
u3(pi/2,pi/4,-pi) q[13];
u3(0,1.4065829705916304,-1.6029325114409922) q[17];
cx q[0],q[17];
u3(0,0,-pi/16) q[17];
cx q[0],q[17];
cx q[0],q[13];
u3(pi/2,0,3*pi/4) q[13];
cx q[10],q[13];
u3(0,0,pi/4) q[13];
cx q[2],q[13];
u3(0,0,-pi/4) q[13];
cx q[10],q[13];
u3(0,0,pi/4) q[13];
cx q[2],q[13];
u3(pi/2,-pi/2,pi/2) q[2];
u3(pi/2,pi/4,3*pi/4) q[13];
cx q[0],q[13];
u3(pi/2,0,pi) q[0];
u3(pi/2,1.6023677549035868,3*pi/4) q[13];
u3(2.252163522673159,2.878221989961391,-1.4682878542011029) q[17];
cx q[17],q[2];
u3(0,0,6.086791049012743) q[2];
cx q[17],q[2];
u3(-pi/2,-pi/2,pi/2) q[2];
u3(-pi/2,-pi/2,pi/2) q[17];
u3(0,0,-2.0152807218621103) q[18];
cx q[6],q[18];
u3(2.4397653231103194,-1.3267322003646058,-1.3987971776496273) q[6];
u3(pi/2,pi/2,0.4444843950672137) q[18];
u3(-pi/2,-pi/2,pi/2) q[19];
cx q[19],q[11];
u3(0,0,2.096274527086669) q[11];
cx q[19],q[11];
u3(pi/2,-pi/2,pi/2) q[11];
cx q[11],q[16];
cx q[16],q[11];
u3(pi/2,0,pi) q[11];
cx q[5],q[11];
u3(0,0,-pi/4) q[11];
u3(1.293187111867191,2.9702969782469903,-1.2559885716919403) q[16];
u3(0,0,1.9461731264637945) q[19];
cx q[14],q[19];
u3(0,0,-1.9461731264637945) q[19];
cx q[14],q[19];
cx q[14],q[11];
u3(0,0,pi/4) q[11];
cx q[5],q[11];
u3(0,0,pi/4) q[5];
u3(0,0,-pi/4) q[11];
cx q[14],q[11];
u3(pi/2,pi/4,-3*pi/4) q[11];
cx q[11],q[0];
u3(0,0,-pi/4) q[0];
cx q[11],q[0];
u3(pi/2,0,-3*pi/4) q[0];
u3(0,1.4065829705916304,-1.4065829705916302) q[11];
cx q[14],q[5];
u3(0,0,-pi/4) q[5];
u3(0,0,pi/4) q[14];
cx q[14],q[5];
u3(pi/2,0,-pi) q[5];
cx q[2],q[5];
u3(0,0,-pi/4) q[5];
cx q[12],q[5];
u3(0,0,pi/4) q[5];
cx q[2],q[5];
u3(0,0,pi/4) q[2];
u3(0,0,-pi/4) q[5];
cx q[12],q[5];
u3(pi/2,-0.35268216585151047,-3*pi/4) q[5];
cx q[12],q[2];
u3(0,0,-pi/4) q[2];
u3(0,0,pi/4) q[12];
cx q[12],q[2];
u3(0.4683953037644607,-2.6911775309924346,-0.26932563746868965) q[2];
u3(pi/2,-pi/2,pi/2) q[12];
u3(pi/2,0,-pi/2) q[14];
cx q[14],q[9];
cx q[9],q[14];
cx q[1],q[14];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[9];
cx q[14],q[1];
u3(0,0,-pi/4) q[1];
cx q[17],q[1];
u3(0,0,pi/4) q[1];
cx q[14],q[1];
u3(0,0,-pi/4) q[1];
u3(0,0,pi/4) q[14];
cx q[17],q[1];
u3(pi/2,0,-3*pi/4) q[1];
cx q[17],q[14];
u3(0,0,-pi/4) q[14];
u3(0,0,pi/4) q[17];
cx q[17],q[14];
cx q[1],q[14];
u3(pi/2,-2.225285857659863,-pi/2) q[1];
u3(0,0,0.31781127983330393) q[14];
u3(1.7274808487908462,-pi/2,pi/2) q[17];
cx q[19],q[8];
u3(0,0,pi/4) q[8];
cx q[15],q[8];
u3(0,0,-pi/4) q[8];
u3(0,0,pi/4) q[15];
cx q[19],q[8];
u3(pi/2,-pi/2,3*pi/4) q[8];
cx q[4],q[8];
u3(0,0,5.728908646444111) q[8];
cx q[4],q[8];
u3(pi/2,-pi/4,-pi) q[4];
u3(-pi/2,-pi/2,pi/2) q[8];
cx q[8],q[13];
u3(0,0,3.5570485844966555) q[13];
cx q[8],q[13];
u3(0,0,pi/4) q[8];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[13],q[12];
u3(0,0,6.160881104093545) q[12];
cx q[13],q[12];
u3(pi/2,-pi,-pi/2) q[12];
cx q[8],q[12];
u3(-1.9897407023002327,0,0) q[12];
cx q[8],q[12];
u3(1.989740702300233,0,0) q[12];
u3(1.8027786340076666,-1.5914298931792283,3.0520724413914433) q[13];
cx q[12],q[13];
u3(pi,0,pi) q[12];
cx q[16],q[4];
u3(pi/2,0,3*pi/4) q[4];
cx q[9],q[4];
u3(0,0,pi/4) q[4];
cx q[0],q[4];
u3(0,0,-pi/4) q[4];
cx q[9],q[4];
u3(0,0,pi/4) q[4];
cx q[0],q[4];
u3(0,0,5.762288768442694) q[0];
cx q[0],q[5];
u3(pi/2,pi/4,3*pi/4) q[4];
u3(-3.035213704010853,0,-5.762288768442694) q[5];
cx q[0],q[5];
u3(0,0,3.084134306467051) q[0];
cx q[0],q[17];
u3(1.5885738878013864,1.4659078188379961,1.7380774885431869) q[5];
u3(0,0,-0.9396730254091126) q[9];
cx q[9],q[5];
u3(0,0,-2.1714854939455623) q[5];
u3(0.8712558453543352,0.6858040901892126,-1.5645490915612426) q[9];
cx q[16],q[4];
u3(pi/2,0,3*pi/4) q[4];
u3(0,0,-pi/2) q[16];
cx q[4],q[16];
u3(0,0,-1.0248591856886917) q[4];
cx q[14],q[4];
u3(-0.7287923886826697,0,-5.030200260217994) q[4];
cx q[14],q[4];
u3(0.6601892825145322,-0.11683237415615855,-0.1446534272328246) q[4];
u3(0,0,3.08691080803748) q[16];
u3(0,0,-3.084134306467051) q[17];
cx q[0],q[17];
u3(pi/2,0,0) q[0];
cx q[0],q[5];
u3(3.04874998127846,0,0) q[0];
u3(-3.04874998127846,0,0) q[5];
cx q[0],q[5];
u3(pi/2,-pi,-pi) q[0];
u3(0,0,2.1714854939455623) q[5];
u3(pi/2,pi/4,1.513337979672154) q[17];
cx q[8],q[17];
u3(pi/2,-pi/2,3*pi/4) q[17];
cx q[19],q[15];
u3(0,0,-pi/4) q[15];
u3(0,0,pi/4) q[19];
cx q[19],q[15];
cx q[10],q[15];
u3(0,0,3.50789165393229) q[15];
cx q[10],q[15];
u3(0,0,1.2820214378619426) q[10];
u3(pi/2,-pi,2.0906800914355284) q[15];
u3(2.164547371685579,-pi/4,pi/2) q[19];
cx q[19],q[11];
u3(0,0,-pi/4) q[11];
cx q[19],q[11];
u3(0,1.4065829705916295,-0.6211848071941821) q[11];
cx q[11],q[18];
u3(0,0,-pi/4) q[18];
cx q[7],q[18];
u3(0,0,pi/4) q[18];
cx q[11],q[18];
u3(0,0,pi/4) q[11];
u3(0,0,-pi/4) q[18];
cx q[7],q[18];
cx q[7],q[11];
u3(0,0,pi/4) q[7];
u3(0,0,-pi/4) q[11];
cx q[7],q[11];
cx q[7],q[15];
u3(2.962203988482831,2.829028948173005,-2.829028948173005) q[7];
u3(0,0,-0.7404539978630529) q[11];
cx q[1],q[11];
u3(-2.316622258674755,0,-3.0141162515912887) q[11];
cx q[1],q[11];
u3(0,0,pi/2) q[1];
u3(2.316622258674755,-1.7432168943277964,0) q[11];
u3(pi/2,0,pi) q[15];
u3(0,-0.4008838091908009,-0.6211848071941821) q[18];
cx q[16],q[18];
u3(-0.23366871350533486,0,-3.702717674518062) q[18];
cx q[16],q[18];
u3(pi/2,pi/4,-pi/2) q[16];
cx q[15],q[16];
u3(pi/2,-pi/2,3*pi/4) q[16];
u3(0.23366871350533486,5.510184454300493,0) q[18];
cx q[19],q[10];
u3(0,0,-2.2133023988938563) q[10];
cx q[19],q[10];
cx q[10],q[3];
u3(0,0,-pi/4) q[3];
cx q[6],q[3];
u3(0,0,pi/4) q[3];
cx q[10],q[3];
u3(0,0,-pi/4) q[3];
cx q[6],q[3];
u3(pi/2,1.3024078837634887,-3*pi/4) q[3];
cx q[3],q[18];
u3(0,0,pi/4) q[10];
cx q[6],q[10];
u3(0,0,pi/4) q[6];
u3(0,0,-pi/4) q[10];
cx q[6],q[10];
u3(0,0,0.09838552419685653) q[6];
cx q[2],q[6];
u3(-1.6505170069925001,0,-5.411554772527138) q[6];
cx q[2],q[6];
u3(1.6505170069925001,5.313169248330281,0) q[6];
u3(pi/2,0,pi) q[10];
cx q[14],q[10];
u3(pi/2,0,pi) q[10];
u3(0,0,pi/4) q[14];
cx q[14],q[10];
u3(0,0,-pi/4) q[10];
cx q[14],q[10];
u3(0,1.4065829705916295,-0.6211848071941821) q[10];
u3(0,0,-1.302407883763489) q[18];
cx q[3],q[18];
u3(0,0,1.302407883763489) q[18];
u3(0.22343269284424463,0,0) q[19];
