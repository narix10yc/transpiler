OPENQASM 2.0;
qreg q[34];
u3(pi/2,-pi,pi/2) q[0];
u3(0,0,-2.806328008890554) q[3];
u3(pi/2,0,pi/2) q[4];
u3(2.7566598387383947,-pi,-pi/2) q[7];
u3(pi/2,-pi/2,pi/2) q[8];
u3(2.093256584977614,-1.0369979563727978,-2.8149831850822027) q[11];
u3(pi/2,0,-pi/2) q[12];
u3(pi/2,pi/4,-pi/2) q[13];
cx q[6],q[13];
u3(pi/2,0,0) q[6];
u3(pi/2,0.616033802369381,3*pi/4) q[13];
u3(2.8233109372779306,1.9959319301284344,0.7534637215905575) q[15];
u3(0,0,-pi/2) q[16];
cx q[14],q[16];
cx q[14],q[4];
u3(pi/2,0,pi) q[4];
u3(0,0,pi/4) q[14];
cx q[14],q[4];
u3(0,0,-pi/4) q[4];
cx q[14],q[4];
u3(pi/2,pi/4,-pi/4) q[4];
cx q[7],q[4];
u3(pi/2,pi/2,3*pi/4) q[4];
u3(0,0,0.916443388946079) q[14];
cx q[6],q[14];
u3(-2.5921088108635573,0,0) q[6];
u3(-2.5921088108635573,0,0) q[14];
cx q[6],q[14];
u3(pi/2,-pi,-pi) q[6];
u3(0,0,-0.916443388946079) q[14];
u3(0,0,pi/2) q[16];
u3(0.5861855942079602,-pi/2,pi/2) q[17];
u3(pi/2,0,-pi/2) q[19];
u3(0,0,pi/4) q[20];
u3(0,0,1.8127096268029455) q[21];
cx q[22],q[10];
u3(0,0,2.6895506265250533) q[10];
cx q[22],q[10];
u3(0,0,-1.627228389078546) q[10];
cx q[13],q[10];
u3(-1.424086909033014,0,-2.1868301291642775) q[10];
cx q[13],q[10];
u3(0.9068556504890556,1.3841000601813356,1.6866773435788378) q[10];
u3(1.5514094693441318,0.3195884861942133,1.6002200837714877) q[13];
u3(0,0,2.953651852737652) q[22];
cx q[22],q[8];
u3(0,0,-2.953651852737652) q[8];
cx q[22],q[8];
u3(0,0,-3.026290605055357) q[8];
u3(0,0,-0.37889068955813254) q[22];
cx q[22],q[8];
u3(-0.4077265646833264,0,-2.261604265219564) q[8];
cx q[22],q[8];
u3(1.1948918921393379,-1.7326109850457974,-2.723185992210376) q[8];
u3(pi/2,-2.438294309530833,-2.830000731287633) q[23];
u3(0,0,pi/2) q[24];
cx q[19],q[24];
cx q[24],q[19];
u3(0,0,-pi/2) q[19];
cx q[10],q[19];
cx q[19],q[10];
u3(0,0,-1.9302112872309882) q[10];
u3(2.138295834325801,-pi,0) q[19];
u3(0,1.4065829705916304,-1.4065829705916302) q[24];
u3(0,0,3.896785772074006) q[25];
cx q[25],q[21];
u3(-1.837386028758201,0,-3.896785772074006) q[21];
cx q[25],q[21];
u3(1.2391587673182012,-1.8531587742465452,-1.4766170738860644) q[21];
u3(0,0,pi/4) q[25];
cx q[25],q[24];
u3(0,0,-pi/4) q[24];
cx q[25],q[24];
u3(pi/2,0,-3*pi/4) q[24];
u3(pi/2,0,0) q[25];
u3(0,0,2.4866586515126285) q[26];
cx q[26],q[5];
u3(0,0,-2.4866586515126285) q[5];
cx q[26],q[5];
u3(1.1854153237829579,-pi,-0.6549340020771646) q[5];
u3(pi/2,-pi/2,pi/2) q[26];
u3(0,0,pi/2) q[27];
cx q[12],q[27];
cx q[27],q[12];
u3(0,0,pi/2) q[12];
cx q[12],q[26];
u3(0.1357255927526848,-pi,0) q[12];
cx q[23],q[12];
u3(-3.0058670608371085,0,0) q[12];
cx q[23],q[12];
cx q[12],q[11];
u3(0,0,pi/2) q[11];
u3(0,0,pi/4) q[23];
u3(pi/2,-pi,-pi) q[27];
cx q[24],q[27];
u3(-1.877375816341238,0,0) q[27];
cx q[24],q[27];
u3(0,0,5.649448135631651) q[24];
u3(1.877375816341238,-2.046203652908203,0) q[27];
cx q[24],q[27];
u3(-1.4604759441220883,0,-5.649448135631651) q[27];
cx q[24],q[27];
u3(1.4604759441220883,6.124855461744957,0) q[27];
cx q[28],q[1];
u3(0,0,4.6921954351907535) q[1];
cx q[28],q[1];
u3(pi,pi/2,pi/2) q[1];
u3(0,0,pi/2) q[28];
cx q[28],q[17];
u3(0,0,0.543458299623152) q[17];
cx q[1],q[17];
u3(0,0,-0.543458299623152) q[17];
cx q[1],q[17];
u3(0,0,pi/2) q[1];
cx q[1],q[8];
u3(pi,0,pi) q[1];
u3(pi/2,0,pi) q[17];
cx q[7],q[17];
u3(0,0,pi/4) q[7];
u3(pi/2,0,pi) q[17];
cx q[7],q[17];
u3(0,0,-pi/4) q[17];
cx q[7],q[17];
u3(0,1.4065829705916295,-0.6211848071941821) q[17];
u3(pi,0,pi) q[28];
cx q[29],q[2];
cx q[2],q[29];
u3(0,0,0.6392255789285386) q[2];
u3(pi,0,pi) q[29];
cx q[28],q[29];
cx q[29],q[28];
cx q[22],q[28];
u3(0,0,2.50038615634412) q[28];
cx q[22],q[28];
u3(pi/2,-pi/2,pi/2) q[29];
cx q[29],q[21];
u3(0,0,0.12568020492859264) q[21];
cx q[29],q[21];
u3(-pi/2,-pi/2,pi/2) q[21];
u3(-pi/2,-pi/2,pi/2) q[29];
cx q[18],q[30];
cx q[18],q[16];
cx q[16],q[18];
u3(pi/2,-pi,-pi) q[16];
u3(0,0,pi/2) q[18];
u3(pi/2,0,0) q[30];
cx q[30],q[2];
u3(-1.1477619639320884,0,0) q[2];
u3(-1.1477619639320884,0,0) q[30];
cx q[30],q[2];
u3(0,0,-0.9285354668624937) q[2];
cx q[25],q[2];
u3(-0.6880452389384648,0,0) q[2];
u3(-0.6880452389384648,0,0) q[25];
cx q[25],q[2];
u3(0,0,-1.2814864388609413) q[2];
u3(pi/2,1.9697367080299095,-pi) q[25];
u3(pi/2,0,pi) q[31];
cx q[9],q[31];
u3(pi,-2.95660487549645,pi/2) q[9];
u3(pi/4,-pi/2,pi/2) q[31];
cx q[31],q[18];
cx q[18],q[31];
u3(pi/2,0,pi) q[18];
cx q[18],q[2];
u3(0,0,pi/2) q[2];
u3(0,0,pi) q[31];
u3(pi/2,0,0) q[32];
cx q[32],q[3];
u3(-0.04947331491856604,0,0) q[3];
u3(-0.04947331491856604,0,0) q[32];
cx q[32],q[3];
u3(0,0,-1.9060609714941361) q[3];
cx q[0],q[3];
cx q[3],q[0];
u3(pi/2,0,0) q[0];
cx q[0],q[9];
u3(2.4715752870383234,0,0) q[0];
u3(2.369558633357976,-pi,0) q[3];
u3(-2.4715752870383234,0,0) q[9];
cx q[0],q[9];
u3(pi/2,-pi,-pi) q[0];
u3(0,0,-1.7557841048882399) q[9];
cx q[26],q[3];
u3(-2.342830347026714,0,0) q[3];
cx q[26],q[3];
u3(pi/2,0,pi) q[3];
cx q[14],q[3];
u3(pi/2,0,pi) q[3];
u3(0,0,pi/4) q[14];
cx q[14],q[3];
u3(0,0,-pi/4) q[3];
cx q[14],q[3];
u3(0,1.4065829705916295,-0.6211848071941821) q[3];
u3(pi/2,1.1380635300840884,-pi) q[32];
cx q[30],q[32];
u3(0.6534143091069463,0,0) q[30];
u3(-0.6534143091069463,0,0) q[32];
cx q[30],q[32];
u3(pi/2,2.446860444221569,-pi) q[30];
cx q[10],q[30];
u3(-2.593325401888067,0,-4.252150900912426) q[30];
cx q[10],q[30];
u3(2.593325401888067,4.94688311028065,0) q[30];
u3(pi/2,pi/4,0.9097416593744923) q[32];
cx q[6],q[32];
u3(pi/2,-pi/2,3*pi/4) q[32];
u3(pi/2,0,-pi/2) q[33];
cx q[20],q[33];
u3(2.2278781402913115,0,0) q[20];
cx q[5],q[20];
u3(-2.2278781402913115,0,0) q[20];
cx q[5],q[20];
u3(pi/2,4.336072082559096,5.486365822790094) q[5];
u3(0,0,2.6096809233507225) q[20];
cx q[0],q[20];
u3(0,0,-2.6096809233507225) q[20];
cx q[0],q[20];
u3(pi/4,pi/2,-pi/2) q[33];
cx q[33],q[15];
u3(0,0,3.3535500689214395) q[15];
cx q[33],q[15];
u3(pi/2,0,pi) q[15];
u3(pi/2,0,pi) q[33];