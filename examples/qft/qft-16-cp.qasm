OPENQASM 2.0;
qreg q[16];
u3(pi/2,0,pi) q[15];
cp(pi/2) q[15],q[14];
u3(pi/2,0,pi) q[14];
cp(pi/4) q[15],q[13];
cp(pi/2) q[14],q[13];
u3(pi/2,0,pi) q[13];
cp(pi/8) q[15],q[12];
cp(pi/4) q[14],q[12];
cp(pi/2) q[13],q[12];
u3(pi/2,0,pi) q[12];
cp(pi/16) q[15],q[11];
cp(pi/8) q[14],q[11];
cp(pi/4) q[13],q[11];
cp(pi/2) q[12],q[11];
u3(pi/2,0,pi) q[11];
cp(pi/32) q[15],q[10];
cp(pi/16) q[14],q[10];
cp(pi/8) q[13],q[10];
cp(pi/4) q[12],q[10];
cp(pi/2) q[11],q[10];
u3(pi/2,0,pi) q[10];
cp(pi/64) q[15],q[9];
cp(pi/32) q[14],q[9];
cp(pi/16) q[13],q[9];
cp(pi/8) q[12],q[9];
cp(pi/4) q[11],q[9];
cp(pi/2) q[10],q[9];
u3(pi/2,0,pi) q[9];
cp(pi/128) q[15],q[8];
cp(pi/64) q[14],q[8];
cp(pi/32) q[13],q[8];
cp(pi/16) q[12],q[8];
cp(pi/8) q[11],q[8];
cp(pi/4) q[10],q[8];
cp(pi/2) q[9],q[8];
u3(pi/2,0,pi) q[8];
cp(pi/256) q[15],q[7];
cp(pi/128) q[14],q[7];
cp(pi/64) q[13],q[7];
cp(pi/32) q[12],q[7];
cp(pi/16) q[11],q[7];
cp(pi/8) q[10],q[7];
cp(pi/4) q[9],q[7];
cp(pi/2) q[8],q[7];
u3(pi/2,0,pi) q[7];
cp(pi/512) q[15],q[6];
cp(pi/256) q[14],q[6];
cp(pi/128) q[13],q[6];
cp(pi/64) q[12],q[6];
cp(pi/32) q[11],q[6];
cp(pi/16) q[10],q[6];
cp(pi/8) q[9],q[6];
cp(pi/4) q[8],q[6];
cp(pi/2) q[7],q[6];
u3(pi/2,0,pi) q[6];
cp(pi/1024) q[15],q[5];
cp(pi/512) q[14],q[5];
cp(pi/256) q[13],q[5];
cp(pi/128) q[12],q[5];
cp(pi/64) q[11],q[5];
cp(pi/32) q[10],q[5];
cp(pi/16) q[9],q[5];
cp(pi/8) q[8],q[5];
cp(pi/4) q[7],q[5];
cp(pi/2) q[6],q[5];
u3(pi/2,0,pi) q[5];
cp(pi/2048) q[15],q[4];
cp(pi/1024) q[14],q[4];
cp(pi/512) q[13],q[4];
cp(pi/256) q[12],q[4];
cp(pi/128) q[11],q[4];
cp(pi/64) q[10],q[4];
cp(pi/32) q[9],q[4];
cp(pi/16) q[8],q[4];
cp(pi/8) q[7],q[4];
cp(pi/4) q[6],q[4];
cp(pi/2) q[5],q[4];
u3(pi/2,0,pi) q[4];
cp(pi/4096) q[15],q[3];
cp(pi/2048) q[14],q[3];
cp(pi/1024) q[13],q[3];
cp(pi/512) q[12],q[3];
cp(pi/256) q[11],q[3];
cp(pi/128) q[10],q[3];
cp(pi/64) q[9],q[3];
cp(pi/32) q[8],q[3];
cp(pi/16) q[7],q[3];
cp(pi/8) q[6],q[3];
cp(pi/4) q[5],q[3];
cp(pi/2) q[4],q[3];
u3(pi/2,0,pi) q[3];
cp(pi/8192) q[15],q[2];
cp(pi/4096) q[14],q[2];
cp(pi/2048) q[13],q[2];
cp(pi/1024) q[12],q[2];
cp(pi/512) q[11],q[2];
cp(pi/256) q[10],q[2];
cp(pi/128) q[9],q[2];
cp(pi/64) q[8],q[2];
cp(pi/32) q[7],q[2];
cp(pi/16) q[6],q[2];
cp(pi/8) q[5],q[2];
cp(pi/4) q[4],q[2];
cp(pi/2) q[3],q[2];
u3(pi/2,0,pi) q[2];
cp(pi/16384) q[15],q[1];
cp(pi/8192) q[14],q[1];
cp(pi/4096) q[13],q[1];
cp(pi/2048) q[12],q[1];
cp(pi/1024) q[11],q[1];
cp(pi/512) q[10],q[1];
cp(pi/256) q[9],q[1];
cp(pi/128) q[8],q[1];
cp(pi/64) q[7],q[1];
cp(pi/32) q[6],q[1];
cp(pi/16) q[5],q[1];
cp(pi/8) q[4],q[1];
cp(pi/4) q[3],q[1];
cp(pi/2) q[2],q[1];
u3(pi/2,0,pi) q[1];
cp(pi/32768) q[15],q[0];
cp(pi/16384) q[14],q[0];
cp(pi/8192) q[13],q[0];
cp(pi/4096) q[12],q[0];
cp(pi/2048) q[11],q[0];
cp(pi/1024) q[10],q[0];
cp(pi/512) q[9],q[0];
cp(pi/256) q[8],q[0];
cp(pi/128) q[7],q[0];
cp(pi/64) q[6],q[0];
cp(pi/32) q[5],q[0];
cp(pi/16) q[4],q[0];
cp(pi/8) q[3],q[0];
cp(pi/4) q[2],q[0];
cp(pi/2) q[1],q[0];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[8];
cp(pi) q[7],q[8];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[8];
cp(pi) q[8],q[7];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[8];
cp(pi) q[7],q[8];
u3(pi/2,0,pi) q[8];
u3(pi/2,0,pi) q[9];
cp(pi) q[6],q[9];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[9];
cp(pi) q[9],q[6];
u3(pi/2,0,pi) q[6];
u3(pi/2,0,pi) q[9];
cp(pi) q[6],q[9];
u3(pi/2,0,pi) q[9];
u3(pi/2,0,pi) q[10];
cp(pi) q[5],q[10];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[10];
cp(pi) q[10],q[5];
u3(pi/2,0,pi) q[5];
u3(pi/2,0,pi) q[10];
cp(pi) q[5],q[10];
u3(pi/2,0,pi) q[10];
u3(pi/2,0,pi) q[11];
cp(pi) q[4],q[11];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[11];
cp(pi) q[11],q[4];
u3(pi/2,0,pi) q[4];
u3(pi/2,0,pi) q[11];
cp(pi) q[4],q[11];
u3(pi/2,0,pi) q[11];
u3(pi/2,0,pi) q[12];
cp(pi) q[3],q[12];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[12];
cp(pi) q[12],q[3];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,pi) q[12];
cp(pi) q[3],q[12];
u3(pi/2,0,pi) q[12];
u3(pi/2,0,pi) q[13];
cp(pi) q[2],q[13];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[13];
cp(pi) q[13],q[2];
u3(pi/2,0,pi) q[2];
u3(pi/2,0,pi) q[13];
cp(pi) q[2],q[13];
u3(pi/2,0,pi) q[13];
u3(pi/2,0,pi) q[14];
cp(pi) q[1],q[14];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[14];
cp(pi) q[14],q[1];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,pi) q[14];
cp(pi) q[1],q[14];
u3(pi/2,0,pi) q[14];
u3(pi/2,0,pi) q[15];
cp(pi) q[0],q[15];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[15];
cp(pi) q[15],q[0];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[15];
cp(pi) q[0],q[15];
u3(pi/2,0,pi) q[15];
