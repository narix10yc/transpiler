OPENQASM 2.0;
qreg q[32];
u3(pi/2,6.069625705269507,0.5636945288791153) q[0];
u3(0,0,-2.647354731165292) q[2];
u3(0,0,-2.8002334777778466) q[4];
u3(0,0,2.9863155603772285) q[5];
cx q[5],q[6];
u3(0,0,-2.9863155603772285) q[6];
cx q[5],q[6];
u3(0,0,2.9863155603772285) q[6];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[8];
u3(0,0,pi/4) q[11];
u3(pi/2,-pi/2,pi/2) q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
u3(0,0,1.3595678714685961) q[11];
u3(2.676210964593677,0,-1.9617301945633625) q[13];
u3(0,0,0.2840913382807906) q[16];
cx q[4],q[16];
u3(-2.005556641973186,0,-0.4448043128439716) q[16];
cx q[4],q[16];
u3(0,0,-3.07522245359958) q[4];
u3(2.005556641973186,-2.9808796790266125,0) q[16];
cx q[16],q[12];
cx q[12],q[16];
u3(0,0,pi/4) q[12];
u3(0,0,pi/2) q[16];
u3(pi/2,pi/4,-pi/2) q[17];
cx q[10],q[17];
u3(0,0,1.99688285053917) q[10];
u3(pi/2,-0.6771018155034678,3*pi/4) q[17];
u3(pi/2,2.2574776792039732,3.0572229432838705) q[19];
cx q[10],q[19];
u3(0,0,-1.99688285053917) q[19];
cx q[10],q[19];
u3(1.597681290996152,-0.0694921675175939,-0.0009345795541904067) q[10];
u3(0,0,2.3001487625846266) q[19];
cx q[20],q[15];
cx q[15],q[20];
cx q[5],q[15];
u3(0,0,5.045480200553585) q[15];
cx q[5],q[15];
u3(0,0,pi/2) q[5];
u3(0,1.4065829705916304,2.067408665862695) q[15];
cx q[20],q[6];
u3(0.40409195622400357,2.1350990665281104,-1.8028982237787643) q[6];
u3(0,0,pi/2) q[20];
u3(pi/2,0,pi) q[21];
cx q[3],q[21];
u3(0,0,pi/4) q[3];
u3(pi/2,0,pi) q[21];
cx q[3],q[21];
u3(0,0,-pi/4) q[21];
cx q[3],q[21];
u3(pi/2,0,-2.2421797323868558) q[3];
cx q[3],q[19];
u3(-0.9085640205625568,0,0) q[3];
u3(-0.9085640205625568,0,0) q[19];
cx q[3],q[19];
u3(pi/2,1.271683593021371,-pi) q[3];
u3(pi/2,0,-0.30326591204545617) q[19];
u3(0,2.977379297386526,-0.6211848071941817) q[21];
u3(pi/2,2.368078324242214,-pi/2) q[22];
cx q[23],q[7];
u3(pi/2,0,pi) q[7];
u3(0,0,pi/4) q[23];
cx q[23],q[7];
u3(0,0,-pi/4) q[7];
cx q[23],q[7];
u3(pi/2,3*pi/4,-pi/4) q[7];
u3(pi/2,0,pi) q[23];
u3(pi/2,0,pi) q[24];
cx q[24],q[21];
u3(-1.493044075804772,0,0) q[21];
cx q[24],q[21];
u3(1.493044075804772,-pi/2,0) q[21];
u3(pi/2,2.537103181501896,0.06558460167243602) q[24];
u3(pi/2,0,0) q[25];
cx q[25],q[2];
u3(-1.7002409590405272,0,0) q[2];
u3(1.7002409590405272,0,0) q[25];
cx q[25],q[2];
u3(0,0,-1.0791504768860394) q[2];
cx q[2],q[8];
u3(0,0,-2.5566800991282554) q[8];
cx q[2],q[8];
cx q[2],q[22];
u3(pi/2,0,pi) q[2];
u3(0,0,-2.1557088812564347) q[8];
cx q[22],q[2];
u3(0,0,-pi/4) q[2];
u3(pi/2,2.5617385718964893,-pi) q[25];
cx q[25],q[17];
u3(-0.16562287570944464,0,-1.4387845305596103) q[17];
cx q[25],q[17];
u3(0.16562287570944467,0.36229833195361927,0) q[17];
cx q[17],q[15];
u3(0,0,-pi/4) q[15];
cx q[17],q[15];
u3(pi/2,0.09132538103160259,-1.1236908795626808) q[15];
u3(0,0,-pi) q[25];
cx q[26],q[14];
u3(0,0,3.7317943325122434) q[14];
cx q[26],q[14];
u3(pi/2,-pi/2,-0.562484590689829) q[14];
cx q[13],q[14];
u3(0,0,2.961634558248959) q[14];
cx q[13],q[14];
u3(-pi/2,-pi/2,pi/2) q[13];
cx q[13],q[16];
u3(-pi/2,-pi/2,pi/2) q[14];
cx q[14],q[25];
u3(0,0,1.5098405830098673) q[14];
u3(-2.3322578633200552,0,0) q[16];
cx q[13],q[16];
u3(0,1.4065829705916304,-1.4065829705916302) q[13];
u3(2.380131117064635,pi/2,-pi) q[16];
u3(1.2422338165059894,2.5326268206234026,0.03735670259914059) q[25];
u3(0,0,1.4903154441904005) q[28];
cx q[28],q[1];
u3(0,0,-1.4903154441904005) q[1];
cx q[28],q[1];
u3(0,0,1.4903154441904005) q[1];
u3(pi/2,0,0) q[28];
cx q[28],q[4];
u3(-0.5849156290204441,0,0) q[4];
u3(-0.5849156290204441,0,0) q[28];
cx q[28],q[4];
u3(0,0,3.0752224535995794) q[4];
cx q[4],q[2];
u3(0,0,pi/4) q[2];
cx q[22],q[2];
u3(0,0,-pi/4) q[2];
cx q[4],q[2];
u3(pi/2,0,-3*pi/4) q[2];
u3(0,0,pi/4) q[22];
cx q[4],q[22];
u3(0,0,pi/4) q[4];
u3(0,0,-pi/4) q[22];
cx q[4],q[22];
cx q[2],q[22];
u3(1.7273805535609985,0.42889743631066946,1.242141743696192) q[4];
u3(0,0,-pi/2) q[22];
u3(pi/2,pi/2,-pi/2) q[28];
cx q[28],q[20];
cx q[20],q[28];
u3(pi/2,-2.862468013017275,-pi) q[20];
u3(0,0,2.3970119757248938) q[28];
cx q[10],q[28];
u3(-0.5220420633759129,0,0) q[10];
u3(-0.5220420633759129,0,0) q[28];
cx q[10],q[28];
u3(pi/2,2.188848848838128,-pi) q[10];
u3(0,0,-0.8262156489299972) q[28];
u3(0,1.4065829705916304,-1.4065829705916302) q[29];
cx q[18],q[29];
u3(0,0,-pi/4) q[29];
cx q[27],q[29];
u3(0,0,pi/4) q[29];
cx q[18],q[29];
u3(0,0,pi/4) q[18];
u3(0,0,-pi/4) q[29];
cx q[27],q[29];
cx q[27],q[18];
u3(0,0,-pi/4) q[18];
u3(0,0,pi/4) q[27];
cx q[27],q[18];
u3(0,1.4065829705916304,-1.4065829705916302) q[18];
cx q[1],q[18];
u3(0,0,-pi/4) q[18];
cx q[0],q[18];
u3(0,0,pi/4) q[18];
cx q[1],q[18];
u3(0,0,pi/4) q[1];
u3(0,0,-pi/4) q[18];
cx q[0],q[18];
cx q[0],q[1];
u3(0,0,pi/4) q[0];
u3(0,0,-pi/4) q[1];
cx q[0],q[1];
u3(pi,-1.735009682998163,1.4065829705916304) q[0];
cx q[12],q[0];
u3(0,0,-pi/4) q[0];
cx q[12],q[0];
u3(pi/2,pi/2,-pi/4) q[0];
cx q[0],q[4];
u3(0,0,-pi/4) q[4];
cx q[10],q[4];
u3(0,0,pi/4) q[4];
cx q[0],q[4];
u3(0,0,pi/4) q[0];
u3(0,0,-pi/4) q[4];
cx q[10],q[4];
u3(0,1.4065829705916295,-0.6211848071941821) q[4];
cx q[10],q[0];
u3(0,0,-pi/4) q[0];
u3(0,0,pi/4) q[10];
cx q[10],q[0];
u3(0,0,pi/2) q[10];
u3(pi/2,0,0) q[12];
cx q[12],q[11];
u3(-1.7656228217916774,0,0) q[11];
u3(1.7656228217916774,0,0) q[12];
cx q[12],q[11];
u3(2.4207118262193164,0,2.1279524767905427) q[11];
cx q[10],q[11];
u3(pi,0,pi) q[10];
u3(pi/2,-pi,-pi) q[12];
u3(pi/2,-pi/2,3*pi/4) q[18];
cx q[5],q[18];
u3(pi/2,-pi,-pi) q[5];
u3(0,0,pi/2) q[18];
cx q[8],q[18];
u3(-1.56938366726971,0,0) q[18];
cx q[8],q[18];
u3(0,0,-2.776129686763159) q[8];
cx q[15],q[8];
u3(0,1.4065829705916295,0.16421335620326616) q[8];
cx q[0],q[8];
u3(0,0,-pi/4) q[8];
u3(1.56938366726971,-pi/2,0) q[18];
u3(0,0,-pi/4) q[27];
u3(pi/2,-pi/2,3*pi/4) q[29];
cx q[30],q[9];
u3(0,0,pi/4) q[9];
cx q[9],q[23];
u3(0,0,-pi/4) q[23];
cx q[9],q[23];
u3(pi/2,0,-3*pi/4) q[23];
cx q[7],q[23];
u3(0,0,-pi/4) q[23];
cx q[7],q[23];
u3(0,0,-0.3747218701756405) q[7];
cx q[19],q[7];
u3(-1.4153026083108455,0,0) q[7];
u3(1.4153026083108455,0,0) q[19];
cx q[19],q[7];
u3(0,1.4065829705916304,-1.0318611004159894) q[7];
cx q[14],q[7];
u3(0,0,-pi/4) q[7];
u3(pi/2,-pi/2,-pi) q[19];
cx q[19],q[16];
u3(pi/2,0,-pi) q[16];
u3(pi,0,pi) q[19];
u3(pi/2,0,-3*pi/4) q[23];
cx q[2],q[23];
u3(pi/2,2.368527237084429,1.4709106300195227) q[2];
u3(pi/2,pi/4,-pi) q[23];
cx q[23],q[13];
u3(0,0,-pi/4) q[13];
cx q[23],q[13];
u3(0,1.4065829705916295,-0.6211848071941821) q[13];
cx q[23],q[7];
u3(0,0,pi/4) q[7];
cx q[14],q[7];
u3(0,0,-pi/4) q[7];
u3(0,0,pi/4) q[14];
cx q[23],q[7];
u3(pi/2,0,-3*pi/4) q[7];
cx q[23],q[14];
u3(0,0,-pi/4) q[14];
u3(0,0,pi/4) q[23];
cx q[23],q[14];
u3(0,0,-pi/2) q[14];
u3(pi,-pi/2,pi/2) q[23];
u3(pi,0,pi) q[30];
cx q[30],q[29];
u3(0,0,-pi/4) q[29];
cx q[21],q[29];
u3(0,0,pi/4) q[29];
cx q[30],q[29];
u3(0,0,-pi/4) q[29];
cx q[21],q[29];
u3(pi/2,0,-3*pi/4) q[29];
u3(0,0,pi/4) q[30];
cx q[21],q[30];
u3(0,0,pi/4) q[21];
u3(0,0,-pi/4) q[30];
cx q[21],q[30];
cx q[21],q[22];
u3(0,0,pi/2) q[21];
cx q[21],q[25];
u3(pi/2,pi/2,0) q[22];
cx q[25],q[21];
cx q[21],q[8];
u3(0,0,pi/4) q[8];
cx q[0],q[8];
u3(0,0,pi/4) q[0];
u3(0,0,-pi/4) q[8];
cx q[21],q[8];
u3(0,-2.7410163271754873,-0.6211848071941812) q[8];
cx q[21],q[0];
u3(0,0,-pi/4) q[0];
u3(0,0,pi/4) q[21];
cx q[21],q[0];
u3(pi/2,0,0) q[0];
cx q[25],q[7];
u3(0,0,-pi/4) q[7];
u3(0,0,pi) q[30];
u3(pi/2,-pi,0) q[31];
cx q[26],q[31];
u3(1.2182073294161901,0,0) q[26];
cx q[9],q[26];
u3(-1.2182073294161901,0,0) q[26];
cx q[9],q[26];
u3(pi/2,0,pi) q[9];
u3(pi/2,0,pi) q[26];
cx q[24],q[26];
u3(0,0,pi/4) q[24];
u3(pi/2,0,pi) q[26];
cx q[24],q[26];
u3(0,0,-pi/4) q[26];
cx q[24],q[26];
u3(0,2.191981133989078,-1.4065829705916302) q[24];
cx q[19],q[24];
u3(0,0,-pi/4) q[24];
u3(0,1.4065829705916295,-0.6211848071941821) q[26];
cx q[26],q[17];
cx q[20],q[17];
u3(0,0,1.7219594084828822) q[17];
cx q[20],q[17];
u3(0,0,-2.0381944863474026) q[17];
u3(pi/2,0,pi) q[26];
cx q[13],q[26];
u3(2.5878167753601335,0.8426034266990916,pi/2) q[26];
cx q[29],q[9];
u3(0,0,5.184729972142058) q[9];
cx q[29],q[9];
u3(pi/2,0,pi) q[9];
u3(pi/2,0,pi) q[29];
u3(pi/2,0,pi) q[31];
cx q[31],q[27];
u3(pi/2,0,pi) q[31];
cx q[27],q[31];
u3(0,0,-pi/4) q[31];
cx q[1],q[31];
u3(0,0,pi/4) q[31];
cx q[27],q[31];
u3(0,0,pi/4) q[27];
u3(0,0,-pi/4) q[31];
cx q[1],q[31];
cx q[1],q[27];
u3(0,0,pi/4) q[1];
u3(0,0,-pi/4) q[27];
cx q[1],q[27];
u3(pi/2,0,pi) q[1];
cx q[5],q[1];
u3(0,0,5.32416056116334) q[1];
cx q[5],q[1];
u3(0.2507597910281613,2.4155005381125525,-1.7046050812710551) q[1];
u3(pi/2,0,pi) q[5];
cx q[5],q[18];
u3(0,0,0.3115508656646774) q[18];
cx q[5],q[18];
u3(pi/2,5.96399281125433,3.6400112199694177) q[5];
cx q[5],q[7];
u3(0,0,pi/4) q[7];
u3(pi/2,-pi/2,-pi) q[18];
cx q[25],q[7];
u3(0,0,-pi/4) q[7];
cx q[5],q[7];
u3(0,-0.06594340143785882,-0.6211848071941821) q[7];
u3(0,0,pi/4) q[25];
cx q[5],q[25];
u3(0,0,pi/4) q[5];
u3(0,0,-pi/4) q[25];
cx q[5],q[25];
u3(0,0,pi/2) q[5];
u3(pi/2,0,0) q[25];
cx q[30],q[18];
u3(pi/2,0,pi) q[18];
cx q[11],q[18];
cx q[18],q[11];
cx q[11],q[18];
u3(pi/2,pi/2,pi/2) q[11];
u3(pi/2,0,pi) q[18];
u3(0,0,-1.2498201454297835) q[30];
u3(pi/2,0,-3*pi/4) q[31];
cx q[31],q[27];
u3(pi/2,-pi/2,-pi) q[27];
cx q[9],q[27];
u3(0,0,-pi/4) q[27];
cx q[29],q[27];
u3(0,0,pi/4) q[27];
cx q[9],q[27];
u3(0,0,pi/4) q[9];
u3(0,0,-pi/4) q[27];
cx q[29],q[27];
u3(pi/2,pi/4,-3*pi/4) q[27];
cx q[27],q[22];
u3(0,0,-pi/4) q[22];
cx q[27],q[22];
u3(0,1.4065829705916295,-0.6211848071941821) q[22];
cx q[22],q[2];
u3(0,0,pi/2) q[2];
cx q[22],q[2];
cx q[2],q[22];
u3(pi/2,-pi,-pi) q[2];
u3(0,0,2.262409060108473) q[22];
u3(0,0,0.2798286735790739) q[27];
cx q[27],q[15];
u3(0,0,-0.2798286735790739) q[15];
cx q[27],q[15];
u3(0,0,1.93492172403852) q[15];
cx q[15],q[7];
u3(-2.8124742957252464,0,-1.6550930504594459) q[7];
cx q[15],q[7];
u3(1.4505103142621312,-0.4891308570669257,-1.2090633880329902) q[7];
u3(0,0,2.0182844487714218) q[27];
cx q[27],q[30];
cx q[29],q[9];
u3(0,0,-pi/4) q[9];
u3(0,0,pi/4) q[29];
cx q[29],q[9];
u3(pi/2,0,pi) q[9];
cx q[9],q[16];
u3(0,0,pi/4) q[9];
u3(pi/2,0,pi) q[16];
cx q[9],q[16];
u3(0,0,-pi/4) q[16];
cx q[9],q[16];
cx q[9],q[26];
u3(pi/2,1.5705082272872524,-pi) q[9];
u3(pi/2,-pi/2,3*pi/4) q[16];
cx q[23],q[16];
u3(0,0,1.8701552929886405) q[16];
cx q[23],q[16];
u3(-pi/2,-pi/2,pi/2) q[16];
u3(1.8380268998319367,-3.0544151614015145,2.773034368706629) q[23];
u3(0,0,-pi/4) q[26];
u3(pi/2,3.0370122579034224,1.0072813366793556) q[29];
cx q[29],q[13];
u3(0,0,4.471017104500057) q[13];
cx q[29],q[13];
cx q[13],q[8];
u3(0,0,-2.135586009412469) q[8];
cx q[13],q[8];
u3(0,0,1.29056521461486) q[8];
u3(0,0,-pi/2) q[13];
u3(0,0,2.3830224276918086) q[29];
u3(-3.086675080369793,0,-2.0182844487714218) q[30];
cx q[27],q[30];
u3(4.921395563965318,-pi/2,pi/2) q[27];
u3(2.995869049403098,-3.0938794628006487,-0.07911499834944324) q[30];
u3(0,0,0.50735436836766) q[31];
cx q[3],q[31];
u3(-0.8656232190636872,0,-1.4516014663415147) q[31];
cx q[3],q[31];
u3(pi/2,0,-pi/2) q[3];
cx q[3],q[28];
cx q[28],q[3];
cx q[3],q[24];
cx q[3],q[1];
u3(0,0,-pi/4) q[1];
cx q[4],q[1];
u3(0,0,pi/4) q[1];
cx q[3],q[1];
u3(pi/2,pi/2,3*pi/4) q[1];
u3(pi/2,pi/4,-pi/2) q[4];
cx q[10],q[1];
u3(-2.2711275315015347,0,0) q[1];
cx q[10],q[1];
u3(2.2711275315015347,-pi/2,0) q[1];
cx q[10],q[5];
u3(0.3000350447820344,0,0) q[5];
u3(0,0,pi/2) q[10];
cx q[14],q[4];
u3(pi/2,-pi/2,3*pi/4) q[4];
cx q[14],q[13];
u3(0,0,pi/2) q[13];
cx q[14],q[10];
u3(-1.0535984766914164,0,0) q[10];
cx q[14],q[10];
u3(1.0535984766914164,-pi,0) q[10];
cx q[21],q[3];
cx q[3],q[21];
u3(1.7129811684426093,0,0) q[3];
cx q[4],q[3];
u3(-1.7129811684426093,0,0) q[3];
cx q[4],q[3];
u3(0,0,pi/4) q[3];
u3(pi/2,0,pi) q[4];
u3(0,0,pi/4) q[24];
cx q[19],q[24];
u3(pi/2,0,pi) q[19];
cx q[20],q[19];
u3(pi/2,-pi/2,pi/2) q[19];
cx q[19],q[17];
cx q[17],q[19];
u3(pi/2,0,pi) q[17];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[17];
u3(0,0,0.694811772739969) q[16];
u3(pi/2,0,-2.8334194859672936) q[17];
u3(pi/2,0,0) q[19];
cx q[19],q[8];
u3(-1.6376018280062683,0,0) q[8];
u3(1.6376018280062683,0,0) q[19];
cx q[19],q[8];
u3(2.6947226146470658,-2.4067729888996463,0.039829217129344396) q[8];
u3(pi/2,-3*pi/4,-pi) q[19];
u3(0,0,-1.5880635317164569) q[20];
u3(pi/2,-2.897028330317422,-3*pi/4) q[24];
cx q[24],q[20];
u3(-1.3956478585087364,0,-3.386156976862164) q[20];
cx q[24],q[20];
u3(1.313062528035154,1.3895999002057593,1.617459739270977) q[20];
cx q[1],q[20];
u3(0,0,-pi/4) q[20];
cx q[15],q[20];
u3(0,0,pi/4) q[20];
cx q[1],q[20];
u3(0,0,pi/4) q[1];
u3(0,0,-pi/4) q[20];
cx q[15],q[20];
cx q[15],q[1];
u3(0,0,-pi/4) q[1];
u3(0,0,pi/4) q[15];
cx q[15],q[1];
u3(0,0,-2.965256317005983) q[1];
cx q[1],q[16];
u3(pi,pi/2,pi/2) q[15];
u3(-0.10792619500388743,0,-3.0849216867656826) q[16];
cx q[1],q[16];
cx q[1],q[7];
u3(0,0,-pi/4) q[7];
u3(1.4971891530250754,-1.649798403800564,-2.319368681248421) q[16];
u3(0,1.4065829705916295,-0.6211848071941821) q[20];
u3(pi/2,0,pi) q[24];
cx q[18],q[24];
u3(0,0,0.6396550101677648) q[24];
cx q[18],q[24];
u3(0,1.4065829705916304,-1.4065829705916302) q[18];
u3(pi/2,0,pi) q[24];
cx q[26],q[15];
u3(pi/2,0,pi) q[26];
cx q[15],q[26];
u3(0,0,-pi/4) q[26];
cx q[27],q[20];
u3(pi/2,pi/4,-pi) q[20];
u3(pi/2,-pi/2,0) q[27];
u3(2.918774094910738,-1.6448565764059138,1.2736194077618848) q[28];
cx q[28],q[18];
u3(0,0,-pi/4) q[18];
u3(2.5995405115458103,-0.5246477648138304,-2.6036355712706563) q[31];
cx q[12],q[31];
u3(0,0,0.8803889636704779) q[31];
cx q[12],q[31];
cx q[6],q[31];
u3(0,0,-1.361890522170644) q[6];
u3(2.964268928985956,-pi/2,-0.7801861770948526) q[12];
cx q[21],q[12];
u3(pi/2,0,pi) q[12];
u3(0,0,pi/4) q[21];
cx q[21],q[12];
u3(0,0,-pi/4) q[12];
cx q[21],q[12];
u3(0,1.4065829705916295,-0.6211848071941821) q[12];
cx q[12],q[4];
u3(0,0,6.253778990667537) q[4];
cx q[12],q[4];
u3(pi/2,0.6641512186265057,-pi) q[4];
cx q[4],q[14];
u3(0,0,-0.6990143039013743) q[12];
u3(0,0,-0.6641512186265058) q[14];
cx q[4],q[14];
u3(pi/2,6.11757321851355,3.1468337599563885) q[4];
u3(0,0,2.872109598088402) q[14];
cx q[21],q[18];
u3(0,0,pi/4) q[18];
cx q[25],q[6];
u3(-1.203086399469537,0,0) q[6];
u3(-1.203086399469537,0,0) q[25];
cx q[25],q[6];
u3(pi/2,-pi/2,-1.7797021314191492) q[6];
cx q[3],q[6];
u3(0,0,-pi/4) q[6];
cx q[3],q[6];
cx q[3],q[9];
u3(0,1.4065829705916295,-0.6211848071941821) q[6];
u3(0,0,-3.141304554082149) q[9];
cx q[3],q[9];
u3(0,0,pi/4) q[3];
cx q[11],q[6];
u3(0,0,-pi/4) q[6];
cx q[8],q[6];
u3(0,0,pi/4) q[6];
cx q[11],q[6];
u3(0,0,-pi/4) q[6];
cx q[8],q[6];
u3(0,2.191981133989078,-0.6211848071941821) q[6];
u3(0,0,pi/4) q[11];
cx q[8],q[11];
u3(0,0,pi/4) q[8];
u3(0,0,-pi/4) q[11];
cx q[8],q[11];
u3(0,0,pi/2) q[8];
u3(2.1650804725424577,0,0) q[11];
u3(pi/2,-pi,-pi) q[25];
cx q[22],q[25];
u3(0,0,-2.262409060108473) q[25];
cx q[22],q[25];
u3(0,0,1.912444242867248) q[22];
cx q[2],q[22];
u3(0.4034110078042404,0,0) q[2];
u3(-0.4034110078042404,0,0) q[22];
cx q[2],q[22];
u3(pi/2,-pi,-pi) q[2];
cx q[2],q[7];
u3(0,0,pi/4) q[7];
cx q[1],q[7];
u3(0,0,pi/4) q[1];
u3(0,0,-pi/4) q[7];
cx q[2],q[7];
cx q[2],q[1];
u3(0,0,-pi/4) q[1];
u3(0,0,pi/4) q[2];
cx q[2],q[1];
u3(pi/2,-pi/2,pi/2) q[1];
u3(pi,pi/2,pi/2) q[2];
u3(0,1.4065829705916295,-0.6211848071941821) q[7];
cx q[14],q[7];
u3(0,0,-2.2079583794618967) q[7];
cx q[14],q[7];
u3(0,0,2.2079583794618967) q[7];
u3(0.3437068651831897,-1.2796198858758665,2.7052853706103015) q[22];
cx q[22],q[9];
u3(pi/2,-pi/4,-pi) q[9];
u3(1.5976000030315423,-2.4539247825629067,-pi) q[22];
u3(0,0,-0.9087344713714893) q[25];
cx q[25],q[17];
u3(0,0,2.262809198056701) q[17];
cx q[25],q[17];
u3(pi/2,0,-pi/2) q[17];
cx q[12],q[17];
u3(pi,0,pi) q[12];
u3(4.717115077442174,3.5524775318478166,1.2967072025367952) q[17];
cx q[17],q[9];
u3(0,0,-pi/4) q[9];
cx q[28],q[18];
u3(0,0,-pi/4) q[18];
cx q[21],q[18];
u3(0.0803684745355527,-2.5514264603370522,-pi/4) q[18];
cx q[18],q[1];
u3(pi,1.0547837544377892,-pi) q[18];
u3(0,0,pi/4) q[28];
cx q[21],q[28];
u3(0,0,pi/4) q[21];
u3(0,0,-pi/4) q[28];
cx q[21],q[28];
u3(0,0,-1.117884837269416) q[21];
u3(0.4505603470276938,-pi/2,pi/2) q[28];
cx q[28],q[27];
u3(0,0,3.3170776573801177) q[27];
cx q[28],q[27];
u3(-pi/2,-pi/2,pi/2) q[27];
u3(pi/2,2.396207847619988,-pi/2) q[28];
cx q[28],q[14];
u3(0,0,-0.8254115208250915) q[14];
cx q[28],q[14];
u3(0,0,1.6108096842225397) q[14];
u3(pi/2,pi/4,-pi) q[28];
u3(0,0,0.9247315809112413) q[31];
cx q[0],q[31];
u3(-2.4512616882897174,0,0) q[0];
u3(-2.4512616882897174,0,0) q[31];
cx q[0],q[31];
u3(2.0298710888645073,-2.7845104446065454,-2.779933697446703) q[0];
cx q[13],q[0];
u3(0,0,3.1060030758326915) q[0];
cx q[13],q[0];
u3(pi/2,0,pi) q[0];
cx q[0],q[26];
u3(pi/2,0,pi) q[13];
cx q[23],q[13];
u3(0,0,0.2851478714330057) q[13];
cx q[23],q[13];
u3(pi/2,0,-pi/2) q[13];
u3(2.722522980048455,2.1690914832309844,-2.1690914832309844) q[23];
cx q[12],q[23];
cx q[23],q[12];
u3(pi/2,0,pi) q[12];
u3(0,0,-pi/2) q[23];
u3(0,0,pi/4) q[26];
cx q[15],q[26];
u3(0,0,pi/4) q[15];
u3(0,0,-pi/4) q[26];
cx q[0],q[26];
cx q[0],q[15];
u3(0,0,pi/4) q[0];
u3(0,0,-pi/4) q[15];
cx q[0],q[15];
u3(pi/2,0,-pi/2) q[0];
cx q[0],q[8];
cx q[8],q[0];
u3(pi/2,0,pi) q[0];
u3(0,1.4065829705916304,-1.4065829705916302) q[8];
cx q[11],q[8];
u3(pi/2,0,pi) q[8];
cx q[8],q[28];
u3(pi/2,0,pi) q[11];
u3(pi/2,0,-3*pi/4) q[26];
cx q[26],q[15];
cx q[15],q[25];
u3(pi/2,0,pi) q[15];
cx q[25],q[15];
u3(0,0,-pi/4) q[15];
u3(pi/2,-pi/2,-pi) q[26];
u3(0,0,-pi/4) q[28];
u3(pi/2,0,2.216861072678552) q[31];
cx q[29],q[31];
u3(0,0,1.9287167886865229) q[31];
cx q[29],q[31];
u3(pi,pi/2,pi/2) q[29];
cx q[5],q[29];
cx q[29],q[5];
cx q[5],q[30];
u3(0,0,-pi/2) q[29];
cx q[20],q[29];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[20],q[28];
u3(1.6014084015052115,1.0203890998171765,0.8871494906206099) q[20];
cx q[27],q[23];
u3(0,0,2.981116131402109) q[23];
u3(pi/2,0,0) q[27];
cx q[27],q[23];
u3(-0.5413529080705263,0,0) q[23];
u3(0.5413529080705263,0,0) q[27];
cx q[27],q[23];
u3(0,0,-1.4103198046072118) q[23];
u3(pi/2,0.8560681210719099,-pi) q[27];
u3(0,0,pi/4) q[28];
cx q[8],q[28];
u3(pi/2,0,0) q[8];
u3(pi/2,1.1759838686551705,3*pi/4) q[28];
u3(pi/2,-pi/2,-pi) q[29];
cx q[30],q[5];
u3(pi/2,0,pi) q[5];
cx q[1],q[5];
u3(0,0,6.133709814621516) q[5];
cx q[1],q[5];
u3(pi/2,0,pi) q[1];
u3(pi/2,-pi/4,pi/2) q[5];
cx q[30],q[26];
u3(0,0,0.9160791528206949) q[26];
cx q[30],q[26];
u3(pi/2,0,pi) q[26];
cx q[26],q[9];
u3(0,0,pi/4) q[9];
cx q[17],q[9];
u3(0,2.1919811339890787,-2.1919811339890782) q[9];
u3(4.4970476003096485,1.482653385052716,-1.482653385052716) q[26];
u3(pi/2,0,pi) q[30];
u3(pi/2,0,pi) q[31];
cx q[24],q[31];
cx q[31],q[24];
cx q[19],q[24];
u3(0,0,5.887631286098529) q[24];
cx q[19],q[24];
cx q[10],q[24];
cx q[19],q[15];
u3(0,0,pi/4) q[15];
u3(0,0,4.659600150951167) q[24];
cx q[10],q[24];
u3(0,0,pi/2) q[10];
cx q[10],q[29];
u3(pi/2,pi/2,-1.3913010340704894) q[10];
u3(pi/2,0,pi) q[24];
cx q[7],q[24];
u3(0,0,4.00685371187308) q[24];
cx q[7],q[24];
cx q[7],q[5];
u3(pi/2,-pi/2,-3*pi/4) q[5];
u3(0,1.4065829705916304,-1.4065829705916302) q[24];
cx q[25],q[15];
u3(0,0,-pi/4) q[15];
cx q[19],q[15];
u3(pi/2,0,-3*pi/4) q[15];
u3(0,0,pi/4) q[25];
cx q[19],q[25];
u3(0,0,pi/4) q[19];
u3(0,0,-pi/4) q[25];
cx q[19],q[25];
cx q[15],q[25];
cx q[15],q[0];
u3(0,0,1.8641911930030548) q[0];
cx q[15],q[0];
u3(2.933743381837781,-1.8556094578076623,-1.2966955671654663) q[0];
cx q[15],q[1];
u3(pi/2,0,pi) q[1];
u3(0,0,pi/4) q[15];
cx q[15],q[1];
u3(0,0,-pi/4) q[1];
cx q[15],q[1];
u3(0,2.977379297386526,-0.6211848071941817) q[1];
u3(pi/2,0,pi) q[15];
cx q[19],q[2];
cx q[2],q[19];
cx q[19],q[30];
u3(0,0,pi/4) q[19];
cx q[22],q[2];
u3(0,0,-0.6876678710268865) q[2];
cx q[22],q[2];
u3(pi/2,-pi/2,2.258464197821783) q[2];
u3(0,0,1.9274127883020256) q[22];
cx q[22],q[17];
u3(0,0,-1.9274127883020256) q[17];
cx q[22],q[17];
u3(0,0,2.712810951699474) q[17];
u3(0,0,pi/2) q[22];
u3(pi/2,0,0) q[25];
cx q[25],q[18];
u3(-2.0459100935827226,0,0) q[18];
u3(2.0459100935827226,0,0) q[25];
cx q[25],q[18];
u3(0,0,-1.0547837544377896) q[18];
cx q[18],q[11];
u3(1.5821683650646559,-2.7197745486063782,3.019479637804327) q[11];
u3(2.612058724385495,-1.9741694493421247,-1.8914828440023739) q[25];
cx q[26],q[5];
u3(0,0,-pi/4) q[5];
u3(pi/2,0,pi) q[29];
cx q[29],q[12];
u3(0,0,3.9600839127128857) q[12];
cx q[29],q[12];
u3(2.507232876056639,-pi,-pi) q[12];
u3(pi/2,0,pi) q[29];
cx q[29],q[15];
u3(0,0,-pi/4) q[15];
cx q[18],q[15];
u3(0,0,pi/4) q[15];
cx q[29],q[15];
u3(0,0,-pi/4) q[15];
cx q[18],q[15];
u3(pi/2,-pi/4,-3*pi/4) q[15];
u3(0,0,pi/4) q[29];
cx q[18],q[29];
u3(0,0,pi/4) q[18];
u3(0,0,-pi/4) q[29];
cx q[18],q[29];
u3(pi/2,-pi/2,pi/2) q[18];
cx q[29],q[22];
u3(-1.0664381244505563,0,0) q[22];
cx q[29],q[22];
u3(1.0664381244505563,-pi/2,0) q[22];
u3(pi/2,0,pi) q[30];
cx q[19],q[30];
u3(0,0,-pi/4) q[30];
cx q[19],q[30];
cx q[19],q[1];
u3(-1.2343617454160052,0,0) q[1];
cx q[19],q[1];
u3(1.2343617454160052,-pi/2,0) q[1];
u3(1.8369034662839667,0.4143092208745083,-0.9596657215751843) q[30];
u3(0,0,-1.836231722176035) q[31];
cx q[31],q[21];
u3(-1.792347160937843,0,-3.4981669055626528) q[21];
cx q[31],q[21];
u3(1.6647718296251124,1.7933455752352563,1.5920282857634271) q[21];
cx q[3],q[21];
u3(0,0,-pi/4) q[21];
cx q[3],q[21];
u3(0,0,0.121089979345812) q[3];
u3(pi/2,pi/4,-3*pi/4) q[21];
cx q[21],q[13];
u3(0,0,-pi/4) q[13];
cx q[21],q[13];
u3(pi/2,2.7723502620731146,-3*pi/4) q[13];
cx q[14],q[13];
u3(pi/2,0,pi) q[13];
u3(0,0,pi/4) q[14];
cx q[14],q[13];
u3(0,0,-pi/4) q[13];
cx q[14],q[13];
u3(0,1.4065829705916295,-0.6211848071941821) q[13];
u3(2.8373422936096424,0,0) q[14];
cx q[11],q[14];
u3(-2.8373422936096424,0,0) q[14];
cx q[11],q[14];
cx q[31],q[6];
u3(0,0,-pi/4) q[6];
cx q[16],q[6];
u3(0,0,pi/4) q[6];
u3(pi/2,0,0.6535865115818567) q[16];
cx q[16],q[24];
u3(0,0,3.992358694060005) q[24];
cx q[16],q[24];
u3(pi/2,0,pi) q[16];
u3(pi/2,pi/2,-pi) q[24];
cx q[25],q[24];
cx q[24],q[25];
u3(pi/2,pi/2,-pi) q[24];
cx q[24],q[18];
u3(pi,0,pi) q[24];
u3(0,0,1.0734310906969862) q[25];
cx q[25],q[0];
u3(0,0,-1.0734310906969862) q[0];
cx q[25],q[0];
u3(0,0,1.0734310906969862) q[0];
cx q[28],q[16];
u3(0,0,-1.1759838686551702) q[16];
cx q[28],q[16];
u3(0,0,1.1759838686551702) q[16];
cx q[16],q[12];
u3(0,0,-pi/4) q[12];
cx q[13],q[12];
u3(0,0,pi/4) q[12];
cx q[16],q[12];
u3(0,0,-pi/4) q[12];
cx q[13],q[12];
u3(pi/2,0,-3*pi/4) q[12];
u3(0,0,pi/4) q[16];
cx q[13],q[16];
u3(0,0,pi/4) q[13];
u3(0,0,-pi/4) q[16];
cx q[13],q[16];
u3(1.4074191545466113,4.143747818441389,2.797604813188746) q[28];
cx q[31],q[6];
u3(pi/2,0,3*pi/4) q[6];
cx q[6],q[4];
u3(0,0,-0.4875082287319179) q[4];
cx q[8],q[4];
u3(-0.9173003982720812,0,0) q[4];
u3(-0.9173003982720812,0,0) q[8];
cx q[8],q[4];
u3(0,0,0.4875082287319179) q[4];
cx q[4],q[5];
u3(0,0,pi/4) q[5];
u3(pi/2,-pi,-pi) q[8];
cx q[8],q[19];
cx q[19],q[8];
cx q[8],q[19];
cx q[21],q[6];
cx q[6],q[9];
u3(0,0,-pi/4) q[9];
cx q[7],q[9];
u3(5.819981404527015,2.4463076373463024,0.19634311400517243) q[7];
u3(0,0,pi/4) q[9];
cx q[6],q[9];
u3(0,0,-pi/2) q[6];
u3(0,1.4065829705916304,-2.1919811339890782) q[9];
u3(pi/2,-pi/2,-1.217130280537321) q[21];
cx q[21],q[10];
u3(0,0,1.001859546979167) q[10];
cx q[21],q[10];
u3(-pi/2,-pi/2,pi/2) q[10];
u3(-pi/2,-pi/2,pi/2) q[21];
cx q[23],q[9];
u3(pi/2,0,pi) q[9];
cx q[26],q[5];
u3(0,0,-pi/4) q[5];
cx q[4],q[5];
u3(pi/2,0,-3*pi/4) q[5];
u3(0,0,pi/4) q[26];
cx q[4],q[26];
u3(0,0,pi/4) q[4];
u3(0,0,-pi/4) q[26];
cx q[4],q[26];
u3(2.1602723064437126,-0.4328633712863206,0.3592912934728436) q[31];
cx q[3],q[31];
u3(-0.9643336512975215,0,-3.6184900030255) q[31];
cx q[3],q[31];
cx q[3],q[6];
u3(0,0,pi/2) q[6];
u3(0.9643336512975215,4.475448052384501,0) q[31];
cx q[1],q[31];
u3(pi/2,0,pi) q[1];
cx q[31],q[1];
u3(0,0,-pi/4) q[1];
cx q[2],q[1];
u3(0,0,pi/4) q[1];
cx q[31],q[1];
u3(0,0,-pi/4) q[1];
cx q[2],q[1];
u3(pi/2,0,-3*pi/4) q[1];
u3(0,0,pi/4) q[31];
cx q[2],q[31];
u3(0,0,pi/4) q[2];
u3(0,0,-pi/4) q[31];
cx q[2],q[31];
cx q[1],q[31];