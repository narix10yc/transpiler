OPENQASM 2.0;
qreg q[16];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,pi/4) q[0];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,-pi/4) q[1];
cx q[0],q[1];
u3(pi/2,0,pi/2) q[1];
u3(pi,pi/2,pi/2) q[2];
u3(0,0,-pi/2) q[3];
u3(0,0,0.8271249292611296) q[4];
u3(3.9794086265531,0.09679560429260821,5.675915954089324) q[5];
u3(pi/2,-pi/2,pi/2) q[6];
u3(pi/2,0,pi) q[7];
u3(5.137991895997664,4.6117685738312675,2.860921922405386) q[8];
u3(pi/2,0,pi) q[9];
cx q[10],q[7];
u3(pi/2,0.5958577110658174,-pi) q[7];
cx q[10],q[0];
cx q[0],q[10];
cx q[10],q[0];
u3(0,0,1.1462538483798577) q[10];
cx q[1],q[10];
u3(-2.4351188312698064,0,0) q[1];
u3(-2.4351188312698064,0,0) q[10];
cx q[1],q[10];
u3(0,0,1.8973983871389253) q[10];
u3(pi/2,-pi/2,pi/2) q[11];
cx q[11],q[6];
cx q[6],q[11];
cx q[12],q[3];
u3(2.1411826087119077,2.5722281405114584,0.049666747092314) q[3];
u3(0,0,1.4963696597373888) q[12];
cx q[12],q[0];
u3(0,0,-0.5352895122662508) q[0];
cx q[12],q[0];
u3(0,0,-0.9136022798810841) q[0];
cx q[1],q[0];
u3(-1.9256087052674622,0,0) q[0];
u3(-1.9256087052674622,0,0) q[1];
cx q[1],q[0];
u3(pi/2,pi/4,-1.6927008614424586) q[0];
u3(pi/2,-pi,-pi) q[1];
u3(0,0,-0.05625784140099199) q[12];
u3(5.577607283370649,5.87672158908052,1.555108436411976) q[13];
cx q[13],q[2];
u3(pi/2,0,pi) q[13];
cx q[2],q[13];
u3(0,0,-pi/4) q[13];
cx q[8],q[13];
u3(0,0,pi/4) q[13];
cx q[2],q[13];
u3(0,0,pi/4) q[2];
u3(0,0,-pi/4) q[13];
cx q[8],q[13];
cx q[8],q[2];
u3(0,0,-pi/4) q[2];
u3(0,0,pi/4) q[8];
cx q[8],q[2];
u3(0,0,pi/2) q[8];
u3(pi/2,0,-3*pi/4) q[13];
cx q[13],q[2];
u3(pi/2,-2.469947449990601,-pi/2) q[2];
cx q[14],q[4];
u3(0,0,-0.8271249292611296) q[4];
cx q[14],q[4];
u3(pi/2,0,0) q[4];
cx q[4],q[7];
u3(-2.9249889091291896,0,0) q[4];
u3(-2.9249889091291896,0,0) q[7];
cx q[4],q[7];
u3(pi/2,-pi,-pi) q[4];
u3(pi/2,-pi/2,0.9749386157290791) q[7];
cx q[8],q[7];
u3(pi,pi/2,-pi) q[8];
cx q[10],q[7];
u3(0,0,-3.043652235518783) q[7];
cx q[10],q[7];
u3(0,0,3.043652235518783) q[7];
cx q[1],q[7];
cx q[7],q[1];
u3(0,0,1.436305818174585) q[1];
u3(pi/2,0,0) q[7];
cx q[7],q[1];
u3(-2.892987197831498,0,0) q[1];
u3(2.892987197831498,0,0) q[7];
cx q[7],q[1];
u3(0,0,-1.4363058181745851) q[1];
u3(pi/2,-pi,-pi) q[7];
cx q[10],q[0];
u3(pi/2,0,3*pi/4) q[0];
cx q[13],q[4];
u3(pi/2,0,pi) q[4];
u3(0,0,pi/2) q[13];
u3(pi/2,0,pi) q[14];
cx q[5],q[14];
u3(pi/2,0,pi) q[14];
cx q[5],q[14];
cx q[14],q[5];
cx q[5],q[14];
u3(pi/2,-pi/2,pi/2) q[5];
cx q[8],q[5];
u3(pi,0,pi) q[8];
u3(0,1.4065829705916304,-1.4065829705916302) q[14];
cx q[15],q[9];
u3(pi/2,0,pi/2) q[9];
cx q[9],q[6];
u3(pi/2,0,pi) q[9];
cx q[6],q[9];
u3(0,0,-pi/4) q[9];
cx q[11],q[9];
u3(0,0,pi/4) q[9];
cx q[6],q[9];
u3(0,0,pi/4) q[6];
u3(0,0,-pi/4) q[9];
cx q[11],q[9];
u3(pi/2,0,-3*pi/4) q[9];
cx q[11],q[6];
u3(0,0,-pi/4) q[6];
u3(0,0,pi/4) q[11];
cx q[11],q[6];
cx q[9],q[6];
cx q[6],q[4];
u3(0,0,-pi/4) q[4];
u3(pi/2,0,-pi/2) q[9];
cx q[9],q[13];
u3(0,0,pi/4) q[11];
cx q[11],q[14];
cx q[13],q[9];
u3(0,0,2.8398917680649083) q[9];
u3(pi/2,0.3395173839559318,-pi) q[13];
cx q[9],q[13];
u3(-2.283963378138331,0,-3.5049256808750746) q[13];
cx q[9],q[13];
u3(0.7134948071290241,0.027521373702894536,-0.03639323652616122) q[13];
u3(0,0,-pi/4) q[14];
cx q[11],q[14];
u3(pi/2,0,0) q[11];
cx q[11],q[12];
u3(0.47633879768262666,0,0) q[11];
u3(-0.47633879768262666,0,0) q[12];
cx q[11],q[12];
u3(pi/2,-pi,-pi) q[11];
u3(pi/2,pi/4,-3.0853348121888016) q[12];
cx q[11],q[12];
u3(0,0,-pi/4) q[12];
u3(2.8337503792756062,0,pi/4) q[14];
cx q[5],q[14];
u3(-2.8337503792756062,0,0) q[14];
cx q[5],q[14];
u3(0,0,4.317124035514071) q[5];
cx q[14],q[12];
u3(0,0,pi/4) q[12];
cx q[11],q[12];
u3(0,0,1.9390552519385853) q[11];
u3(pi/2,pi/16,3*pi/4) q[12];
u3(1.5711928104827673,3.141487775244059,-1.423111250456162) q[15];
cx q[15],q[3];
u3(-2.720193867632416,0,0) q[3];
u3(2.720193867632416,0,0) q[15];
cx q[15],q[3];
u3(0,0,1.058483649874912) q[3];
cx q[3],q[4];
u3(0,0,pi/4) q[4];
cx q[6],q[4];
u3(0,0,-pi/4) q[4];
cx q[3],q[4];
u3(pi/2,0,-3*pi/4) q[4];
u3(0,0,pi/4) q[6];
cx q[3],q[6];
u3(0,0,pi/4) q[3];
u3(0,0,-pi/4) q[6];
cx q[3],q[6];
u3(0,0,pi/2) q[3];
cx q[6],q[0];
u3(0,0,pi/4) q[0];
cx q[8],q[4];
cx q[4],q[8];
u3(0,0,1.6851179880029794) q[4];
cx q[5],q[4];
u3(-3.0076345957080557,0,-4.317124035514071) q[4];
cx q[5],q[4];
u3(3.0076345957080557,2.6320060475110916,0) q[4];
u3(0,0,2.825532331959378) q[5];
u3(pi/2,0,0) q[8];
cx q[13],q[5];
u3(-1.708111289578659,0,0) q[5];
u3(-1.708111289578659,0,0) q[13];
cx q[13],q[5];
u3(0,0,1.775320080089318) q[5];
u3(pi/2,-pi,pi/2) q[13];
u3(pi/2,-0.10635282942965363,-pi) q[15];
cx q[15],q[2];
u3(-0.10743451011210949,0,-3.0352398241601395) q[2];
cx q[15],q[2];
u3(0.10743451011210949,0.7927982937660512,0) q[2];
cx q[2],q[3];
u3(-2.0026862625004416,0,0) q[3];
cx q[2],q[3];
u3(pi,pi/4,-pi) q[2];
u3(1.779419731032813,-1.1913095593737084,-2.878982520490239) q[3];
cx q[12],q[3];
u3(0,0,-pi/16) q[3];
cx q[12],q[3];
u3(0,1.4065829705916304,-1.210233429742268) q[3];
cx q[12],q[4];
u3(0,0,-pi/16) q[4];
cx q[4],q[3];
u3(0,0,pi/16) q[3];
cx q[4],q[3];
u3(0,1.4065829705916304,-1.6029325114409922) q[3];
cx q[12],q[4];
u3(0,0,pi/16) q[4];
cx q[4],q[3];
u3(0,0,-pi/16) q[3];
cx q[4],q[3];
u3(0,1.4065829705916304,-1.210233429742268) q[3];
cx q[4],q[14];
u3(0,0,-pi/16) q[14];
cx q[14],q[3];
u3(0,0,pi/16) q[3];
cx q[14],q[3];
u3(0,1.4065829705916304,-1.6029325114409922) q[3];
cx q[12],q[14];
u3(0,0,pi/16) q[14];
cx q[14],q[3];
u3(0,0,-pi/16) q[3];
cx q[14],q[3];
u3(0,1.4065829705916304,-1.210233429742268) q[3];
cx q[4],q[14];
u3(2.951872568978661,-pi/2,pi/2) q[4];
u3(0,0,-pi/16) q[14];
cx q[14],q[3];
u3(0,0,pi/16) q[3];
cx q[14],q[3];
u3(0,1.4065829705916304,-1.6029325114409922) q[3];
cx q[12],q[14];
u3(0,0,pi/16) q[14];
cx q[14],q[3];
u3(0,0,-pi/16) q[3];
cx q[14],q[3];
u3(pi/2,pi/4,-15*pi/16) q[3];
cx q[15],q[0];
u3(0,0,-pi/4) q[0];
cx q[6],q[0];
u3(0,0,pi/4) q[0];
u3(2.7825639072248802,-pi/2,-2.589631338019876) q[6];
cx q[2],q[6];
u3(0,0,-pi/4) q[6];
cx q[2],q[6];
u3(0,0,-pi/4) q[2];
u3(0,1.4065829705916295,-0.6211848071941821) q[6];
cx q[15],q[0];
u3(pi/2,pi/4,3*pi/4) q[0];
cx q[10],q[0];
u3(pi/2,pi/4,3*pi/4) q[0];
u3(0,0,3*pi/4) q[10];
cx q[1],q[10];
u3(-0.333303548242666,0,0) q[10];
cx q[1],q[10];
u3(0,0,1.166873031485224) q[1];
cx q[5],q[1];
u3(-2.7952993580466567,0,-4.600852412048695) q[1];
cx q[5],q[1];
u3(2.7952993580466567,3.4339793805634713,0) q[1];
u3(pi/2,0,pi) q[5];
u3(0.3673163630259599,2.922971338405006,-1.5866999566284543) q[10];
cx q[11],q[0];
u3(0,0,-1.9390552519385853) q[0];
cx q[11],q[0];
u3(0,0,1.9390552519385853) q[0];
u3(2.4849236762576394,-pi/2,0) q[11];
cx q[14],q[0];
cx q[0],q[14];
cx q[14],q[0];
cx q[14],q[10];
u3(0,0,-0.31588019167827497) q[10];
u3(0,0,pi/2) q[14];
u3(0,0,1.5081313851928342) q[15];
cx q[15],q[9];
u3(0,0,-1.5081313851928342) q[9];
cx q[15],q[9];
u3(0,0,1.5081313851928342) q[9];
u3(0,0,pi/16) q[15];
cx q[15],q[8];
u3(0,0,-pi/16) q[8];
cx q[15],q[8];
u3(0,1.4065829705916304,-1.210233429742268) q[8];
cx q[15],q[7];
u3(0,0,-pi/16) q[7];
cx q[7],q[8];
u3(0,0,pi/16) q[8];
cx q[7],q[8];
u3(0,1.4065829705916304,-1.6029325114409922) q[8];
cx q[15],q[7];
u3(0,0,pi/16) q[7];
cx q[7],q[8];
u3(0,0,-pi/16) q[8];
cx q[7],q[8];
cx q[7],q[9];
u3(0,1.4065829705916304,-1.210233429742268) q[8];
u3(0,0,-pi/16) q[9];
cx q[9],q[8];
u3(0,0,pi/16) q[8];
cx q[9],q[8];
u3(0,1.4065829705916304,-1.6029325114409922) q[8];
cx q[15],q[9];
u3(0,0,pi/16) q[9];
cx q[9],q[8];
u3(0,0,-pi/16) q[8];
cx q[9],q[8];
cx q[7],q[9];
u3(pi/2,-pi,0) q[7];
u3(0,1.4065829705916304,-1.210233429742268) q[8];
u3(0,0,-pi/16) q[9];
cx q[9],q[8];
u3(0,0,pi/16) q[8];
cx q[9],q[8];
u3(0,1.4065829705916304,-1.6029325114409922) q[8];
cx q[15],q[9];
u3(0,0,pi/16) q[9];
cx q[9],q[8];
u3(0,0,-pi/16) q[8];
cx q[9],q[8];
u3(pi/2,0,-15*pi/16) q[8];
cx q[8],q[6];
u3(pi/2,0,pi) q[6];
cx q[2],q[6];
cx q[6],q[2];
cx q[2],q[6];
u3(pi/2,0,pi) q[8];
cx q[7],q[8];
u3(0,0,2.00323333138011) q[8];
cx q[7],q[8];
u3(pi,0,-pi) q[7];
u3(0,1.4065829705916304,-1.4065829705916302) q[8];
u3(0,1.4065829705916304,-1.4065829705916302) q[9];
cx q[3],q[9];
u3(0,0,-pi/4) q[9];
cx q[3],q[9];
u3(pi/2,0,pi) q[3];
cx q[4],q[3];
u3(pi/2,0,pi) q[3];
cx q[3],q[14];
u3(0,2.191981133989078,-0.6211848071941821) q[9];
cx q[9],q[5];
u3(0,0,-pi/4) q[5];
cx q[9],q[5];
u3(pi/2,0,-3*pi/4) q[5];
u3(0,0,pi/2) q[9];
cx q[2],q[9];
u3(-1.956960023191532,0,0) q[9];
cx q[2],q[9];
u3(pi/2,0,pi) q[2];
u3(1.956960023191532,2.464958569043831,0) q[9];
cx q[10],q[2];
u3(0,0,-pi/4) q[2];
u3(-1.9678066188070988,0,0) q[14];
cx q[3],q[14];
u3(0.8044127645820823,-1.4294810126326762,0.6730436331221115) q[3];
u3(0.3970102920122024,-pi,0) q[14];
cx q[14],q[11];
cx q[11],q[14];
u3(1.7444780821304537,-2.5495404649539797,2.783662612134389) q[11];
u3(pi/2,0,0) q[14];
u3(0,0,pi/2) q[15];
cx q[12],q[15];
u3(-0.18219565000583296,0,0) q[15];
cx q[12],q[15];
u3(pi/2,pi/4,-pi) q[12];
cx q[1],q[12];
u3(pi/2,0,3*pi/4) q[12];
cx q[0],q[12];
u3(0,0,pi/4) q[12];
cx q[13],q[12];
u3(0,0,-pi/4) q[12];
cx q[0],q[12];
u3(pi/2,0,pi) q[0];
u3(0,0,pi/4) q[12];
cx q[13],q[12];
u3(pi/2,pi/4,3*pi/4) q[12];
cx q[1],q[12];
u3(0,0,2.1616467652277542) q[1];
cx q[7],q[1];
u3(-0.588651961735501,0,0) q[1];
u3(0.588651961735501,0,0) q[7];
cx q[7],q[1];
u3(pi/2,0,-2.161646765227754) q[1];
u3(1.212716529741259,0.5731815254246189,0.9201251120697629) q[7];
u3(pi/2,0,3*pi/4) q[12];
cx q[4],q[12];
u3(0,0,-pi/4) q[12];
cx q[5],q[12];
u3(0,0,pi/4) q[12];
cx q[4],q[12];
u3(0,0,pi/4) q[4];
u3(0,0,-pi/4) q[12];
cx q[5],q[12];
cx q[5],q[4];
u3(0,0,-pi/4) q[4];
u3(0,0,pi/4) q[5];
cx q[5],q[4];
u3(pi/2,pi/4,-pi) q[4];
cx q[5],q[2];
u3(0,0,pi/4) q[2];
cx q[8],q[4];
u3(pi/2,0,3*pi/4) q[4];
cx q[10],q[2];
u3(0,0,-pi/4) q[2];
cx q[5],q[2];
u3(1.1111062090837551,0.8095922932304793,-0.6956655571801829) q[2];
u3(0,0,pi/4) q[10];
cx q[5],q[10];
u3(0,0,pi/4) q[5];
u3(0,0,-pi/4) q[10];
cx q[5],q[10];
cx q[5],q[9];
u3(pi/2,0,pi) q[5];
cx q[9],q[5];
u3(0,0,-pi/4) q[5];
u3(0.7074811961796968,0.7144948542898053,1.1033500735871273) q[10];
u3(0,1.4065829705916295,-0.6211848071941821) q[12];
cx q[12],q[4];
u3(0,0,pi/4) q[4];
cx q[6],q[4];
u3(0,0,-pi/4) q[4];
cx q[12],q[4];
u3(0,0,pi/4) q[4];
cx q[6],q[4];
u3(pi/2,pi/4,3*pi/4) q[4];
cx q[7],q[6];
u3(0,0,-1.902552888332848) q[6];
cx q[7],q[6];
u3(0,0,-1.532488530635942) q[6];
u3(5.438047291228609,-pi/2,pi/2) q[7];
cx q[8],q[4];
u3(0,-pi,-pi/4) q[4];
u3(1.4272231512388283,2.624307745478178,0) q[8];
u3(0,0,2.4837330329155733) q[12];
cx q[3],q[12];
u3(-1.3150993212664035,0,0) q[3];
u3(-1.3150993212664035,0,0) q[12];
cx q[3],q[12];
u3(pi/2,1.768563879248977,-pi) q[3];
cx q[4],q[3];
u3(-2.2057328241940524,0,0) q[3];
u3(-2.2057328241940524,0,0) q[4];
cx q[4],q[3];
u3(0,0,1.373028774340816) q[3];
u3(pi/2,-pi/2,-pi) q[4];
cx q[3],q[4];
u3(-1.6879967040989408,0,0) q[4];
cx q[3],q[4];
u3(pi/2,0,pi) q[3];
u3(pi/2,1.687996704098941,pi/2) q[4];
u3(pi/2,-pi/2,-0.9129367061206772) q[12];
u3(pi,-2.0660633388145566,pi/2) q[13];
cx q[1],q[13];
u3(0.6431781159342977,0,0) q[1];
u3(-0.6431781159342977,0,0) q[13];
cx q[1],q[13];
u3(pi,-0.3289523345111953,2.8126403190785982) q[1];
u3(0,0,-2.6463256415701335) q[13];
cx q[13],q[1];
u3(pi/2,0,-pi) q[1];
cx q[13],q[2];
u3(0,0,-pi/4) q[2];
u3(0.48914419283369814,-2.6832893636755832,-0.46940132176692373) q[15];
cx q[15],q[0];
u3(0,0,3.0372610790158303) q[0];
cx q[14],q[0];
u3(-2.6080198057092927,0,0) q[0];
u3(-2.6080198057092927,0,0) q[14];
cx q[14],q[0];
u3(0,1.4065829705916304,1.8393412575721264) q[0];
u3(pi/2,-pi,-pi) q[14];
cx q[15],q[5];
u3(0,0,pi/4) q[5];
cx q[9],q[5];
u3(0,0,-pi/4) q[5];
u3(0,0,pi/4) q[9];
cx q[15],q[5];
u3(pi/2,0,-3*pi/4) q[5];
cx q[15],q[9];
u3(0,0,-pi/4) q[9];
u3(0,0,pi/4) q[15];
cx q[15],q[9];
cx q[5],q[9];
u3(0,0,1.2889261284412927) q[5];
cx q[5],q[14];
cx q[9],q[2];
u3(0,0,pi/4) q[2];
cx q[13],q[2];
u3(0,0,-pi/4) q[2];
cx q[9],q[2];
u3(pi/2,0,-3*pi/4) q[2];
cx q[2],q[7];
cx q[7],q[2];
cx q[2],q[7];
u3(pi/2,0,pi) q[2];
cx q[7],q[3];
u3(pi/2,pi/2,-pi) q[3];
u3(pi/2,0,pi) q[7];
u3(0,0,pi/4) q[13];
cx q[9],q[13];
u3(0,0,pi/4) q[9];
u3(0,0,-pi/4) q[13];
cx q[9],q[13];
u3(0,0,pi/2) q[9];
cx q[12],q[9];
u3(-2.063407693708017,0,0) q[9];
cx q[12],q[9];
u3(2.063407693708017,-pi/2,0) q[9];
u3(0,0,1.2881738536454206) q[12];
u3(0,0,0.3985568140060498) q[13];
cx q[13],q[1];
u3(0,0,-0.3985568140060498) q[1];
cx q[13],q[1];
u3(0,0,-2.7957217138668957) q[1];
u3(0,0,-1.2889261284412927) q[14];
cx q[5],q[14];
u3(pi/2,0,pi) q[5];
u3(0,0,-1.7219142235392597) q[14];
cx q[14],q[6];
u3(-2.5913847058904658,0,-0.9326810412651483) q[6];
cx q[14],q[6];
u3(2.5913847058904658,1.2261298066441448,0) q[6];
cx q[6],q[2];
u3(pi/2,0,-pi/2) q[2];
cx q[3],q[2];
u3(0,0,pi/16) q[2];
u3(pi/2,-pi,-pi) q[3];
u3(0,0,pi/2) q[6];
u3(0,0,pi/2) q[14];
cx q[13],q[14];
u3(-0.8607516717212791,0,0) q[14];
cx q[13],q[14];
u3(pi/2,0,-pi/2) q[13];
cx q[13],q[6];
cx q[6],q[13];
u3(pi/2,pi/2,-pi) q[6];
u3(0.860751671721279,-pi/4,0) q[14];
u3(0,0,pi/4) q[15];
cx q[15],q[0];
u3(0,0,-pi/4) q[0];
cx q[15],q[0];
u3(0,1.4065829705916295,-0.6211848071941821) q[0];
cx q[0],q[8];
u3(0,0,-2.505045172600935) q[8];
cx q[0],q[8];
u3(0,0,-0.4377140994642138) q[0];
u3(pi/2,1.5350447002559449,2.5194283058522284) q[8];
cx q[12],q[0];
u3(-3.0278616216565646,0,-1.2881738536454206) q[0];
cx q[12],q[0];
u3(1.5883274798658473,-3.0292151594571415,-1.724901836434464) q[0];
u3(pi/2,0,pi) q[12];
cx q[7],q[12];
u3(0,0,4.958780412887741) q[12];
cx q[7],q[12];
u3(pi/2,0,pi) q[7];
u3(pi/2,0,pi) q[12];
cx q[14],q[0];
u3(0,0,-pi/4) q[0];
cx q[14],q[0];
u3(0,1.4065829705916295,-0.6211848071941821) q[0];
cx q[3],q[0];
u3(0,0,0.4872497975643216) q[0];
cx q[3],q[0];
u3(pi/2,0,pi) q[0];
u3(pi/2,0,pi) q[3];
u3(0,1.4065829705916304,-1.4065829705916302) q[14];
cx q[12],q[14];
u3(0,0,-pi/4) q[14];
cx q[8],q[14];
u3(0,0,pi/4) q[14];
cx q[12],q[14];
u3(0,0,pi/4) q[12];
u3(0,0,-pi/4) q[14];
cx q[8],q[14];
cx q[8],q[12];
u3(0,0,pi/4) q[8];
u3(0,0,-pi/4) q[12];
cx q[8],q[12];
u3(pi/2,0,pi) q[8];
cx q[3],q[8];
u3(0,0,5.151712864217835) q[8];
cx q[3],q[8];
u3(pi/2,0,pi) q[3];
u3(pi/2,0,-pi) q[8];
u3(pi,pi/2,pi/2) q[12];
cx q[12],q[6];
u3(pi/2,0,pi) q[6];
u3(0,0,pi/4) q[12];
cx q[12],q[6];
u3(0,0,-pi/4) q[6];
cx q[12],q[6];
u3(2.0502346913086233,0.3407777433249466,-1.9645701153556288) q[6];
u3(0,0,pi/2) q[12];
u3(pi/2,0,pi/4) q[14];
cx q[15],q[5];
u3(0,0,-pi/4) q[5];
cx q[10],q[5];
u3(0,0,pi/4) q[5];
cx q[15],q[5];
u3(0,0,-pi/4) q[5];
cx q[10],q[5];
u3(pi/2,3.0882061151816718,-3*pi/4) q[5];
cx q[5],q[9];
u3(0,0,-3.0882061151816718) q[9];
cx q[5],q[9];
u3(6.1143624218932455,5.9835423613848455,1.5015084738847413) q[5];
u3(0,0,-0.8387847018055696) q[9];
u3(0,0,pi/4) q[15];
cx q[10],q[15];
u3(0,0,pi/4) q[10];
u3(0,0,-pi/4) q[15];
cx q[10],q[15];
cx q[10],q[1];
u3(pi/2,0,0.05268587428315197) q[1];
cx q[2],q[1];
u3(0,0,-pi/16) q[1];
cx q[2],q[1];
u3(0,1.4065829705916304,-1.210233429742268) q[1];
cx q[2],q[13];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[9],q[10];
u3(pi,0,pi) q[9];
u3(0,0,0.38501512245671704) q[10];
cx q[10],q[0];
u3(0,0,-0.38501512245671704) q[0];
cx q[10],q[0];
u3(0.27838799743787257,0,0.3850151224567169) q[0];
u3(0,0,-pi/16) q[13];
cx q[13],q[1];
u3(0,0,pi/16) q[1];
cx q[13],q[1];
u3(0,1.4065829705916304,-1.6029325114409922) q[1];
cx q[2],q[13];
u3(0,0,pi/16) q[13];
cx q[13],q[1];
u3(0,0,-pi/16) q[1];
cx q[13],q[1];
u3(0,1.4065829705916304,-1.210233429742268) q[1];
cx q[13],q[5];
u3(0,0,-pi/16) q[5];
cx q[5],q[1];
u3(0,0,pi/16) q[1];
cx q[5],q[1];
u3(0,1.4065829705916304,-1.6029325114409922) q[1];
cx q[2],q[5];
u3(0,0,pi/16) q[5];
cx q[5],q[1];
u3(0,0,-pi/16) q[1];
cx q[5],q[1];
u3(0,1.4065829705916304,-1.210233429742268) q[1];
cx q[13],q[5];
u3(0,0,-pi/16) q[5];
cx q[5],q[1];
u3(0,0,pi/16) q[1];
cx q[5],q[1];
u3(0,1.4065829705916304,-1.6029325114409922) q[1];
cx q[2],q[5];
u3(pi/2,-2.293932994940976,-2.00461354609279) q[2];
u3(0,0,pi/16) q[5];
cx q[5],q[1];
u3(0,0,-pi/16) q[1];
cx q[5],q[1];
u3(pi/2,-2.1706114351683734,-15*pi/16) q[1];
cx q[13],q[11];
u3(pi/2,pi/2,-3*pi/4) q[11];
u3(0,0,pi/4) q[13];
cx q[15],q[4];
u3(0,0,0.5653240220185548) q[4];
cx q[15],q[4];
u3(0.8529265395445732,-3.0463152695421334,0.2608500157217213) q[4];
cx q[5],q[4];
u3(0,0,-0.305459456984836) q[4];
cx q[5],q[4];
u3(0,0,pi/16) q[4];
cx q[4],q[3];
u3(0,0,-pi/16) q[3];
cx q[4],q[3];
u3(0,1.4065829705916304,-1.210233429742268) q[3];
cx q[4],q[5];
u3(0,0,-pi/16) q[5];
cx q[5],q[3];
u3(0,0,pi/16) q[3];
cx q[5],q[3];
u3(0,1.4065829705916304,-1.6029325114409922) q[3];
cx q[4],q[5];
u3(0,0,pi/16) q[5];
cx q[5],q[3];
u3(0,0,-pi/16) q[3];
cx q[5],q[3];
u3(0,1.4065829705916304,-1.210233429742268) q[3];
cx q[5],q[10];
u3(0,0,-pi/16) q[10];
cx q[10],q[3];
u3(0,0,pi/16) q[3];
cx q[10],q[3];
u3(0,1.4065829705916304,-1.6029325114409922) q[3];
cx q[4],q[10];
u3(0,0,pi/16) q[10];
cx q[10],q[3];
u3(0,0,-pi/16) q[3];
cx q[10],q[3];
u3(0,1.4065829705916304,-1.210233429742268) q[3];
cx q[5],q[10];
cx q[5],q[12];
u3(0,0,-pi/16) q[10];
cx q[10],q[3];
u3(0,0,pi/16) q[3];
cx q[10],q[3];
u3(0,1.4065829705916304,-1.6029325114409922) q[3];
cx q[4],q[10];
u3(1.2122633849267066,0.9374811158739473,-0.9374811158739473) q[4];
cx q[4],q[6];
cx q[6],q[4];
cx q[6],q[2];
u3(0,0,3.9737082416354177) q[2];
cx q[6],q[2];
u3(0.5348669365045635,-0.710612390630506,-1.9872569556583082) q[2];
u3(0,0,pi/16) q[10];
cx q[10],q[3];
u3(0,0,-pi/16) q[3];
cx q[10],q[3];
u3(pi/2,2.4221355538925593,-15*pi/16) q[3];
u3(0,1.4065829705916304,-1.4065829705916302) q[10];
u3(-0.16395602762682351,0,0) q[12];
cx q[5],q[12];
u3(0.16395602762682351,-7*pi/16,0) q[12];
u3(pi/2,0,pi/2) q[15];
cx q[7],q[15];
u3(0,0,pi/4) q[7];
u3(pi/2,0,pi) q[15];
cx q[7],q[15];
u3(0,0,-pi/4) q[15];
cx q[7],q[15];
u3(0,0,-0.4575248307833286) q[7];
cx q[14],q[7];
u3(-1.5055670404858583,0,0) q[7];
u3(-1.5055670404858583,0,0) q[14];
cx q[14],q[7];
u3(0,0,0.44084289986952596) q[7];
cx q[1],q[7];
u3(-1.6493381732444496,0,-0.9709812184214198) q[7];
cx q[1],q[7];
u3(0,0,4.798446221456403) q[1];
u3(1.6493381732444494,0.20226498593777453,0) q[7];
u3(pi/2,-pi,-pi) q[14];
cx q[14],q[0];
u3(-0.27838799743787257,0,0) q[0];
cx q[14],q[0];
cx q[0],q[8];
u3(pi/2,0,pi) q[0];
cx q[8],q[0];
u3(0,0,-pi/4) q[0];
u3(0,0,2.927592455746394) q[14];
u3(0,-1.9167399219816188,-0.6211848071941821) q[15];
cx q[15],q[9];
u3(0,0,-2.9598624146063375) q[9];
cx q[15],q[9];
u3(pi/2,0,-0.18173023898345608) q[9];
cx q[11],q[9];
u3(0,0,3.381434967312744) q[9];
cx q[11],q[9];
u3(pi/2,-2.6378694320050085,-pi) q[9];
cx q[9],q[3];
u3(-2.806347091844961,0,-2.9252789453136514) q[3];
cx q[9],q[3];
u3(2.806347091844961,0.5031433914210925,0) q[3];
u3(0,1.4065829705916304,-1.4065829705916302) q[9];
u3(pi/2,pi/4,-pi) q[11];
cx q[11],q[10];
u3(0,0,-pi/4) q[10];
cx q[11],q[10];
cx q[5],q[11];
u3(0,-0.1642133562032666,-0.6211848071941821) q[10];
u3(0,0,4.122073001613786) q[11];
cx q[5],q[11];
u3(0,0,1.7299178371626753) q[5];
u3(0.10118901162470303,0,0) q[11];
cx q[14],q[9];
u3(0,0,-pi/4) q[9];
cx q[3],q[9];
u3(0,0,pi/4) q[9];
cx q[14],q[9];
u3(0,0,-pi/4) q[9];
cx q[3],q[9];
u3(0,1.4065829705916295,-0.6211848071941821) q[9];
u3(0,0,pi/4) q[14];
cx q[3],q[14];
u3(0,0,pi/4) q[3];
u3(0,0,-pi/4) q[14];
cx q[3],q[14];
u3(0,0,2.0494995167584675) q[3];
u3(0,0,0.872575086607533) q[14];
u3(pi/2,0,pi) q[15];
cx q[13],q[15];
u3(0,0,-pi/4) q[15];
cx q[13],q[15];
cx q[13],q[0];
u3(0,0,pi/4) q[0];
cx q[8],q[0];
u3(0,0,-pi/4) q[0];
u3(0,0,pi/4) q[8];
cx q[13],q[0];
u3(pi/2,0,-3*pi/4) q[0];
cx q[13],q[8];
u3(0,0,-pi/4) q[8];
u3(0,0,pi/4) q[13];
cx q[13],q[8];
cx q[0],q[8];
cx q[0],q[10];
u3(pi/2,-pi/2,-pi) q[8];
cx q[4],q[8];
u3(0,0,-pi/4) q[8];
u3(0,0,3.141573872245363) q[10];
cx q[9],q[10];
u3(0,0,-1.5707775454504667) q[10];
cx q[9],q[10];
u3(0,1.4065829705916304,-1.4065829705916302) q[9];
cx q[10],q[9];
u3(0,0,-pi/4) q[9];
u3(pi/2,0,pi) q[13];
cx q[12],q[13];
u3(0,0,-pi/16) q[13];
cx q[12],q[13];
cx q[12],q[7];
u3(0,0,-pi/16) q[7];
u3(0,1.4065829705916304,-1.210233429742268) q[13];
cx q[7],q[13];
u3(0,0,pi/16) q[13];
cx q[7],q[13];
cx q[12],q[7];
u3(0,0,pi/16) q[7];
u3(0,1.4065829705916304,-1.6029325114409922) q[13];
cx q[7],q[13];
u3(0,0,-pi/16) q[13];
cx q[7],q[13];
u3(0,1.4065829705916304,-1.210233429742268) q[13];
cx q[14],q[0];
u3(0,0,-0.872575086607533) q[0];
cx q[14],q[0];
u3(0,0,0.872575086607533) q[0];
u3(pi/2,0.3270104344838076,-3*pi/4) q[15];
cx q[1],q[15];
u3(-2.5035523252647764,0,-4.798446221456403) q[15];
cx q[1],q[15];
u3(pi/2,0,pi) q[1];
cx q[5],q[1];
u3(0,0,-1.7299178371626753) q[1];
cx q[5],q[1];
u3(0,0,1.7299178371626753) q[1];
u3(pi/2,0,pi) q[5];
cx q[14],q[5];
u3(0,0,0.956234974628748) q[5];
cx q[14],q[5];
u3(pi/2,0,pi) q[5];
cx q[5],q[2];
u3(0,0,1.3150468034357283) q[2];
cx q[5],q[2];
u3(2.5035523252647764,4.471435786972595,0) q[15];
cx q[7],q[15];
u3(0,0,-pi/16) q[15];
cx q[15],q[13];
u3(0,0,pi/16) q[13];
cx q[15],q[13];
cx q[12],q[15];
u3(0,1.4065829705916304,-1.6029325114409922) q[13];
u3(0,0,pi/16) q[15];
cx q[15],q[13];
u3(0,0,-pi/16) q[13];
cx q[15],q[13];
cx q[7],q[15];
u3(2.3776737572829236,1.1674361672628937,-1.162677452762422) q[7];
u3(0,1.4065829705916304,-1.210233429742268) q[13];
u3(0,0,-pi/16) q[15];
cx q[15],q[13];
u3(0,0,pi/16) q[13];
cx q[15],q[13];
cx q[12],q[15];
cx q[12],q[8];
u3(0,0,pi/4) q[8];
cx q[4],q[8];
u3(0,0,pi/4) q[4];
u3(0,0,-pi/4) q[8];
cx q[12],q[8];
u3(pi/2,0,-3*pi/4) q[8];
cx q[8],q[7];
u3(-0.0020804165189576783,0,0) q[7];
cx q[8],q[7];
u3(0,0,pi/4) q[7];
cx q[12],q[4];
u3(0,0,-pi/4) q[4];
u3(0,0,pi/4) q[12];
cx q[12],q[4];
u3(0,0,pi/16) q[4];
u3(0,0,0.7954652249927688) q[12];
cx q[6],q[12];
u3(0,0,-0.7954652249927688) q[12];
cx q[6],q[12];
u3(0,1.4065829705916304,-1.4065829705916302) q[6];
cx q[7],q[6];
u3(0,0,-pi/4) q[6];
cx q[7],q[6];
u3(0,1.4065829705916295,-0.6211848071941821) q[6];
u3(-pi/2,-pi/2,pi/2) q[12];
u3(0,1.4065829705916304,-1.6029325114409922) q[13];
u3(0,0,pi/16) q[15];
cx q[15],q[13];
u3(0,0,-pi/16) q[13];
cx q[15],q[13];
u3(pi/2,0,-15*pi/16) q[13];
cx q[3],q[13];
u3(0,0,-2.0494995167584675) q[13];
cx q[3],q[13];
u3(2.121544763230422,0,2.049499516758468) q[13];
cx q[3],q[13];
u3(-2.121544763230422,0,0) q[13];
cx q[3],q[13];
u3(0.3459893610034391,4.282617290542096,-4.282617290542096) q[3];
u3(pi/2,-pi/2,pi/2) q[13];
cx q[15],q[11];
u3(-0.10118901162470303,0,0) q[11];
cx q[15],q[11];
u3(pi/2,0,pi) q[11];
cx q[4],q[11];
u3(0,0,-pi/16) q[11];
cx q[4],q[11];
cx q[4],q[0];
u3(0,0,-pi/16) q[0];
u3(0,1.4065829705916304,-1.210233429742268) q[11];
cx q[0],q[11];
u3(0,0,pi/16) q[11];
cx q[0],q[11];
cx q[4],q[0];
u3(0,0,pi/16) q[0];
u3(0,1.4065829705916304,-1.6029325114409922) q[11];
cx q[0],q[11];
u3(0,0,-pi/16) q[11];
cx q[0],q[11];
cx q[0],q[1];
u3(0,0,-pi/16) q[1];
u3(0,1.4065829705916304,-1.210233429742268) q[11];
cx q[1],q[11];
u3(0,0,pi/16) q[11];
cx q[1],q[11];
cx q[4],q[1];
u3(0,0,pi/16) q[1];
u3(0,1.4065829705916304,-1.6029325114409922) q[11];
cx q[1],q[11];
u3(0,0,-pi/16) q[11];
cx q[1],q[11];
cx q[0],q[1];
u3(0,0,1.4313935099104265) q[0];
u3(0,0,-pi/16) q[1];
u3(0,1.4065829705916304,-1.210233429742268) q[11];
cx q[1],q[11];
u3(0,0,pi/16) q[11];
cx q[1],q[11];
cx q[4],q[1];
u3(0,0,pi/16) q[1];
u3(0,0,-pi/2) q[4];
u3(0,1.4065829705916304,-1.6029325114409922) q[11];
cx q[1],q[11];
u3(0,0,-pi/16) q[11];
cx q[1],q[11];
u3(pi/2,0,pi) q[1];
cx q[8],q[1];
u3(pi/2,0,pi) q[1];
u3(pi/2,0,-7*pi/16) q[11];
cx q[14],q[4];
u3(0,0,pi/2) q[4];
cx q[15],q[9];
u3(0,0,pi/4) q[9];
cx q[10],q[9];
u3(0,0,-pi/4) q[9];
u3(0,0,pi/4) q[10];
cx q[15],q[9];
u3(0,1.7268220858839243,-0.6211848071941821) q[9];
cx q[9],q[0];
u3(-1.3198174739260544,0,-3.497003261097799) q[0];
cx q[9],q[0];
u3(1.3198174739260544,2.0656097511873726,0) q[0];
cx q[15],q[10];
u3(0,0,-pi/4) q[10];
u3(0,0,pi/4) q[15];
cx q[15],q[10];
u3(0,0,4.208903601727303) q[10];
u3(1.4656526395244358,0.5043002661094951,-0.5043002661094951) q[15];