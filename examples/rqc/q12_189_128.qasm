OPENQASM 2.0;
qreg q[12];
u3(0,0,-pi/2) q[1];
u3(0,0,-3*pi/4) q[2];
cx q[3],q[1];
u3(0,0,-pi) q[1];
u3(pi/2,0,pi/4) q[3];
cx q[0],q[5];
cx q[5],q[0];
cx q[0],q[5];
u3(pi/2,-pi/2,pi/2) q[0];
u3(pi/2,-pi/2,pi/2) q[5];
u3(pi/2,pi/2,pi/2) q[6];
cx q[1],q[6];
u3(pi,2.272446263278007,-pi) q[1];
u3(0,0,pi/2) q[6];
u3(pi/2,-1.7342138823879605,pi/2) q[7];
u3(pi/2,-3.019323151636632,0) q[8];
u3(0.9741724902888634,-1.9399714358324656,-1.9521943774010868) q[9];
cx q[9],q[5];
cx q[5],q[6];
u3(-0.7940436412859986,0,0) q[6];
cx q[5],q[6];
u3(pi,0,pi) q[5];
u3(0.7767526855088978,0,-pi) q[6];
u3(pi,1.1621996524230047,-pi) q[9];
cx q[4],q[10];
cx q[10],q[4];
cx q[4],q[10];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[7],q[4];
u3(0,0,4.690434616757905) q[4];
cx q[7],q[4];
u3(pi/2,-pi/2,0) q[4];
u3(pi/2,0,-pi) q[7];
cx q[6],q[7];
cx q[7],q[6];
u3(0,0,pi/4) q[6];
u3(pi/2,pi/2,-pi) q[7];
u3(pi/2,-pi/2,pi/2) q[10];
cx q[10],q[0];
u3(0,0,0.12867348828950195) q[0];
cx q[10],q[0];
u3(pi,3*pi/4,-pi/2) q[0];
u3(pi/2,pi/2,-0.6478573499881359) q[10];
cx q[2],q[10];
u3(-0.9229389768067607,0,0) q[10];
cx q[2],q[10];
u3(0.926623298698562,3.1350035626713293,-3.1350035626713293) q[2];
cx q[10],q[4];
u3(pi/2,-2.116250564670759,-pi) q[4];
u3(2.7083623430149215,0,pi/4) q[10];
u3(1.0573853345234239,-1.2213428249222664,0) q[11];
cx q[8],q[11];
u3(0.4420866847177207,0,0) q[8];
u3(-0.4420866847177207,0,0) q[11];
cx q[8],q[11];
u3(pi/2,1.8269911811823132,-pi) q[8];
cx q[1],q[8];
u3(-1.0531796885127602,0,-2.272446263278007) q[8];
cx q[1],q[8];
u3(pi/2,0,pi) q[1];
cx q[0],q[1];
u3(0,0,-pi/4) q[1];
cx q[0],q[1];
u3(0,0,-pi/2) q[0];
u3(0,2.1445014578660784,-0.6211848071941817) q[1];
cx q[5],q[0];
u3(2.5756248593529874,3*pi/4,pi/2) q[0];
u3(1.0531796885127602,3.5870477356854873,0) q[8];
u3(0,0,-0.8754268778923322) q[11];
cx q[3],q[11];
u3(-1.1257333556515394,0,0) q[3];
u3(-1.1257333556515394,0,0) q[11];
cx q[3],q[11];
u3(pi/2,-2.795649037854681,-pi) q[3];
cx q[3],q[8];
u3(0,0,-0.34594361573511206) q[8];
cx q[3],q[8];
u3(0,0,2.799882043732687) q[3];
cx q[3],q[2];
u3(0,0,-2.799882043732687) q[2];
cx q[3],q[2];
u3(pi/2,0,-0.3417106098571061) q[2];
u3(0,0,1.3369479593753706) q[3];
cx q[3],q[5];
u3(0,0,-1.3369479593753706) q[5];
cx q[3],q[5];
u3(pi/2,0,0) q[3];
u3(0,0,2.122346122772819) q[5];
u3(0,1.4065829705916304,2.288966731031543) q[8];
cx q[4],q[8];
u3(0,0,-pi/4) q[8];
cx q[4],q[8];
u3(pi/2,1.6356262480734456,2.3458049434194237) q[4];
u3(0,2.418630272643073,-0.6211848071941821) q[8];
cx q[3],q[8];
u3(-1.987799832755812,0,0) q[3];
u3(-1.987799832755812,0,0) q[8];
cx q[3],q[8];
u3(pi/2,-pi,-pi) q[3];
u3(0,1.4065829705916304,-2.4186302726430733) q[8];
u3(0,0,2.0967697028145977) q[11];
cx q[9],q[11];
u3(0,0,-1.8392410619592459) q[11];
cx q[9],q[11];
u3(1.169703956326871,2.920299550968034,-3.063946146666524) q[9];
u3(0,1.4065829705916304,0.4326580913676157) q[11];
cx q[6],q[11];
u3(0,0,-pi/4) q[11];
cx q[6],q[11];
u3(2.308430958795416,0,-pi) q[6];
cx q[2],q[6];
u3(-2.308430958795416,0,0) q[6];
cx q[2],q[6];
u3(0,0,0.056440026314990364) q[2];
cx q[3],q[2];
u3(0,0,-0.056440026314990364) q[2];
cx q[3],q[2];
u3(0,0,0.9825311206900118) q[2];
u3(pi/2,0,-pi) q[3];
u3(0,0,1.0664336259735792) q[6];
cx q[5],q[6];
u3(0,0,-1.0664336259735792) q[6];
cx q[5],q[6];
u3(1.4021740562079412,-2.738801025920088,-0.7743839148425753) q[6];
u3(pi/2,0,-3*pi/4) q[11];
cx q[7],q[11];
u3(0,0,2.2678419564415866) q[11];
cx q[7],q[11];
u3(2.5841814006275423,-3*pi/4,-pi) q[7];
cx q[7],q[8];
u3(0,0,-pi/4) q[8];
cx q[7],q[8];
u3(0,0,0.48289512422396946) q[7];
cx q[5],q[7];
u3(0,0,-0.48289512422396946) q[7];
cx q[5],q[7];
u3(2.3481319043115114,0,0) q[5];
u3(0,0,pi/4) q[7];
u3(0,-1.5536365241748977,-0.6211848071941821) q[8];
cx q[8],q[2];
u3(-2.6363145216279174,0,-3.0021599513288457) q[2];
cx q[8],q[2];
u3(2.6363145216279174,2.019628830638834,0) q[2];
cx q[2],q[3];
u3(0,0,pi/4) q[2];
u3(pi/2,0,pi) q[3];
cx q[2],q[3];
u3(0,0,-pi/4) q[3];
cx q[2],q[3];
u3(0,1.4065829705916295,-0.6211848071941821) q[3];
cx q[5],q[3];
u3(0,0,1.6259903980998605) q[3];
cx q[5],q[3];
u3(pi/2,-0.50416886945639,0) q[5];
u3(0,1.4065829705916304,-1.4065829705916302) q[8];
cx q[7],q[8];
u3(0,0,-pi/4) q[8];
cx q[7],q[8];
u3(pi/2,0,pi) q[7];
u3(pi/2,-pi/2,3*pi/4) q[8];
u3(pi/2,0,pi) q[11];
cx q[11],q[10];
u3(-2.7083623430149215,0,0) q[10];
cx q[11],q[10];
u3(1.1283741356146717,-pi,-pi) q[10];
cx q[1],q[10];
u3(-0.8072212356590454,0,0) q[10];
cx q[1],q[10];
u3(1.2395944644605859,-1.8767108731680224,-1.6407909328438193) q[1];
u3(0.9486619075336432,0,0) q[10];
u3(pi/2,0,pi) q[11];
cx q[9],q[11];
u3(0,0,0.4941062794397203) q[11];
cx q[9],q[11];
u3(0,1.4065829705916304,-1.4065829705916302) q[9];
cx q[9],q[10];
u3(-0.9486619075336432,0,0) q[10];
cx q[9],q[10];
u3(pi/2,-pi/2,pi/2) q[9];
cx q[8],q[9];
u3(0,0,1.9875785862448265) q[9];
cx q[8],q[9];
u3(pi/2,1.8948650807957819,-pi/2) q[8];
cx q[3],q[8];
u3(0,0,-0.3240687540008853) q[8];
cx q[3],q[8];
u3(0,0,-0.6704442895622789) q[3];
cx q[5],q[3];
u3(-2.0255774809863123,0,0) q[3];
u3(-2.0255774809863123,0,0) q[5];
cx q[5],q[3];
u3(0,0,-2.810690645476458) q[3];
u3(pi,-0.3289523345111953,2.8126403190785982) q[5];
u3(0,0,-1.648551195598189) q[8];
u3(pi/2,1.33497342040247,-pi/2) q[9];
u3(pi/2,pi/2,pi/4) q[10];
u3(pi/2,0,pi) q[11];
cx q[0],q[11];
cx q[11],q[0];
cx q[0],q[11];
u3(pi/2,3.127960685067377,0.25396788377596735) q[0];
cx q[2],q[0];
u3(-0.12447811037504641,0,0) q[0];
cx q[2],q[0];
u3(0.12447811037504644,2.6174684893634232,0) q[0];
u3(pi/2,0,0) q[2];
cx q[2],q[9];
u3(3.045197542781365,0,0) q[2];
u3(-3.045197542781365,0,0) q[9];
cx q[2],q[9];
u3(pi/2,3*pi/4,-pi) q[2];
cx q[3],q[2];
u3(0,0,-2.802050372140849) q[2];
cx q[3],q[2];
u3(0,0,2.802050372140849) q[2];
u3(pi/2,0,pi) q[3];
u3(pi/2,pi/2,1.8066192331873232) q[9];
u3(0,0,1.9174519490211677) q[11];
cx q[4],q[11];
u3(-1.1712277097290635,0,-3.676495145476315) q[11];
cx q[4],q[11];
u3(pi/2,0,pi) q[4];
cx q[7],q[4];
u3(0,0,5.777255416756119) q[4];
cx q[7],q[4];
u3(pi/2,2.1310517265893747,-pi) q[4];
cx q[1],q[4];
u3(-0.9618365055987343,0,0) q[1];
u3(-0.9618365055987343,0,0) q[4];
cx q[1],q[4];
u3(pi/2,-pi,pi/2) q[1];
u3(pi/2,pi/4,2.581337253795315) q[4];
u3(pi/2,-1.6018653729542134,-pi) q[7];
cx q[10],q[1];
u3(0,0,0.09236138939657568) q[1];
cx q[10],q[1];
u3(pi/2,pi/2,-pi) q[1];
u3(pi/2,-pi/2,-pi) q[10];
u3(1.1712277097290635,1.7590431964551476,0) q[11];
cx q[6],q[11];
u3(0,0,-0.7396824165524685) q[11];
cx q[6],q[11];
u3(0,0,-0.5313886746706316) q[6];
cx q[7],q[6];
u3(-1.7150033533116429,0,-4.168022968637082) q[6];
cx q[7],q[6];
u3(1.5579536996711985,-1.715015329526315,-1.568931280573619) q[6];
cx q[6],q[8];
u3(2.065937274667662,0,0) q[6];
u3(pi/2,-pi/2,pi/2) q[7];
u3(-2.065937274667662,0,0) q[8];
cx q[6],q[8];
u3(pi/2,-pi,-pi) q[6];
u3(0,0,1.6485511955981895) q[8];
cx q[8],q[10];
u3(0,0,0.5383645940002058) q[8];
cx q[2],q[8];
u3(0,0,-0.5383645940002058) q[8];
cx q[2],q[8];
u3(0,0,3.766438428833687) q[2];
u3(pi,pi/2,pi/2) q[8];
u3(pi/2,0,pi) q[10];
u3(0,0,-2.634535332364427) q[11];
cx q[11],q[0];
u3(-2.108321058003221,0,-2.908967558262691) q[0];
cx q[11],q[0];
u3(2.108321058003221,5.003888049283958,0) q[0];
cx q[0],q[4];
u3(0,pi/2,-pi/4) q[4];
cx q[4],q[1];
u3(0,0,2.7905020874694992) q[1];
cx q[4],q[1];
u3(0,1.4065829705916304,-1.4065829705916302) q[1];
u3(0,0,pi/4) q[4];
cx q[6],q[0];
cx q[0],q[6];
u3(pi/2,-pi/2,-pi) q[0];
u3(pi/2,0,pi) q[6];
cx q[3],q[6];
u3(0,0,4.434968075881877) q[6];
cx q[3],q[6];
u3(1.2968810791626175,0,-pi) q[3];
u3(pi/2,0,pi) q[6];
cx q[9],q[6];
u3(0,0,3.6077441864224387) q[6];
cx q[9],q[6];
u3(pi/2,-pi/2,pi/2) q[11];
cx q[7],q[11];
u3(0,0,4.823713700209051) q[11];
cx q[7],q[11];
u3(pi/2,3*pi/4,0.02003530257681163) q[7];
u3(pi/2,-pi/2,-pi) q[11];
cx q[5],q[11];
u3(0,0,0.0883184797375118) q[11];
cx q[5],q[11];
u3(0,1.4065829705916304,-1.4065829705916302) q[5];
cx q[7],q[5];
u3(0,0,-pi/4) q[5];
cx q[7],q[5];
u3(pi/2,pi/4,-3*pi/4) q[5];
cx q[5],q[0];
u3(0,0,-pi/4) q[0];
cx q[5],q[0];
u3(pi/2,0,-3*pi/4) q[0];
u3(pi/2,-pi/2,pi/2) q[7];
u3(pi/2,0,pi) q[11];
cx q[11],q[10];
u3(0,0,0.832866683513691) q[10];
cx q[11],q[10];
u3(pi/2,0,pi) q[10];
cx q[4],q[10];
u3(0,0,-pi/4) q[10];
cx q[4],q[10];
u3(pi/2,0,-3*pi/4) q[10];
u3(0,0,pi/2) q[11];
cx q[11],q[7];
u3(pi,0,pi) q[11];