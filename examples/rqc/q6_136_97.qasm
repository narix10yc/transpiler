OPENQASM 2.0;
qreg q[6];
u3(pi/2,-pi/2,pi/2) q[0];
cx q[2],q[3];
u3(0,0,2.532162727857619) q[3];
cx q[2],q[3];
u3(0,1.4065829705916304,-1.4065829705916302) q[3];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[4],q[0];
u3(0,0,5.459869114373822) q[0];
cx q[4],q[0];
u3(pi/2,2.222608618462699,-pi/2) q[0];
cx q[0],q[2];
u3(0,0,-0.6518122916678021) q[2];
cx q[0],q[2];
u3(pi/2,pi/2,0) q[0];
u3(pi/2,0,-0.9189840351270946) q[2];
u3(0.774115457716062,2.26655236922842,-1.9682063607611004) q[4];
cx q[4],q[3];
u3(0,0,2.841584548853175) q[3];
cx q[4],q[3];
u3(pi/2,-pi/2,-pi) q[3];
cx q[4],q[3];
u3(0,0,pi/2) q[3];
u3(pi/2,0,pi) q[5];
cx q[1],q[5];
u3(0,0,pi/4) q[1];
u3(pi/2,0,pi) q[5];
cx q[1],q[5];
u3(0,0,-pi/4) q[5];
cx q[1],q[5];
u3(0,0,pi/2) q[1];
u3(pi/2,-pi/2,3*pi/4) q[5];
cx q[1],q[5];
u3(pi,pi/2,-pi) q[1];
cx q[2],q[1];
cx q[1],q[2];
u3(1.5499859682008654,0.8092169730454257,-1.5663800333836915) q[1];
u3(2.8695218894177152,pi/2,1.7769995334430284) q[2];
cx q[3],q[2];
u3(pi/2,0,pi) q[2];
u3(0,0,pi/4) q[3];
cx q[3],q[2];
u3(0,0,-pi/4) q[2];
cx q[3],q[2];
u3(0,-3.0065773153526427,-0.6211848071941812) q[2];
u3(0.8950531130030855,1.0927032035843873,-1.0927032035843873) q[3];
u3(0.9872660864212348,-1.4511028352993574,-pi/2) q[5];
cx q[0],q[5];
u3(2.85157115848391,0,0) q[0];
u3(-2.85157115848391,0,0) q[5];
cx q[0],q[5];
u3(pi/2,-pi,-pi) q[0];
cx q[4],q[0];
cx q[0],q[4];
u3(0,0,-2.8191316870497216) q[0];
cx q[2],q[0];
u3(-0.3110634458588334,0,-3.008782765880871) q[0];
cx q[2],q[0];
u3(0.3110634458588334,5.827914452930592,0) q[0];
cx q[2],q[3];
u3(0,1.4065829705916304,-1.4065829705916302) q[2];
cx q[1],q[2];
u3(0,0,-pi/4) q[2];
cx q[1],q[2];
u3(0,1.2883553242588413,-0.6211848071941821) q[2];
u3(pi/2,-pi/2,-0.1196934914955392) q[5];
cx q[4],q[5];
u3(pi/2,0,pi) q[4];
cx q[0],q[4];
u3(0,0,pi/2) q[0];
cx q[3],q[0];
u3(-2.3394036789038153,0,0) q[0];
cx q[3],q[0];
u3(pi/2,0.8021889746859783,-pi/2) q[0];
u3(pi/2,-pi/2,pi/2) q[3];
u3(pi/2,0,pi) q[4];
u3(pi/2,1.4304882269237966,-pi) q[5];
cx q[4],q[5];
u3(pi/2,-pi/2,pi/2) q[4];
cx q[3],q[4];
u3(0,0,5.10781602174039) q[4];
cx q[3],q[4];
u3(-pi/2,-pi/2,pi/2) q[3];
cx q[3],q[0];
u3(0,0,5.410437594530867) q[0];
cx q[3],q[0];
u3(pi/2,0.439718330587056,-pi) q[0];
u3(pi/2,0,pi) q[3];
u3(-pi/2,-pi/2,pi/2) q[4];
cx q[2],q[4];
cx q[4],q[2];
cx q[2],q[4];
u3(pi/2,pi/4,-pi/2) q[4];
cx q[2],q[4];
u3(pi/2,-pi,0) q[2];
u3(pi/2,-3.020833628888971,3*pi/4) q[4];
cx q[2],q[4];
u3(0.6145208511914358,0,0) q[2];
u3(-0.6145208511914358,0,0) q[4];
cx q[2],q[4];
u3(pi/2,1.8647956160723842,-pi) q[2];
u3(pi/2,0,3.020833628888971) q[4];
cx q[5],q[1];
u3(0,0,pi/4) q[1];
cx q[1],q[3];
u3(0,0,2.4707080721091637) q[3];
cx q[1],q[3];
u3(0,0,-2.962037918842838) q[1];
u3(0,2.977379297386526,-1.4065829705916302) q[3];
u3(0,0,0.3072785714637156) q[5];
cx q[0],q[5];
u3(-2.00265210160255,0,-3.074408342559436) q[5];
cx q[0],q[5];
u3(0,0,-0.9736562340606053) q[0];
cx q[1],q[0];
u3(-2.951268499358057,0,-2.5536692230718123) q[0];
cx q[1],q[0];
u3(1.499560402764628,2.9649528142751143,1.1913741323528662) q[0];
u3(pi/2,0,-pi/2) q[1];
cx q[1],q[3];
cx q[3],q[1];
u3(2.4826766915430047,-2.3302772018685625,-pi) q[1];
u3(pi,0,-pi) q[3];
cx q[3],q[2];
u3(-0.5520411931247018,0,0) q[2];
u3(-0.5520411931247018,0,0) q[3];
cx q[3],q[2];
u3(0,0,1.276797037517409) q[2];
u3(pi/2,-pi,-pi) q[3];
cx q[1],q[3];
u3(0,0,-0.8113154517212307) q[3];
cx q[1],q[3];
u3(0,0,pi/2) q[1];
u3(0,0,2.382111778516127) q[3];
u3(1.909422187473831,-2.681805945791762,1.7338318104014512) q[5];
cx q[0],q[5];
cx q[5],q[0];
u3(1.1975870802106567,-2.6773728570933364,-1.6607314668921804) q[0];
u3(pi/2,0,pi) q[5];
cx q[5],q[4];
u3(pi/2,0,-pi) q[4];
cx q[2],q[4];
u3(0,0,-pi/4) q[4];
cx q[5],q[4];
u3(0,0,pi/4) q[4];
cx q[2],q[4];
u3(0,0,pi/4) q[2];
u3(0,0,-pi/4) q[4];
cx q[5],q[4];
u3(pi/2,-pi/2,3*pi/4) q[4];
cx q[1],q[4];
u3(0.2496516279674793,-pi/2,pi/2) q[1];
u3(pi,-pi,-pi) q[4];
cx q[0],q[4];
u3(0,0,-pi/4) q[4];
cx q[0],q[4];
u3(pi/2,0,pi) q[0];
u3(pi/2,-pi/2,3*pi/4) q[4];
cx q[5],q[2];
u3(0,0,-pi/4) q[2];
u3(0,0,pi/4) q[5];
cx q[5],q[2];
u3(3*pi/4,-pi,-pi/2) q[2];
u3(pi/2,0,-pi/2) q[5];
cx q[3],q[5];
u3(pi/2,0,pi) q[3];
cx q[1],q[3];
u3(0,0,0.6015247564176018) q[3];
cx q[1],q[3];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[1],q[0];
u3(0,0,3.846007629725452) q[0];
cx q[1],q[0];
u3(pi/2,0.2314458699339399,-pi) q[0];
u3(pi/2,-pi/2,pi/2) q[1];
cx q[1],q[4];
u3(0.7544315604766015,-0.2711982030902216,0.36453421767917016) q[3];
u3(0,0,6.017175741851628) q[4];
cx q[1],q[4];
u3(-pi/2,-pi/2,pi/2) q[1];
u3(-pi/2,-pi/2,pi/2) q[4];
u3(1.2610871810482855,0.5649731918663736,-0.3884709979941614) q[5];
cx q[2],q[5];
cx q[5],q[2];
u3(pi/2,0,-pi/2) q[2];
u3(0,0,pi/2) q[5];
cx q[2],q[5];
cx q[5],q[2];
u3(0,1.4065829705916304,-1.4065829705916302) q[2];
cx q[1],q[2];
u3(0,0,-pi/4) q[2];
cx q[4],q[2];
u3(0,0,pi/4) q[2];
cx q[1],q[2];
u3(0,0,pi/4) q[1];
u3(0,0,-pi/4) q[2];
cx q[4],q[2];
u3(0,1.4065829705916295,-0.6211848071941821) q[2];
cx q[4],q[1];
u3(0,0,-pi/4) q[1];
u3(0,0,pi/4) q[4];
cx q[4],q[1];
u3(pi/2,pi/4,-pi) q[1];
cx q[2],q[1];
u3(pi/2,0,3*pi/4) q[1];
u3(pi/2,0,pi) q[4];
u3(pi/2,pi/2,-pi) q[5];
cx q[3],q[5];
cx q[5],q[3];
u3(0,0,pi/4) q[3];
cx q[3],q[4];
u3(0,0,-pi/4) q[4];
cx q[3],q[4];
u3(pi/2,pi/4,-pi/2) q[3];
u3(1.2368225254815477,2.313349884699525,0.5050997719091583) q[4];
u3(pi/2,0,pi) q[5];
cx q[5],q[1];
u3(0,0,pi/4) q[1];
cx q[0],q[1];
u3(0,0,-pi/4) q[1];
cx q[5],q[1];
u3(0,0,pi/4) q[1];
cx q[0],q[1];
u3(pi/2,0,pi) q[0];
u3(pi/2,pi/4,3*pi/4) q[1];
cx q[2],q[1];
u3(pi/2,pi/4,3*pi/4) q[1];
cx q[1],q[0];
u3(0,0,-pi/4) q[0];
cx q[1],q[0];
u3(pi/2,0,-3*pi/4) q[0];
cx q[2],q[3];
u3(pi/2,-pi/2,3*pi/4) q[3];
u3(0,0,-pi/4) q[5];