OPENQASM 2.0;
qreg q[10];
u3(0.48099322655194476,-1.0617991574289123,-0.18057602246943105) q[0];
u3(0.9159095418890253,1.0689219607433609,-1.8243205434220118) q[1];
u3(2.7297717642284067,-2.4686361806667074,-1.4176309991473428) q[2];
u3(1.8639866581936408,0.08308071821907159,0.07728871221893385) q[3];
u3(2.5712802809863704,2.9653196915595927,-0.03325346085827574) q[4];
u3(1.8387545640106238,0.008242197685119024,-2.3848520080272877) q[5];
cx q[0],q[5];
u3(0.8383535917175905,-pi/2,-pi/2) q[0];
u3(1.566228901570444,-3.0405898212031293,0.045036254909339046) q[5];
cx q[0],q[5];
u3(0.3528651711304614,-pi,pi/2) q[0];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[5];
cx q[0],q[5];
u3(1.2227062446189636,-2.410134149398729,1.4513213703973742) q[0];
u3(2.5423120803875667,-1.731638093439305,-1.0316787504200358) q[5];
u3(2.5944209564508944,1.518094405690186,-0.47717280821138663) q[6];
cx q[6],q[4];
u3(1.5678986391722916,-3.0405283446135467,0.02856622358121097) q[4];
u3(1.041516265381622,-pi/2,-pi/2) q[6];
cx q[6],q[4];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[4];
u3(0.764116174526431,-pi,pi/2) q[6];
cx q[6],q[4];
u3(1.3339435491635534,-2.006564336841742,-2.043490383371603) q[4];
cx q[4],q[5];
u3(0.6138004901015037,-pi/2,-pi/2) q[4];
u3(1.5434372673205055,-3.0442468247906938,0.2731263478435322) q[5];
cx q[4],q[5];
u3(0.01183578792457166,-pi,pi/2) q[4];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[5];
cx q[4],q[5];
u3(1.3976394915072676,-2.818933293052404,2.8891772772312656) q[4];
u3(2.3506680850722437,-1.2035805948303615,1.8293244235294353) q[5];
u3(1.2439160096259152,-0.5557416883363482,-2.566863559215202) q[6];
u3(1.984562705377863,-2.0374053764906366,-1.8313444119903306) q[7];
cx q[3],q[7];
u3(1.0410594309131862,-pi/2,-pi/2) q[3];
u3(1.5739504008478773,-3.040535995024473,-0.031094551322111474) q[7];
cx q[3],q[7];
u3(0.18131546448375185,0,-pi/2) q[3];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[7];
cx q[3],q[7];
u3(2.1854194862910377,0.06104367513416031,0.9328310509895585) q[3];
u3(1.7234454819213003,-2.6170733617561988,-0.48220729523660966) q[7];
u3(1.4904466301576529,0.2060995737836655,1.3033275267292783) q[8];
cx q[1],q[8];
u3(0.7471169232613926,-pi/2,-pi/2) q[1];
u3(1.5542418100627948,-3.0418468767417277,0.16392451810817787) q[8];
cx q[1],q[8];
u3(0.10943610400863452,0,-pi/2) q[1];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[8];
cx q[1],q[8];
u3(2.773834349934947,1.400585837682999,1.5952416307996806) q[1];
cx q[1],q[6];
u3(1.0028488452032454,-pi/2,-pi/2) q[1];
u3(1.5399389955773473,-3.045295546323537,0.30915588025540597) q[6];
cx q[1],q[6];
u3(0.36702998924715513,0,-pi/2) q[1];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[6];
cx q[1],q[6];
u3(1.5665344202324185,1.8040211419258325,2.6725764728738763) q[1];
u3(1.706150919657602,1.5184734239023996,-1.1246376984849071) q[6];
u3(1.8170603672790266,3.0072907747960365,1.5976850454032059) q[8];
cx q[7],q[8];
u3(1.026493444833743,-pi/2,-pi/2) q[7];
u3(1.5726789274212811,-3.0405044226679454,-0.018557720176203496) q[8];
cx q[7],q[8];
u3(0.5397564437501542,-pi,pi/2) q[7];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[8];
cx q[7],q[8];
u3(0.7785933177669063,-1.8293666499189125,2.512570706093541) q[7];
u3(0.616897791054598,-0.6432876968737711,1.104969351357937) q[8];
cx q[8],q[4];
u3(1.5761930199857928,-3.0406305950561396,-0.05322037896292642) q[4];
u3(0.7520742765425009,-pi/2,-pi/2) q[8];
cx q[8],q[4];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[4];
u3(0.4575781608869212,0,-pi/2) q[8];
cx q[8],q[4];
u3(2.008975183541903,2.68345022027263,-1.60398585409928) q[4];
u3(1.3306167276330025,-0.6389774369238168,2.4940472032324603) q[8];
cx q[8],q[4];
u3(1.5388808595322048,-3.045640094190313,0.32013518799562624) q[4];
u3(1.0285325823534186,-pi/2,-pi/2) q[8];
cx q[8],q[4];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[4];
u3(0.1303269390757016,-pi,pi/2) q[8];
cx q[8],q[4];
u3(2.058995536319582,-2.32039793185174,0.9928318587912335) q[4];
u3(0.5918681747429687,0.3708679466891853,3.1268324022092973) q[8];
u3(1.1358982640494064,1.2409749657862559,-0.19643603763137873) q[9];
cx q[9],q[2];
u3(1.5456285958346216,-3.0436591340151917,0.25074792065631835) q[2];
u3(0.720764968209273,-pi/2,-pi/2) q[9];
cx q[9],q[2];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[2];
u3(0.40238458830255003,-pi,pi/2) q[9];
cx q[9],q[2];
u3(2.2892124982323865,-1.6865117411500168,0.8627695927172709) q[2];
cx q[2],q[0];
u3(1.5571086521022197,-3.0414146257690495,0.135339448496695) q[0];
u3(0.9976910915481372,-pi/2,-pi/2) q[2];
cx q[2],q[0];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[0];
u3(0.1283393929094505,0,-pi/2) q[2];
cx q[2],q[0];
u3(1.3551224803808322,1.3272113877051694,0.9308164651619748) q[0];
cx q[0],q[5];
u3(1.09954191007685,-pi/2,-pi/2) q[0];
u3(1.3747374437107438,3.030195933057235,2.286466053232985) q[2];
u3(1.5738861284848933,-3.0405340162386842,-0.030460717781064517) q[5];
cx q[0],q[5];
u3(0.1492832620514186,0,-pi/2) q[0];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[5];
cx q[0],q[5];
u3(2.686420868182875,2.415465895971021,0.502083897957756) q[0];
u3(1.5941660381476093,2.402640971476931,-2.713489352898134) q[5];
cx q[6],q[2];
u3(1.5669872859520726,-3.0405584853843504,0.03755439759329615) q[2];
u3(1.2224402174290367,-pi/2,-pi/2) q[6];
cx q[6],q[2];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[2];
u3(0.3854798049517913,-pi,pi/2) q[6];
cx q[6],q[2];
u3(1.4830361160024652,2.6113620632206285,2.676048063368813) q[2];
cx q[2],q[0];
u3(1.5732561114994765,-3.04051677806374,-0.02424831219130663) q[0];
u3(0.555686177780982,-pi/2,-pi/2) q[2];
cx q[2],q[0];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[0];
u3(0.5118680184409995,-pi,pi/2) q[2];
cx q[2],q[0];
u3(1.1646012056369788,-1.542939493928419,-2.0411074743180038) q[0];
u3(2.805562737125995,-1.789776430031301,2.234764076435953) q[2];
u3(0.18587116608845625,-1.5537084299861488,1.2984880423650331) q[6];
cx q[8],q[2];
u3(1.5611178891013586,-3.0409496863068792,0.09554782590190669) q[2];
u3(0.8791786066569802,-pi/2,-pi/2) q[8];
cx q[8],q[2];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[2];
u3(0.08572550306830423,0,-pi/2) q[8];
cx q[8],q[2];
u3(1.3364402852918258,-2.908404332756253,-2.543801073379588) q[2];
u3(2.0107292329574604,2.378593640283274,-2.2056608256636734) q[8];
u3(2.198788248207056,0.6425717835705811,2.586628269477327) q[9];
cx q[9],q[3];
u3(1.5586110545906737,-3.041221439418,0.12040585040112273) q[3];
u3(1.0785271945577564,-pi/2,-pi/2) q[9];
cx q[9],q[3];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[3];
u3(0.3368010308383077,0,-pi/2) q[9];
cx q[9],q[3];
u3(0.519872968910199,2.410219970437014,-0.06181032398153974) q[3];
cx q[3],q[1];
u3(1.5524869163309503,-3.0421530551564233,0.18148914486049073) q[1];
u3(0.8232827590689493,-pi/2,-pi/2) q[3];
cx q[3],q[1];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[1];
u3(0.025227257151314516,-pi,pi/2) q[3];
cx q[3],q[1];
u3(0.9674214051812794,2.888788215387061,2.51806947877993) q[1];
cx q[1],q[6];
u3(0.9987940123081364,-pi/2,-pi/2) q[1];
u3(2.0833760506402017,-0.49278608197383633,-2.4383249445916677) q[3];
cx q[5],q[3];
u3(1.5486482935280002,-3.042934558269213,0.2201202467220007) q[3];
u3(0.7538713014892789,-pi/2,-pi/2) q[5];
cx q[5],q[3];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[3];
u3(0.10607880724874028,0,-pi/2) q[5];
cx q[5],q[3];
u3(1.5238327749422171,2.3275938420998816,-0.8941739254683165) q[3];
cx q[3],q[0];
u3(1.5309719905733428,-3.048635910869642,0.40362293639614766) q[0];
u3(0.9420192445489888,-pi/2,-pi/2) q[3];
cx q[3],q[0];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[0];
u3(0.1831516297983107,-pi,pi/2) q[3];
cx q[3],q[0];
u3(2.4878624511185796,2.213467665547709,-0.860838966879153) q[0];
cx q[0],q[8];
u3(1.1004513967740723,-pi/2,-pi/2) q[0];
u3(1.764765493496044,-2.9678565913360737,3.0702310319649477) q[3];
u3(1.2130631806104613,-0.9518895156554148,1.4265414080769672) q[5];
u3(1.5553999067287656,-3.041662165797434,0.15236199559761676) q[6];
cx q[1],q[6];
u3(0.35996034513273645,-pi,pi/2) q[1];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[6];
cx q[1],q[6];
u3(0.5643851498364811,-0.9576816396852865,0.9877475047815851) q[1];
u3(0.4868549107569036,0.3496542352851848,-1.4615766276987638) q[6];
cx q[4],q[6];
u3(0.8048268378673241,-pi/2,-pi/2) q[4];
u3(1.5402456187511293,-3.04519810187608,0.305981713544929) q[6];
cx q[4],q[6];
u3(0.23337324884191393,-pi,pi/2) q[4];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[6];
cx q[4],q[6];
u3(1.5391955908368269,-1.1198832163567305,-1.0736188316043265) q[4];
u3(0.3905284203049541,1.315620579652708,-1.4976304553617457) q[6];
u3(1.580658551120779,-3.040967470018945,-0.09736802933241329) q[8];
cx q[0],q[8];
u3(0.06745625502934645,-pi,pi/2) q[0];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[8];
cx q[0],q[8];
u3(0.402416613070901,0.35367619289504093,2.1450555068228123) q[0];
u3(1.4848115774445008,1.306107741253859,-0.9422769263317052) q[8];
u3(2.406323812088186,-1.6659244449131094,2.9496083164266294) q[9];
cx q[9],q[7];
u3(1.566828994306033,-3.040564556533237,0.03911583572718769) q[7];
u3(0.7463393947944357,-pi/2,-pi/2) q[9];
cx q[9],q[7];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[7];
u3(0.3149541454256088,0,-pi/2) q[9];
cx q[9],q[7];
u3(1.9327182943417447,-0.5805179539018495,1.6714865915061026) q[7];
u3(0.7387302081601959,2.536980810202053,0.357185148603155) q[9];
cx q[7],q[9];
u3(0.8613326616819612,-pi/2,-pi/2) q[7];
u3(1.55484322587492,-3.041749239352013,0.15791722517179885) q[9];
cx q[7],q[9];
u3(0.13565738188631762,0,-pi/2) q[7];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[9];
cx q[7],q[9];
u3(0.7705321924793278,0.43927364193140006,-1.4410435155805945) q[7];
cx q[5],q[7];
u3(0.8555866293457002,-pi/2,-pi/2) q[5];
u3(1.5720173051010145,-3.040494301329182,-0.012035375095766021) q[7];
cx q[5],q[7];
u3(0.2112112948204897,0,-pi/2) q[5];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[7];
cx q[5],q[7];
u3(0.8474985325007385,-3.0677288598707806,1.1317499412632293) q[5];
cx q[5],q[6];
u3(0.6992987754627626,-pi/2,-pi/2) q[5];
u3(1.5773862505096092,-3.040701212473129,-0.06500302135041824) q[6];
cx q[5],q[6];
u3(0.20167507416722388,-pi,pi/2) q[5];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[6];
cx q[5],q[6];
u3(2.683519597327535,1.7097378792118132,-0.786353000516395) q[5];
u3(2.427973683734452,1.6430560198990012,-2.911494248138008) q[6];
u3(1.4292450355667226,-0.8120720512793094,1.625098619716348) q[7];
cx q[2],q[7];
u3(0.924706999006448,-pi/2,-pi/2) q[2];
u3(1.570248154143044,-3.040488434752765,0.005403317701194954) q[7];
cx q[2],q[7];
u3(0.2736877338308481,0,-pi/2) q[2];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[7];
cx q[2],q[7];
u3(1.4429662869152027,-0.5923756974026508,1.7229942412840407) q[2];
u3(1.250171696143095,1.1323860159797485,0.6887987087640202) q[7];
cx q[7],q[0];
u3(1.5462678051660568,-3.0434975706614704,0.2442452410821394) q[0];
u3(1.1110890783484009,-pi/2,-pi/2) q[7];
cx q[7],q[0];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[0];
u3(0.482881735915554,-pi,pi/2) q[7];
cx q[7],q[0];
u3(1.7628947182428985,-0.021738807927066173,0.26726166209890856) q[0];
u3(0.9102409798121773,-2.9666004965055395,1.6822444167083077) q[7];
cx q[8],q[6];
u3(1.5742370671990233,-3.0405453173481876,-0.033921722386264896) q[6];
u3(0.451037566115195,-pi/2,-pi/2) q[8];
cx q[8],q[6];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[6];
u3(0.06054410597751377,-pi,pi/2) q[8];
cx q[8],q[6];
u3(0.9133796220898194,-0.27156700141486034,2.520512040474774) q[6];
u3(0.8582868808925591,-0.6596203415081274,-1.7792054259177559) q[8];
u3(0.732692979052442,-1.275729464626206,0.5029982087029952) q[9];
cx q[1],q[9];
u3(0.7878166376658976,-pi/2,-pi/2) q[1];
u3(1.5386274355653855,-3.045724528397465,0.32277073306396975) q[9];
cx q[1],q[9];
u3(0.48643236257181216,-pi,pi/2) q[1];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[9];
cx q[1],q[9];
u3(0.8251161259037638,1.9733329129972956,1.3405006054172262) q[1];
cx q[1],q[4];
u3(1.057342547639167,-pi/2,-pi/2) q[1];
u3(1.5800320257407034,-3.0409082316102864,-0.0911643060063363) q[4];
cx q[1],q[4];
u3(0.06685735646935306,-pi,pi/2) q[1];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[4];
cx q[1],q[4];
u3(0.819447136907095,-0.5953594569859444,2.938116517725705) q[1];
u3(1.942964128455565,1.2925189544917384,1.3294586922706984) q[4];
cx q[2],q[4];
u3(0.4110200609646541,-pi/2,-pi/2) q[2];
u3(1.5483644848152114,-3.0429985027703035,0.22298924317391222) q[4];
cx q[2],q[4];
u3(0.11266260084965049,0,-pi/2) q[2];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[4];
cx q[2],q[4];
u3(1.1898085353886763,-2.014300686276612,-1.7274814424503062) q[2];
cx q[2],q[6];
u3(0.751345178369846,-pi/2,-pi/2) q[2];
u3(2.2319673202368704,-3.1175254822816374,-1.288577003269122) q[4];
cx q[5],q[1];
u3(1.5803134879878973,-3.04093435886871,-0.09395082751940942) q[1];
u3(0.6446610089837066,-pi/2,-pi/2) q[5];
cx q[5],q[1];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[1];
u3(0.24825304992161062,-pi,pi/2) q[5];
cx q[5],q[1];
u3(2.9733702843299903,-2.4978514962784573,1.5217780758959707) q[1];
u3(0.8407376957025705,0.6274080926543393,-0.9979259181060769) q[5];
cx q[0],q[5];
u3(0.6770968310308193,-pi/2,-pi/2) q[0];
u3(1.5238029232718713,-3.0520388136803938,0.4819988653800875) q[5];
cx q[0],q[5];
u3(0.06827300791318504,-pi,pi/2) q[0];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[5];
cx q[0],q[5];
u3(1.231535464660452,0.9797356748173067,0.928034757549856) q[0];
u3(1.7937525715975018,2.419352170166901,3.0628156293499575) q[5];
u3(1.550940590637066,-3.042449298257333,0.19701404494583752) q[6];
cx q[2],q[6];
u3(0.22742312990114935,-pi,pi/2) q[2];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[6];
cx q[2],q[6];
u3(0.8208034566234687,0.0681678284811329,1.3312772042461898) q[2];
u3(1.007423648471434,1.54812071197472,2.645561513600212) q[6];
cx q[0],q[6];
u3(0.8561775280085685,-pi/2,-pi/2) q[0];
u3(1.552530283057662,-3.042145105411798,0.18105441682717416) q[6];
cx q[0],q[6];
u3(0.5396215180852485,-pi,pi/2) q[0];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[6];
cx q[0],q[6];
u3(1.235185323917993,0.23733019524198795,-3.076906702356479) q[0];
u3(1.173417359691909,1.3336909235972199,1.2835234376733107) q[6];
cx q[0],q[6];
u3(0.45277448876477727,-pi/2,-pi/2) q[0];
u3(1.522782026905529,-3.0525809563376036,0.49341563779796926) q[6];
cx q[0],q[6];
u3(0.19130132324293636,-pi,pi/2) q[0];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[6];
cx q[0],q[6];
u3(2.0032579719561934,1.279497376261677,-0.10317103296255592) q[0];
u3(0.7916394550886432,2.323820440358701,1.987215354516545) q[6];
cx q[7],q[4];
u3(1.5523289587320037,-3.042182175938991,0.18307288431448043) q[4];
u3(1.1615855086414402,-pi/2,-pi/2) q[7];
cx q[7],q[4];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[4];
u3(0.1532181930920131,-pi,pi/2) q[7];
cx q[7],q[4];
u3(1.218300890838211,-1.405723189786201,-0.40997792925638255) q[4];
cx q[4],q[2];
u3(1.5757095790614013,-3.0406059980060887,-0.048448848074342354) q[2];
u3(0.6250369598393634,-pi/2,-pi/2) q[4];
cx q[4],q[2];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[2];
u3(0.04201622974755175,-pi,pi/2) q[4];
cx q[4],q[2];
u3(1.7004197338739842,-0.4289234023142239,-0.7907995262088603) q[2];
u3(2.041086088530568,-0.5321426340561004,-1.1786866197149926) q[4];
u3(1.7607487832435107,-1.115815588020054,0.020931610505414255) q[7];
cx q[5],q[7];
u3(0.8733515601509805,-pi/2,-pi/2) q[5];
u3(1.5669465890935501,-3.0405600226407383,0.03795583500681943) q[7];
cx q[5],q[7];
u3(0.7649418586610809,-pi,pi/2) q[5];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[7];
cx q[5],q[7];
u3(0.6253840025712368,-1.213750225371982,-2.7169853933942107) q[5];
u3(1.8791786675032587,1.9157131722289824,1.0823378325384443) q[7];
cx q[5],q[7];
u3(0.5071212328027114,-pi/2,-pi/2) q[5];
u3(1.5464204982004999,-3.0434596308316593,0.2426934935201044) q[7];
cx q[5],q[7];
u3(0.21062697859580773,0,-pi/2) q[5];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[7];
cx q[5],q[7];
u3(1.6474135773148375,2.444223977455078,-0.08644155397565356) q[5];
u3(1.5119288400129962,-0.9298029514774919,1.8967263475609313) q[7];
u3(2.0456183715122083,-1.473727932931505,1.5054102862259473) q[9];
cx q[3],q[9];
u3(0.6809494800095086,-pi/2,-pi/2) q[3];
u3(1.5523358988360543,-3.042180891037578,0.18300329042796282) q[9];
cx q[3],q[9];
u3(0.1354975596845639,-pi,pi/2) q[3];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[9];
cx q[3],q[9];
u3(1.0721459481268,1.0901755948772927,-0.3972127087255757) q[3];
u3(0.7408117401551361,-0.11483282573791875,-1.0252961206031403) q[9];
cx q[3],q[9];
u3(0.5314525026221797,-pi/2,-pi/2) q[3];
u3(1.563627803867846,-3.040740538305747,0.07071966002034413) q[9];
cx q[3],q[9];
u3(0.27551084707092804,0,-pi/2) q[3];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[9];
cx q[3],q[9];
u3(1.0961015790563935,0.34329842122966703,0.3412670744694495) q[3];
cx q[8],q[3];
u3(1.5629227725967638,-3.0407929522809605,0.07768867813422853) q[3];
u3(0.5749345783615474,-pi/2,-pi/2) q[8];
cx q[8],q[3];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[3];
u3(0.41322080266231676,-pi,pi/2) q[8];
cx q[8],q[3];
u3(1.7856714054021348,0.8616096074411779,0.31048104820241385) q[3];
u3(1.3645019635693274,-0.5678635180128495,1.023480830830497) q[8];
u3(2.29115651539282,-1.7360534678600505,-1.1549984412605878) q[9];
cx q[9],q[1];
u3(1.5774080793046865,-3.040702635807146,-0.06521865220261347) q[1];
u3(0.8601207151378755,-pi/2,-pi/2) q[9];
cx q[9],q[1];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[1];
u3(0.16223283456511767,0,-pi/2) q[9];
cx q[9],q[1];
u3(0.5083089432906732,1.1052449363491297,-1.0389176506129347) q[1];
cx q[3],q[1];
u3(1.5760650073299969,-3.0406238567434123,-0.0519567863450634) q[1];
u3(0.5882100521104886,-pi/2,-pi/2) q[3];
cx q[3],q[1];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[1];
u3(0.023057705106500192,0,-pi/2) q[3];
cx q[3],q[1];
u3(2.722691354508126,2.5331896153375313,-2.167091725010567) q[1];
u3(1.7655329500611656,0.14125403500401124,1.5642179613594012) q[3];
cx q[1],q[3];
u3(0.7788383309943194,-pi/2,-pi/2) q[1];
u3(1.566797401204057,-3.0405657978949865,0.0394274914655206) q[3];
cx q[1],q[3];
u3(0.29070799958487764,0,-pi/2) q[1];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[3];
cx q[1],q[3];
u3(1.3513820822747644,1.4072562615249762,0.32355286889397084) q[1];
u3(0.5322206713389699,2.2743189674047954,-3.0604649733866065) q[3];
u3(1.8062772584827602,-0.3867286366034781,-0.11756234333204363) q[9];
cx q[9],q[8];
u3(1.5781889378309641,-3.040756662042634,-0.07293430520963673) q[8];
u3(0.9590795436070068,-pi/2,-pi/2) q[9];
cx q[9],q[8];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[8];
u3(0.43836174760784924,0,-pi/2) q[9];
cx q[9],q[8];
u3(1.7033687036148444,0.23865761804632113,-0.4120791185256021) q[8];
cx q[8],q[2];
u3(1.5199087364987773,-3.054188909996176,0.5259421355598959) q[2];
u3(0.5276698775701241,-pi/2,-pi/2) q[8];
cx q[8],q[2];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[2];
u3(0.25026707100305257,0,-pi/2) q[8];
cx q[8],q[2];
u3(2.0538454315722876,2.5348700744554877,2.966289719157243) q[2];
u3(2.392802669471375,0.21627896062932228,2.3998033590201766) q[8];
u3(2.5714554334914,-1.2018854323970358,-2.9734696638779887) q[9];
cx q[9],q[4];
u3(1.5791439645329346,-3.0408309770003914,-0.08237699768048223) q[4];
u3(1.0938662689621732,-pi/2,-pi/2) q[9];
cx q[9],q[4];
u3(1.6713839441184557,0.01023951313391791,-1.4691752037207917) q[4];
u3(0.42715973365870735,-pi,pi/2) q[9];
cx q[9],q[4];
u3(0.9165468581287178,-2.9749454989262993,1.1180440413196804) q[4];
u3(2.6379865182934203,-2.804209038677085,2.119740401743746) q[9];