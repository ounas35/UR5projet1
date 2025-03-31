import numpy as np

import cv2 as cv
#from Transfo import *
import rtde_receive
import rtde_control
import time

import dashboard_client

"""
Connexion avec le robot UR5 sur son IP
"""
robot_r = rtde_receive.RTDEReceiveInterface("10.2.30.60")
robot = rtde_control.RTDEControlInterface("10.2.30.60")
dashboard =dashboard_client.DashboardClient("10.2.30.60")

"""
Determination des points de prise de vue
"""
joints = []

j1 = [-0.17486173311342412, -2.2085731665240687, 2.101492404937744, -1.5007994810687464, -1.5932686964618128, -1.798398796712057]
j2=[-0.24864751497377569, -2.2884085814105433, 2.201618194580078, -1.5824602285968226, -1.5330846945392054, -1.8782876173602503]
j3=[0.4223828911781311, -2.0259526411639612, 2.1541404724121094, -1.7057684103595179, -1.8133609930621546, -1.2221172491656702]
j3=[0.4141250252723694, -1.9756386915790003, 2.074881076812744, -1.6026318709002894, -1.8039997259723108, -0.9094918409930628]
j4=[0.00659319618716836, -1.914450470601217, 2.003910541534424, -1.498591725026266, -1.5935800711261194, -1.6362980047809046]
j5=[-0.28595477739443, -1.727450195943014, 1.7902007102966309, -1.2560408751117151, -1.3998363653766077, -1.822967831288473]
j6=[-0.1460188070880335, -1.9548547903644007, 1.9519643783569336, -1.431725804005758, -1.5722816626178187, -1.6573832670794886]
j7=[-0.2952335516559046, -2.4373095671283167, 2.2029104232788086, -1.4911873976336878, -1.5048535505877894, -1.8041160742389124]
j8=[-0.6732013861285608, -2.463039223347799, 2.1890668869018555, -1.375216309224264, -1.4104450384723108, -2.3667007128344935]
j9=[0.2043018341064453, -1.7862237135516565, 1.902026653289795, -1.4070771376239222, -1.787839714680807, -1.4227798620807093]

j10=[-0.5544984976397913, -1.8652423063861292, 1.9463157653808594, -1.3317387739764612, -1.479638401662008, -2.216503922139303]
j11=[-0.13374358812441045, -2.243775192891256, 2.229771614074707, -1.6100958029376429, -1.5354307333575647, -1.7333901564227503]
j12=[0.22273693978786469, -1.896721665059225, 1.8545160293579102, -1.3607800642596644, -1.7427809874164026, -1.3604653517352503]

j13=[-0.48129493394960576, -1.92041522661318, 1.8601527214050293, -1.3440869490252894, -1.5477269331561487, -2.322958771382467]
j14=[0.18014949560165405, -1.8936908880816858, 1.880281925201416, -1.3642242590533655, -1.7654302755938929, -1.5313079992877405]

j15=[-0.1374118963824671, -2.094706360493795, 2.1196436882019043, -1.5771926085101526, -1.5799077192889612, -1.7054017225848597]

j16=[-0.4557588736163538, -2.1337626616107386, 1.9408950805664062, -1.3175180594073694, -1.4924610296832483, -2.027187172566549]
j17=[-0.10653192201723272, -1.7968714872943323, 1.7232956886291504, -1.2468722502337855, -1.4616907278644007, -1.560643498097555]
j18=[0.631935715675354, -2.2363107840167444, 2.1390280723571777, -1.5520642439471644, -1.899668041859762, -0.8921702543841761]
j19=[0.18577122688293457, -2.1607559362994593, 2.0893235206604004, -1.6309755484210413, -1.5855944792376917, -1.2954009214984339]
j20=[-0.1966317335711878, -2.212874237691061, 2.091620922088623, -1.5193999449359339, -1.6278918425189417, -1.915999714528219]
j21=[-0.1966317335711878, -2.219368282948629, 2.2780861854553223, -1.6992891470538538, -1.6279280821429651, -1.9160836378680628]
j22=[-0.6801789442645472, -2.219116512929098, 2.296225070953369, -1.5815003553973597, -1.40057880083193, -2.374845568333761]
j23=[-0.6801312605487269, -2.218086067830221, 2.129814624786377, -1.4161980787860315, -1.4006384054767054, -2.3747976461993616]
j24=[-0.3034575621234339, -2.2331350485431116, 2.151423931121826, -1.4905632177936, -1.3883417288409632, -1.992342774068014]
j25=[0.36487695574760437, -2.0011428038226526, 2.141636371612549, -1.6792619864093226, -1.8105714956866663, -1.331771198903219]
j26=[0.36485299468040466, -2.010965649281637, 1.9911541938781738, -1.5189202467547815, -1.8105958143817347, -1.3317354361163538]
j27=[-0.4055870215045374, -2.232093159352438, 2.082742214202881, -1.4512384573565882, -1.4985197226153772, -2.096870724354879]
j28=[-0.478189770375387, -2.3359907309161585, 2.110442638397217, -1.3695762793170374, -1.496591869984762, -2.169687573109762]
j29=[-0.5565727392779749, -2.3658884207354944, 1.9558773040771484, -1.1793106237994593, -1.4949515501605433, -2.248248879109518]
j30=[-0.36434632936586553, -2.1306961218463343, 1.8925962448120117, -1.3654120604144495, -1.499836270009176, -2.0555713812457483]





#
#
#
# pose1 = [-0.3693747992179975, -0.10808537487570291, 0.24733758836877148, 0.1865517846653007, -3.1108191584178204, 0.032606145310106874]
# joint1 = [0.014264972880482674, -1.8673866430865687, 1.8654184341430664, -1.6010602156268519, -1.5998895804034632, -1.6093743483172815]
#
# pose2=[-0.3467331780438481, -0.11007433063137818, 0.20988309506622502, -0.011984719691816876, -3.1298364178197553, 0.08822427536902772]
# joint2=[-0.0035241285907190445, -1.9140422979937952, 1.9866433143615723, -1.6564629713641565, -1.5661638418780726, -1.499573055897848]
#
# pose3=[-0.2567513803923196, -0.16150109221515727, 0.26692731633436256, -0.4718306155345673, -2.9194175747792106, -0.13978620003780928]
# joint3=[0.30882036685943604, -2.1050689856158655, 2.1082167625427246, -1.693960968648092, -1.7362802664386194, -0.9514759222613733]
#
#
# pose4 =  [-0.37220614145986786, -0.1743337333753965, 0.1274980300844485, 0.0017312452576492912, 3.0894742060951246, -0.04389274920147587]
# joint4 =  [0.1774287074804306, -1.7254264990436, 1.9431562423706055, -1.736138645802633, -1.5850680510150355, -1.326458756123678]
#
# pose5= [-0.3175467137038899, -0.14305551244340237, 0.1450213618072016, -0.018524463523447592, -3.014266994122155, -0.13874696697874947]
# joint5 = [0.24910634756088257, -1.9690383116351526, 2.114905834197998, -1.8035004774676722, -1.7415717283831995, -1.2470133940326136]
#
#
# pose6 =  [-0.35146537101214326, -0.14303508126963013, 0.11925308124845163, -0.019511318332709953, -3.0033541083036575, -0.13807165302182128]
# joint6 =  [0.22707560658454895, -1.8791011015521448, 2.131035327911377, -1.9240229765521448, -1.7420027891742151, -1.2693188826190394]
#
# pose7=[-0.3479271519420039, 0.0042004941193957215, 0.2062395110656564, -0.1451521738969473, 3.0622991131039794, -0.378186823467146]
# joint7=[-0.4487093130694788, -1.8964341322528284, 1.9620757102966309, -1.5013397375689905, -1.4233644644366663, -2.0582311789142054]
#
# pose8=[-0.3250855199399336, 0.05902285268894494, 0.20145736105205353, 0.19454353046383846, -3.038802175166908, 0.5340872805644326]
# joint8=[-0.7258694807635706, -1.9555853048907679, 2.0183653831481934, -1.4598425070392054, -1.3236454168902796, -2.37708551088442]
#
# pose9=[-0.32510028782306727, -0.1367813050532006, 0.20148521322389581, 0.19591193519509462, -3.035094090818599, -0.2598322318695324]
# joint9=[0.25264206528663635, -1.9276145140277308, 2.0444517135620117, -1.7405903975116175, -1.8063939253436487, -1.3848426977740687]
#
# pose10=[-0.4098477476077388, -0.12347517324922626, 0.19976105457428842, -0.18597693028818046, 3.114543823020568, -0.022655747581451297]
# joint10=[0.05488887429237366, -1.7198808828936976, 1.8633718490600586, -1.6989052931415003, -1.607802693043844, -1.5686996618853968]
#
#
# pose11=[-0.28855685798026026, -0.11529328672767158, 0.17671947365672155, 0.2266863633069437, -2.7938350133818344, -0.22725595937710757]
# joint11=[0.2312588393688202, -2.186251942311422, 2.3277041912078857, -2.019461456929342, -1.827270809804098, -1.4666503111468714]
#
# pose12=[-0.28854275793907036, -0.1259354124191257, 0.17043332448827256, -0.18484283235365292, -2.821782784893667, -0.1196764346324312]
# joint12=[0.2362573891878128, -2.1505959669696253, 2.305713176727295, -1.9857857863055628, -1.790868107472555, -1.1577041784869593]
#
# pose13=[-0.3898718689380851, -0.1258994961429316, 0.17042114028489116, 0.16464126935407186, 3.0194989258367153, 0.16528621809812494]
# joint13=[0.12710918486118317, -1.665325943623678, 1.87308931350708, -1.6374667326556605, -1.7082799116717737, -1.2612360159503382]
#
# pose14=[-0.43535307374864546, -0.12493795323996147, 0.22453143793357022, 1.6245224548833666, 2.331106513069722, -0.2623589543474474]
# joint14=[-0.12017327943910772, -1.6038501898394983, 1.6834259033203125, -1.4305499235736292, -1.3028457800494593, -0.44566518465151006]
#
# pose15=[-0.4359579781698023, -0.2352782856245605, 0.20152414990300022, 0.14729641184676365, 2.732542402860148, 0.48187144025910134]
# joint15=[0.3749687969684601, -1.244471851979391, 1.31480073928833, -1.1381977240191858, -1.7727564016925257, -0.9790523687945765]
#
# pose16=[-0.2740511735371339, -0.02417641433611851, 0.22520509468856695, -0.11017844294555611, -2.9636003558806046, 0.24399846131953776]
# joint16=[-0.5062316099749964, -2.287833038960592, 2.1513524055480957, -1.5338118712054651, -1.3958609739886683, -1.9228785673724573]
#
#
# pose17=[-0.41486799813173797, 0.03124840380923536, 0.225167813902907, 0.04637676212814677, 2.992583045952265, -0.43149995069974983]
# joint17=[-0.4467433134662073, -1.678272549306051, 1.7387595176696777, -1.4259055296527308, -1.4078105131732386, -1.9394515196429651]
#
# pose18=[-0.2776614108640607, 0.031271096004439156, 0.2061726631402373, 0.019934462319210096, -2.923889649582998, 0.42462807317090945]
# joint18=[-0.8326094786273401, -2.239030186329977, 2.166069507598877, -1.448658291493551, -1.2713878790484827, -2.3524118105517786]
#
#
# pose19=[-0.30912761964120955, -0.0631414590314656, 0.16527527192561198, 0.004900052106876251, -2.939061986103916, 0.4346626967807372]
# joint19=[-0.4632518927203577, -2.205169979725973, 2.1838974952697754, -1.5987561384784144, -1.2773750464068812, -1.9585164229022425]
# pose20 =  [-0.309131658844317, -0.09216468799963544, 0.09994256135845955, 0.02747729855425485, -2.840104322403402, 0.41039615435448484]
# joint20 =  [-0.3871844450580042, -2.281614128743307, 2.4204068183898926, -1.8805306593524378, -1.2506950537311, -1.8753631750689905]
# pose21 =  [-0.33045727932447694, -0.14863630920639148, 0.09644386552078424, -0.010525850341736543, -2.8427243140864475, 0.24766706138557507]
# joint21 =  [0.017381245270371437, -2.114378277455465, 2.3886287212371826, -2.139974896107809, -1.4584820906268519, -1.4548528830157679]
# pose22 =  [-0.3337438161071934, -0.20726130067155749, 0.09308669609564371, -0.0331364698347538, -2.813494265855926, 0.03386054575032611]
# joint22 =  [0.38824883103370667, -1.9381559530841272, 2.3135502338409424, -2.2408998648272913, -1.7235196272479456, -1.106408421193258]
# pose23 =  [-0.24886414615741068, -0.20103861376568005, 0.1102997847207216, -0.05623001628944253, -2.6504729656410926, 0.15830160899851897]
# joint23 =  [0.5003215670585632, -2.2960646788226526, 2.505129337310791, -2.2394607702838343, -1.7526691595660608, -0.9910591284381312]
# pose24 =  [-0.29648280412050887, -0.1922849365481564, 0.1177316943354639, -0.029580004789440135, -2.8181423642134757, -0.05983735597886169]
# joint24 =  [0.44985297322273254, -2.0415499846087855, 2.294382095336914, -2.076648537312643, -1.7974398771869105, -1.053802792225973]
# pose25 =  [-0.3722868286963586, -0.1982560432557662, 0.15685109690264598, -0.0034910816693126496, -3.09721317120002, -0.07392753915380994]
# joint25 =  [0.30261141061782837, -1.741511646901266, 1.899871826171875, -1.7435181776629847, -1.6795008818255823, -1.1989358107196253]
# pose26 =  [-0.37230182153416264, -0.19826009951076934, 0.14424338898583472, -0.00347397393891796, -3.0971654887052202, -0.07396197592568479]
# joint26 =  [0.30263540148735046, -1.7360017935382288, 1.9280309677124023, -1.7772234121905726, -1.6795371214496058, -1.1989238897906702]
# pose27 =  [-0.4106253236861958, -0.2013515296236587, 0.12670407052528895, -0.0009625821922168667, 2.897325366084999, 0.1218038467369144]
# joint27 =  [0.25846752524375916, -1.4554336706744593, 1.6562309265136719, -1.5058038870440882, -1.6404021422015589, -1.2433798948871058]
# pose28 =  [-0.4106416782688052, -0.20134671111359825, 0.18203942986568708, -0.00098195076904492, 2.897332734840863, 0.12181174847179518]
# joint28 =  [0.2584555447101593, -1.4723666349994105, 1.532740592956543, -1.3653882185565394, -1.640414063130514, -1.243403736745016]
# pose29 =  [-0.4152709449128713, -0.1266588932417652, 0.18041339020318592, 0.027739983397556626, 2.913035629772895, -0.12655103575935298]
# joint29 =  [0.012442996725440025, -1.5766046682940882, 1.647923469543457, -1.41937762895693, -1.5319231192218226, -1.4827044645892542]
# pose30 =  [-0.4235089266898213, -0.07987097717472405, 0.18046354199358886, 0.027916263599283155, 2.913171521713978, -0.12669652319793764]
# joint30 =  [-0.08430654207338506, -1.578137222920553, 1.6487016677856445, -1.4161017576800745, -1.5533173719989222, -1.5769594351397913]
# pose31 =  [-0.34706284762902234, -0.07985024761067873, 0.1804486920157729, 0.015314761928864362, 3.125715881062331, -0.13722916656324882]
# joint31 =  [-0.11832744279970342, -1.9248712698565882, 1.9603886604309082, -1.5919883886920374, -1.536173168812887, -1.6130788961993616]
# pose32 =  [-0.34704514064974956, -0.07996706295559106, 0.14531764522722596, -0.014583223087995645, -3.0990622320225407, 0.4828189123539736]
# joint32 =  [-0.3091519514666956, -1.9491527716266077, 2.076017379760742, -1.6263917128192347, -1.3231547514544886, -1.812169377003805]
# pose33 =  [-0.3698569851562802, -0.14664124165066586, 0.14532464494225297, -0.014578468458324282, -3.1277455543447634, 0.2396247903852333]
# joint33 =  [0.01899949088692665, -1.8377278486834925, 2.001612663269043, -1.744298283253805, -1.4697969595538538, -1.474959675465719]
# pose34 =  [-0.3698166073384225, -0.18129421008922098, 0.12794581339849329, -0.013684969137290724, -3.135998536149062, -0.03647870410120736]
# joint34 =  [0.2398889809846878, -1.7412603537188929, 1.9662761688232422, -1.7853220144854944, -1.645466152821676, -1.2549641768084925]
# pose35 =  [-0.4003895674537122, -0.1047748705474016, 0.1376879463523026, -0.05894851691139269, 2.9008006646076505, -0.03381969092269151]
# joint35 =  [0.009385894984006882, -1.5908802191363733, 1.770632266998291, -1.5147917906390589, -1.6020081678973597, -1.5378173033343714]
# pose36 =  [-0.3698475937018654, -0.18727838913390762, 0.12794090954196388, -0.011407871508576991, -3.125625740474588, -0.1329009753930123]
# joint36 =  [0.2975653111934662, -1.7148626486407679, 1.9456219673156738, -1.7758315245257776, -1.7059934774981897, -1.197244946156637]
# pose37 =  [-0.36314354042071595, -0.1872995698883134, 0.21753270789746354, -0.011758278956167784, -3.1278117220931185, -0.06339010734282924]
# joint37 =  [0.2719516456127167, -1.7705209890948694, 1.7606263160705566, -1.550852123891012, -1.6638310591327112, -1.223856274281637]
# pose38 =  [-0.36311102822127506, -0.18585189962543358, 0.16600426912280733, -0.009629213549609128, -3.104502586710427, -0.3815592316751419]
# joint38 =  [0.40047386288642883, -1.6814587751971644, 1.8245124816894531, -1.6098316351519983, -1.8492243925677698, -1.0827801863299769]
# pose39 =  [-0.36304336993697445, -0.24948793032959277, 0.14500493450500088, -0.009606828570132677, -3.1047534821892886, -0.3819046114860037]
# joint39 =  [0.5253697037696838, -1.5729039351092737, 1.7685737609863281, -1.6276152769671839, -1.8341181913958948, -0.954078499470846]
# pose40 =  [-0.40815292147570104, -0.1742916592205939, 0.14500657883682586, -0.0061446395853189595, 2.9051103214327947, 0.13538756626742315]
# joint40 =  [0.21499362587928772, -1.4989646116839808, 1.6585173606872559, -1.4740870634662073, -1.6627891699420374, -1.287666145955221]

joints.append(j1)
joints.append(j2)
joints.append(j3)#
joints.append(j4)
joints.append(j5)
joints.append(j6)#
joints.append(j7)#
joints.append(j8)#
joints.append(j9)
joints.append(j10)
joints.append(j11)
joints.append(j12)
joints.append(j13)
joints.append(j14)
joints.append(j15)
joints.append(j16)
joints.append(j17)
joints.append(j18)#
joints.append(j19)
joints.append(j20)
joints.append(j21)
joints.append(j22)#
joints.append(j23)#
joints.append(j24)#
joints.append(j25)#
joints.append(j26)
joints.append(j27)
joints.append(j28)#
joints.append(j29)
joints.append(j30)




# joints.append(joint31)
# joints.append(joint32)
# joints.append(joint33)
# joints.append(joint34)
# joints.append(joint35)
# joints.append(joint36)
# joints.append(joint37)
# joints.append(joint38)
# joints.append(joint39)
# joints.append(joint40)


# Bonnes pos
# joints.append(joint1)
# joints.append(joint2)
# joints.append(joint3)
# joints.append(joint4)
# joints.append(joint5)
# joints.append(joint6)
# joints.append(joint7)
# # joints.append(joint8)
# # joints.append(joint9)
# # joints.append(joint10)
# joints.append(joint11)
# joints.append(joint12)
# joints.append(joint13)
# joints.append(joint14)
# joints.append(joint15)
# joints.append(joint16)
# joints.append(joint17)
# # joints.append(joint18)
# # joints.append(joint19)
# # joints.append(joint20)
# # joints.append(joint21)
# # joints.append(joint22)
# # joints.append(joint23)
# # joints.append(joint24)
# joints.append(joint25)
# joints.append(joint26)
# joints.append(joint27)
# joints.append(joint28)
# joints.append(joint29)
# joints.append(joint30)
# joints.append(joint31)
# joints.append(joint32)
# joints.append(joint33)
# joints.append(joint34)
# joints.append(joint35)
# joints.append(joint36)
# joints.append(joint37)
# # joints.append(joint38)S
# # joints.append(joint39)
# joints.append(joint40)

"""
CalibrateHandEye() pour determiner la transformation Camera dans le repere de l'outil
"""
poses = []


def priseDeVue(joints):
    i = 0
    rotationGripper2Base = []
    translationGripper2Base = []
    poses = []

    # for pose in poses:
    for joint in joints:
        camera = cv.VideoCapture(6)
        robot.moveJ(joint)
        pose = robot_r.getActualTCPPose()
        robot_r.getActualQ()
        poses.append(pose)
        print('*******image', i + 1, '...')

'''
        # Enregistrement de pose dans le dossier JointPositions
        pose_path = 'C:/Users/instalb/PycharmProjects/TP/Calibration/Cal2/JointPositions/JointPos_'
        pose_path_name = pose_path + str(i + 1).zfill(3)
        np.savez_compressed(pose_path_name, pose)

        # Calcul et enregistrement T_gripper2Base
        #T_gripper2base = create_matrice(pose)
        tcp_path = '/home/robot/Documgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray,  140 , 255, cv2.THRESH_BINARY)
cv2.imshow('Image Seuillée', mask)
cv2.waitKey(0)
elements, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

c = sorted(elements, key=cv2.contourArea)cuments/Projet/UR5projet1/images'
        time.sleep(3)
        # Capture de l'image
        camera = cv.VideoCapture(1,cv.CAP_DSHOW)
        time.sleep(0.5)
        ret, img = camera.read()
        # numeroté les images
        image_path_name = image_path+str(i+1).zfill(3) + ".png"
        i = i + 1
        # enregister l'image dans le chemin image_path_name
        print(image_path_name)
        cv.imwrite(image_path_name, img)
'''
    #return rotationGripper2Base, translationGripper2Base
#tab1, tab2 = priseDeVue(joints)


priseDeVue(joints)

