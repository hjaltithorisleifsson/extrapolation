import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt 
from mpmath import *
import math
from extrapolation import *

mp.dps = 500

folder = '/Users/hjaltithorisleifsson/Documents/extrapolation/emr_plots/' #The folder to which results are written.

class ExplicitMidpointRule(Scheme):

	def __init__(self):
		super(ExplicitMidpointRule, self).__init__(2)

	def apply(self, ivp, n):
		h = (ivp.b - ivp.a) / (2 * n)
		y_sl = ivp.y0
		y_l = ivp.y0 + h * ivp.f(ivp.a, ivp.y0)

		for i in range(1, 2 * n):
			tmp = y_l
			y_l = y_sl + 2 * h * ivp.f(ivp.a + i * h, y_l)
			y_sl = tmp

		return y_l

	def get_evals(self, n):
		return 2 * n

class InitialValueProblem:
	def __init__(self, f, y0, a, b, ans, tex, ref):
		self.f = f
		self.y0 = y0
		self.a = a
		self.b = b
		self.ans = ans
		self.tex = tex
		self.ref = ref

oscillation_solution = np.array([0.847798681677116845, 0.56856899809517149])
#500 correct digits
oscillation_solution_hp = np.array([mpf('0.84779868167711684466367833188091892852048694878058392338945182372640630434688566338353630224868585976468140927461204156022529742699705231816982332875857346500673133786889916492046579446443336573454234448912717260825308361136810065270881200029802650122919460400481602696972357890233330486310651655280804379429035809638912058441220815487971711586081657146212645642189375079112255277643985217962293910119704130097616142891422432437391145846219393489820840131425517719029786828908788598595270867026923369'), 
	mpf('0.56856899809517148994173516302245889081638289029800038320616537052959766024841080148281818712844186765300851429899969930131242230797649624727126014481734539577929952609402984793251351341405040646241211439611813973862922778926381712605778151989795730885202122131996430203863213479810288598230734098332684874465028063198951013936721519329558610769737497582995347991161206624419371276898648988997691904623723371850615097727878192863338896660305324545705036191412764764458195896384935977161329263091278345')])

federpendel_solution_1 = np.array([0.98196696582217845, 0.49335546798350335, -0.05449716841744909, -0.02304200355186740])
#500 correct digits
federpendel_solution_1_hp = np.array([mpf('0.98196696582217844720319657954919759480867944130559325336717107885226840029194668537546599580718275493078481180650217303945645979298918706112255593505505626685102547536251828245902064758977315697810980003041775025607529367664638776665518639316652745962224476408233686057684574922030895683568627055060053379858290677222729974308155700102559653851623870181639231532304294556258406979390490261455173467397881497189867405108793688928536548109014102191774977544220410936638197206151636634436423913415562755'),
	mpf('0.49335546798350334626041620593851945416396626519837138591308850652625943297872531022435283639791643816215666656344985806930404094615344197083157552614944297128599991716468553830196779287602566748280602038872196712489924019695274403281608946318516254403134467824486021626022434745815153955369537221397337406711474039822233987947870868876535838241993296027192451258265454232143359143046007803845068321044214822314881129930911306463010391141587450151146417694046339447862830560090943990593856861344868184'),
	mpf('-0.054497168417449092396564692225277150158654249263273506069580872211139539308332921017183899865929552758056806081203650952356671774365740816189687626311834428013129538817979728576002102542805967179984145909860049511182076050552585325721312328821818561702429270968402730822217441482557867420055841500117550908728647830863352423646797486489760901115782149061190926028828423329291592505638956227897810714981510350185421442915816513117559281274595080602397863560840902157944618028123455200134949086739518845'),
	mpf('-0.023042003551867396194472265323709348808682640283852755901237351944819701571612757287153644623500483004940312003667781664764488432695366967333759943953427347462264741129870563019465004158420978604944153433044987024569241387972985977072936384006540779279179582791092121395142833676579702494986566477701894748765600871807915390298956166235028079010912883272530176653403789518263161024751885850800471261775340997299026434283495205453539990235152187062626116527819150160731687552555019773931010852710787326')])

federpendel_solution_2 = np.array([0.908485951398498847696, -0.040012744018780337043361193, -0.0578905193478659361183269820, -1.033663279605998919684942677430])
federpendel_solution_2_hp = np.array([mpf('0.90848595139849884769603653291496062503270523806688097650559051463679059621717406765127072562974902443098692105491434882819713510498544780275140566262920923919094308879461423242023249006528802304016469019659450867666853516641897160581599858863899725312812829812684512320657317082138495267978162519228332646143785995651440296148577391748961328189085871243444861978503160776609750334301861509139734505417881523161797903118442562808951135430345052299394995970839644880196049228869133677552355615201639841'), 
	mpf('-0.04001274401878033704336119311971319694982740419870372756532600713589357815311625888442455990290263338843920016918602606157039461358161414116875822338011746054046268046036596809856610785677948220161247049818070416285194976844006296124477109268266833636190723006029713149111709247362257659423681770012878721823547658511754828372204936485071190516706898628581042433198667327558306613787976959627739409422402942858732160505254216934224179224762870110338678881856667907336082354739548076217771507305280004'),
	mpf('-0.057890519347865936118326982012381058902284924568376618432084836703266355938858444709332291800676700169423933907719722038513059512636970593782361434831658097674960924145345347938203698716492781871838123431284351121896401807895747317005745314418159998831991872785307347966987045509215485283303825359947217525206610787068565140850708766935937312360126964825271205268059256423103439064362628012057942225535815942692762334818955055874353118056680076137468438477813164951871956636577739297079502516623843152'),
	mpf('-1.0336632796059989196849426774308533254847751473576804483689250158112186053792008897494219080916773325220468585407095643617208654861189091510798255322936786498098603887151833920930228566895164749459579768574320635663289125546215560356989982492005036765245846059130661617492068003636650284045924568537722452196933353861560059772798358938815297674151715603076136215737037500050392610451483325045547225290040950956466096120578479317086472139909045481432231784472145723227375922414201905077415299034086814')])

lorenz_solution = np.array([2.13310761864451495, 4.47142017718541514, 1.11389888577863626])
#500 correct digits
lorenz_solution_hp = np.array([mpf('2.13310761864451495227262406625812422912394267271452942441038842547981985850328349502607199061542736287510564231561063894454331488123113015328267043236490744150400848958149875972751266139537538153971268943254968106139160159633675489496074827335908449576202418683866646767507277804376121553135648265973548202625975421264392517334460924418900145357503551303938576244146468631416856578456386945664641869222037753129778663414326731021819612092621400259658133513494690907359150864648231418179355291471759314'),
	mpf('4.47142017718541514144768149714955948178729534192540257511574048784314325213911465974925412046344845530807597030429607398650598696865475884073021711207976388402679150036500814393325140313252781424291477191000786469984449210890844436736578796103819642579013770794361274064107752739700835289660400641502054132209544122685445472395259342563943159612601566551713128588384324984574209390318737462851994834113221314800051350964714090956173970566185175105419695088070765740270273191311935512029771689688429543'),
	mpf('1.11389888577863625731302964882355772343878431607277010723122303413525503530680783190158338774903098908487036356993642712781089106443515646105411494856828417574865381580126042193791912788712784979341374206818318926176708853579943311158336199047324403002590571863484482453954758060985395157599771991699936423160903705997793623893673731693689753344533055687765801242758209663300073936686571266960410814319888376333321997681838499112634281822402308285758402229675374550710803351539930624104657444820187144')])

lorenz_solution_02 = np.array([6.5425275558923681112676463800, 13.731186714070480190, 4.180197411970522114424545])
lorenz_solution_02_hp = np.array([mpf('6.542527555892368111267646380056566685378845926279379139276799665635421316775602906854332493629191349662252559499701449652158530304484442629618853169588624358417086336053658808574704985776603672270038412127036844816894683672128148782804481301836934764648129350476681569552715204271951312336070017296934671899079829740156908868754387634388628013182199474473871390892880560515499123304649817834122729210936663692723623744007585747603656511944101445982907371083871199385452452700485044162321513874439727'),
	mpf('13.731186714070480190374714284158841878594661610595311014469689966976896343663093836396509776621991038051032934348812977347635925174724221848484379339253863581590167495693816436416308289206971555435930516675387827399537475997651106549034781332158230394710600862732045511648678958208435238125082323000251989675436150952185873499040005119332913118555026745114821205022173383642749554643779775134222517073775971616436789506307773942016101601285406869561868000908825550106619337472683999868442735310930247'),
	mpf('4.1801974119705221144245450826488465324310770319261171975583143107979753221724745884037621935930278861777190487037628826799748162631355962391921895983484491046620054963551710955899267326161536257916806757114930837898595890021663616185509933482198645452430122343692020291628624750667278596518629281978764354821368536865928139426390330320669682519491942998235038729282304018512768544537018297200855640225897140156635015228342712441697781109673946849246057668282385528851111258164560565690957394559826368')])

lotka_volterra_solution = np.array([0.00052020049824246967594, 5.1128592857103855683])
#250 correct digits
lotka_volterra_solution_hp = np.array([mpf('0.0005202004982424696759392378709752313010367942418871686121466966557843043694571103661381951667543078110404250351898096339107497177677175206919476313184505318552911933137673213538990728459246508825834500848569054445828411282207192144300002422827753459597'),
	mpf('5.1128592857103855682878128981920703127866843287439215827496227954563301984740805602645514728451279432721520818380570855125934674913946008212074961279042175146117685226403685107224153778361804593518643453338027090457990569103700394621227781960576455143')])

def fp(t, y): 
	q = y[0:2]
	p = y[2:4]
	q_n = np.linalg.norm(q)
	qp = -(q_n - 1) * q / q_n - np.array([mpf('0'), mpf('1')])
	return np.array([p[0], p[1], qp[0], qp[1]])

def plot_basic():

	results_ivp_seq = []
	ivps = []

	ivp = InitialValueProblem(lambda t,y: y, 1.0, 0.0, 1.0, math.e, '$y\'=y$, $y(0) = 0$', 'exp_growth')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])


	ivp = InitialValueProblem(lambda t,y: y * (1 - y), 0.5, 0.0, 1.0, 1.0 / (1.0 + math.exp(-1.0)), '$y\' = y(1-y)$, $y(0) = 1/2$', 'logistic')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	ivp = InitialValueProblem(lambda t,y: 1.0 + y*y, 0.0, 0.0, 1.0, math.tan(1.0), '$y\' = 1 + y^2$, $y(0) = 0$', 'tangens')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	ivp = InitialValueProblem(lambda t,y: np.array([-y[1], y[0]]), np.array([1.0,0.0]), 0.0, math.pi / 2, np.array([0.0, 1.0]), '$(y_1,y_2)\' = (-y_2,y_1)$, $y(0) = (1,0)$', 'rotation') 
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	ivp = InitialValueProblem(lambda t,y: y * y, 0.5, 0.0, 1.0, 1.0, '$y\' = y^2$, $y(0) = 1/2$', 'singularity_0')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	a2 = 10.0 ** (-2)
	ivp = InitialValueProblem(lambda t,y: y * y, 1.0 / (1.0 + a2), 0.0, 1.0, 1 / a2, '$y\'=y^2$, $y(0) = 1/(1+10^{-2})$', 'singularity_2')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	a4 = 10.0 ** (-4)
	ivp = InitialValueProblem(lambda t,y: y * y, 1.0 / (1.0 + a4), 0.0, 1.0, 1 / a4, '$y\'=y^2$, $y(0) = 1/(1+10^{-4})$', 'singularity_4')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	ivp = InitialValueProblem(lambda t,y: -0.5 / y, math.sqrt(2.0), 0.0, 1.0, 1.0, '$y\' = -1/2y$, $y(0) = \sqrt{2}$', 'quad_sing_0')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	ivp = InitialValueProblem(lambda t,y: -0.5 / y, math.sqrt(1.0 + a2), 0.0, 1.0, math.sqrt(a2), '$y\' = -1/2y$, $y(0) = \sqrt{1+10^{-2}}$', 'quad_sing_2')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	ivp = InitialValueProblem(lambda t,y: -0.5 / y, math.sqrt(1.0 + a4), 0.0, 1.0, math.sqrt(a4), '$y\' = -1/2y$, $y(0) = \sqrt{1+10^{-4}}$', 'quad_sing_4')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	ivp = InitialValueProblem(lambda t,y: np.array([y[1], -math.sin(y[0])]), np.array([0.0, 1.0]), 0.0, 1.0, oscillation_solution, '$y\'\' + sin(y) = 0$', 'oscillation')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	#ivp = InitialValueProblem(lambda t,y: np.array([2.0/3.0*y[0] - 4.0/3.0*y[0]*y[1], y[0]*y[1] - y[1]]), np.array([10.0, 5.0]), 0.0, 1.0, lotka_volterra_solution, 'Lotkta Volterra', 'lotka_volterra')
	#result_rom = analyze(ivp, emr, romberg_short, False)
	#result_harmonic = analyze(ivp, emr, harmonic_short, False)
	#results_ivp_seq.append([result_rom, result_harmonic])
	
	ivp = InitialValueProblem(fp, np.array([1.0, 0.0, 0.0, 1.0]), 0.0, 1.0, federpendel_solution_1, 'Federpendel, estimate after 1 time unit', 'federpendel')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	ivp = InitialValueProblem(fp, np.array([1.0, 0.0, 0.0, 1.0]), 0.0, 2.0, federpendel_solution_2, 'Federpendel, estimate after 2 time units.', 'federpendel_2')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	ivp = InitialValueProblem(lambda t,y: np.array([10.0 * (y[1] - y[0]), y[0] * (28.0 - y[2]) - y[1], y[0] * y[1] - 8.0/3.0 * y[2]]), np.array([1.0, 1.0, 1.0]), 0.0, 0.1, lorenz_solution, 'Lorenz, estimate after 0.1 time steps.', 'lorenz')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	ivp = InitialValueProblem(lambda t,y: np.array([10.0 * (y[1] - y[0]), y[0] * (28.0 - y[2]) - y[1], y[0] * y[1] - 8.0/3.0 * y[2]]), np.array([1.0, 1.0, 1.0]), 0.0, 0.2, lorenz_solution_02, 'Lorenz, estimate after 0.2 time steps.', 'lorenz_02')
	ivps.append(ivp)
	result_rom = analyze(ivp, emr, romberg_short, False)
	result_harmonic = analyze(ivp, emr, harmonic_short, False)
	results_ivp_seq.append([result_rom, result_harmonic])

	results_ivp_seq_hp = []
	ivps_hp = []

	ivp_hp = InitialValueProblem(lambda t,y: y, mpf('1'), mpf('0'), mpf('1'), mp.e, '$y\'=y$, $y(0) = 0$', 'exp_growth_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(lambda t,y: mpf(y) * (mpf('1') - mpf(y)), mpf('0.5'), mpf('0'), mpf('1'), mpf('1') / (mpf('1') + mp.exp(mpf('-1'))), '$y\' = y(1-y)$', 'logistic_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(lambda t,y: mpf('1') + y*y, mpf('0'), mpf('0'), mpf('1'), mp.tan(1), '$y\' = 1 + y^2$, $y(0) = 0$', 'tangens_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(lambda t,y: np.array([-y[1], y[0]]), np.array([mpf('1'),mpf('0')]), mpf('0'), mp.pi / 2, np.array([mpf('0'), mpf('1')]), '$(y_1,y_2)\' = (-y_2,y_1)$, $y(0) = (1,0)$', 'rotation_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(lambda t,y: y * y, mpf('0.5'), mpf('0'), mpf('1'), mpf('1'), '$y\' = y^2$, $y(0) = 1/2$', 'singularity_0_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	b2 = mpf('0.01')
	ivp_hp = InitialValueProblem(lambda t,y: mpf(y) * mpf(y), 1.0 / (1.0 + b2), mpf('0'), mpf('1'), 1 / b2, '$y\'=y^2$, $y(0) = 1/(1+10^{-2})$', 'singularity_2_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	b4 = mpf('0.0001')
	ivp_hp = InitialValueProblem(lambda t,y: mpf(y) * mpf(y), 1.0 / (1.0 + b4), mpf('0'), mpf('1'), 1 / b4, '$y\'=y^2$, $y(0) = 1/(1+10^{-4})$', 'singularity_4_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(lambda t,y: mpf('-0.5') / y, mp.sqrt(2.0), mpf('0'), mpf('1'), mpf('1'), '$y\' = -1/2y$, $y(0) = \sqrt{2}$', 'quad_sing_0_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(lambda t,y: mpf('-0.5') / y, mp.sqrt(1.0 + b2), mpf('0'), mpf('1'), mp.sqrt(a2), '$y\' = -1/2y$, $y(0) = \sqrt{1+10^{-2}}$', 'quad_sing_2_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])	

	ivp_hp = InitialValueProblem(lambda t,y: mpf('-0.5') / y, mp.sqrt(1.0 + b4), mpf('0'), mpf('1'), mp.sqrt(a4), '$y\' = -1/2y$, $y(0) = \sqrt{1+10^{-4}}$', 'quad_sing_4_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(lambda t,y: np.array([y[1], -mp.sin(y[0])]), np.array([mpf('0'), mpf('1')]), mpf('0'), mpf('1'), oscillation_solution_hp, '$y\'\' + sin(y) = 0$', 'oscillation_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	#ivp_hp = HPInitialValueProblem(lambda t,y: np.array([mpf('2')/mpf('3')*y[0] - mpf('4')/mpf('3')*y[0]*y[1], y[0]*y[1] - y[1]]), np.array([mpf('10'), mpf('5')]), mpf('0'), mpf('1'), lotka_volterra_solution_hp, 'Lotkta Volterra', 'lotka_volterra_hp')
	#result_rom_hp = analyze(ivp, emr, romberg, True)
	#result_harmonic_hp = analyze(ivp, emr, harmonic, True)
	#results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(fp, np.array([mpf('1'), mpf('0'), mpf('0'), mpf('1')]), mpf('0'), mpf('1'), federpendel_solution_1_hp, 'Federpendel, estimate after 1 time unit.', 'federpendel_1_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(fp, np.array([mpf('1'), mpf('0'), mpf('0'), mpf('1')]), mpf('0'), mpf('2'), federpendel_solution_2_hp, 'Federpendel, estimate after 2 time units.', 'federpendel_2_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(lambda t,y: np.array([mpf('10') * (y[1] - y[0]), y[0] * (mpf('28') - y[2]) - y[1], y[0] * y[1] - mpf('8') / mpf('3') * y[2]]), np.array([mpf('1'), mpf('1'), mpf('1')]), mpf('0'), mpf('0.1'), lorenz_solution_hp, 'Lorenz, estimate after 0.1 time steps.', 'lorenz_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	ivp_hp = InitialValueProblem(lambda t,y: np.array([mpf('10') * (y[1] - y[0]), y[0] * (mpf('28') - y[2]) - y[1], y[0] * y[1] - mpf('8') / mpf('3') * y[2]]), np.array([mpf('1'), mpf('1'), mpf('1')]), mpf('0'), mpf('0.2'), lorenz_solution_02_hp, 'Lorenz, estimate after 0.2 time steps.', 'lorenz_02_hp')
	ivps_hp.append(ivp_hp)
	result_rom_hp = analyze(ivp_hp, emr, romberg, True)
	result_harmonic_hp = analyze(ivp_hp, emr, harmonic, True)
	results_ivp_seq_hp.append([result_rom_hp, result_harmonic_hp])

	for (results_seq, ivp) in zip(results_ivp_seq, ivps):
		plot_eval_error(results_seq, 'Equation: %s Double precision' % ivp.tex, ivp.ref, True, folder)

	for (results_seq, ivp) in zip(results_ivp_seq_hp, ivps_hp):
		plot_eval_error(results_seq, 'Equation: %s' % ivp.tex, ivp.ref, True, folder)
		plot_trend(results_seq, 'Equation: %s' % ivp.tex, ivp.ref, True, folder)

	for (results_seq, ivp) in zip(results_ivp_seq_hp, ivps_hp):
		plot_steps_error(results_seq, 'Equation: %s' % ivp.tex, ivp.ref, True, 30, folder)

	file1 = open(folder + 'all_results_evals_error.txt', 'w')
	file2 = open(folder + 'all_results_steps_error.txt', 'w')
	for results_seq in results_ivp_seq_hp:
		for result in results_seq:
			ln_e = result.ln_e
			p = opt.curve_fit(fit_func, result.evals, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
			q = opt.curve_fit(fit_func, np.array([i+1 for i in range(len(ln_e))]), ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
			file1.write('%s & %s & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) \\\\\n' % (result.prob_ref, result.seq_ref, p[0], p[1], p[2]))
			file2.write('%s & %s & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) \\\\\n' % (result.prob_ref, result.seq_ref, q[0], q[1], q[2]))

	file1.close()
	file2.close()

def plot_q_sing():
	param_ivp = lambda q: InitialValueProblem(lambda t,y: mpf(y) * mpf(y), 1.0 / (1.0 + q), mpf('0'), mpf('1'), 1 / q, '', '')
	ps = np.array([mpf('0.5') ** i for i in range(18)])
	title = 'IVP: $y\' = y^2$, $y(0) = 1/(1 + a)$'
	ref = "sing"
	plot_by_param(param_ivp, emr, ps, title, seqs, ref, folder)

def plot_q_quad_sing():
	param_ivp = lambda q: InitialValueProblem(lambda t,y: mpf('-0.5') / y, mp.sqrt(1.0 + q), mpf('0'), mpf('1'), mp.sqrt(q), '', '')
	ps = np.array([mpf('0.5') ** i for i in range(18)])
	title = 'IVP: $y\' = -1/2y$, $y(0) = \sqrt{1 + a}$'
	ref = "quad_sing"
	plot_by_param(param_ivp, emr, ps, title, seqs, ref, folder)

romberg = Sequence([2 ** i for i in range(13)], 'Romberg')
harmonic = Sequence([i + 1 for i in range(110)], 'Harmonic')

romberg_short = Sequence([2 ** i for i in range(10)], 'Romberg')
harmonic_short = Sequence([i + 1 for i in range(44)], 'Harmonic')

seqs = [romberg, harmonic]

emr = ExplicitMidpointRule()

def main():
	plot_basic()
	plot_q_sing()
	plot_q_quad_sing()

main()
