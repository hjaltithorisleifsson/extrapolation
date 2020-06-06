import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt 
import math
from mpmath import *
import os

mp.dps = 500

class Scheme: 
	#An abstract class that represents a numerical scheme which is assumed to have asymptotic
	#expansion in h^p

	def __init__(self, p):
		self.p = p

	#Applies the scheme to a problem, using parameter h/n, where h 
	#is a parameter specified by the problem.
	def apply(self, problem, n):
		pass

	#Returns the number of function evaluations that are used when
	#apply(self, problem, n) is called.
	def get_evals(self, n):
		pass

class Result: 
	#A holder for the results of the convergence analysis. 

	#prob_ref (str): A reference to problem which was being analyzed.
	#seq_ref (str): A reference to the sequence which was used.
	#evals (np.ndarray<int>): An array holding the number of function evaluations used.  
	#ln_e (np.ndarray<float>): An array holding ln of the errors
	#log_e (n.ndarray<float>): An array holding log of the errors.
	#ref (str): A unique reference to the results (used for cache)
	def __init__(self, prob_ref, seq_ref, evals, ln_e, log_e, ref):
		self.prob_ref = prob_ref
		self.seq_ref = seq_ref
		self.evals = evals
		self.ln_e = ln_e
		self.log_e = log_e
		self.ref = ref

	def write(self, folder):
		if not os.path.isdir(folder):
			os.mkdir(folder)

		my_folder = os.path.join(folder, self.ref)
		if not os.path.isdir(my_folder):
			os.mkdir(my_folder)

		evals_file = open(os.path.join(my_folder, 'evals'), 'wb')
		ln_e_file = open(os.path.join(my_folder, 'ln_e'), 'wb')

		self.evals.tofile(evals_file)
		self.ln_e.tofile(ln_e_file)

		evals_file.close()
		ln_e_file.close()

class Sequence: 
	#A class representing a sequence to use in extrapolation. 

	#seq (List<int>): The sequence to use.
	#ref (str): A refernce to the sequence.
	def __init__(self, seq, ref):
		self.seq = seq
		self.len = len(seq)
		self.ref = ref

#Returns the first n elements in the Bulirsch sequence.
def bulirsch_seq(n):
	bulirsch = [0 for i in range(n)]
	for i in range(3):
		bulirsch[i] = i + 1

	for i in range(3, n):
		bulirsch[i] = 2 * bulirsch[i - 2]

	bulirsch = np.array(bulirsch)
	return Sequence(bulirsch, 'Bulirsch')

#Returns the first n elements in Romberg sequence.
def romberg_seq(n):
	romberg = np.array([2 ** i for i in range(n)])
	return Sequence(romberg, 'Romberg')

#Returns the first n elements in identity sequence.
def harmonic_seq(n):
	harmonic = np.array([i + 1 for i in range(n)])
	return Sequence(harmonic, 'Harmonic')

#sc (Scheme): The scheme to extrapolate	
#prob: The problem to apply the scheme to. We assume that sch is an 
#      implementation of Scheme which can be applied to prob.
#seq (Sequence): The sequence to use in the extrapolation
#hp (bool): Indicates whether to use high precision arithmetic (true) 
#           or standard double precision (false).
#returns: The extrapolation table as a list of lists. 
def extrapolate(sc, prob, seq, hp):
	n = len(seq)
	X = [[0 for j in range(i + 1)] for i in range(n)]

	#X[i][j] = T_ij
	for i in range(n):
		X[i][0] = sc.apply(prob, seq[i])
		for j in range(1, i + 1):
			#r = h_{i-j} / h_i = seq[i] / seq[i-j]
			#rp = r^p.
			#Must cast the elements of seq to hp numbers if in hp mode.
			rp = ((mpf('1') * seq[i]) / (mpf('1') * seq[i-j]) if hp else seq[i] / seq[i-j]) ** sc.p
			X[i][j] = (rp * X[i][j-1] - X[i-1][j-1]) / (rp - 1)

	return X

#Extrapolates the scheme using sequence h / seq[i].
#
#problem: The problem to analyze
#scheme (Scheme): The scheme to apply to the problem and extrapolate.
#seq (Sequence): The sequence to use in the extrapolation.
#hp (bool): Whether to use high precision arithmetic or not.
#returns: An object of type Result, containing the error on the diagonal 
#         in the extrapolation table.
def analyze(problem, scheme, seq, hp, ref = None, folder = None):
	result = parse_from_file(problem.tex, seq.ref, folder, ref)
	if result == None:
		ans = problem.ans
		T = extrapolate(scheme, problem, seq.seq, hp)
		diag = np.array([T_i[len(T_i)-1] for T_i in T])
		error_all = [np.linalg.norm(x - ans) for x in diag]
		upTo = 0
		while upTo < len(diag) and error_all[upTo] != 0:
			upTo += 1

		ln_error = np.array([float(mp.log(x)) for x in error_all[0:upTo]])
		log_error = ln_error / math.log(10)
		evals = np.cumsum([scheme.get_evals(seq_i) for seq_i in seq.seq[0:upTo]])
		result = Result(problem.tex, seq.ref, evals, ln_error, log_error, ref)
		if hp:
			result.write(folder)

	return result

def parse_from_file(prob_ref, seq_ref, folder, ref):
	if folder == None:
		return None
	
	my_folder = os.path.join(folder, ref)
	if os.path.isdir(my_folder):
		print("##############################")
		print("PARSING FROM FILE")
		evals_path = os.path.join(my_folder, 'evals')
		print(evals_path)
		ln_e_path = os.path.join(my_folder, 'ln_e')
		print(ln_e_path)
		print("###############################")
		evals = np.fromfile(evals_path, 'int64')
		ln_e = np.fromfile(ln_e_path, 'float')
		log_e = ln_e / math.log(10)
		return Result(prob_ref, seq_ref, evals, ln_e, log_e, ref)
	
	else: return None


#results (list<Results>): The results from the extrapolation.
#title (str): The title of the plot.
#ref (str): A reference to the plot.
#by_seq (bool): Whether to label each curve by sequence used in the corresponding extrapolation.
#folder (str): The folder to which the plot should be saved.
#
#Will save the plot as a png file named ref in the folder given.
def plot_eval_error(results, title, ref, by_seq, folder):
	for result in results:
		my_label = result.seq_ref if by_seq else result.prob_ref
		plt.plot(result.evals, result.log_e, '.', label = my_label)
		plt.legend()

	plt.xlabel('Number of function evaluations, $N$')
	plt.ylabel('Base $10$ logarithm of absolute error, $\log \epsilon $')
	plt.title(title)
	#plt.show()
	plt.savefig(folder + ref + '.png')
	plt.clf()


#results (list<Results>): The results from the extrapolation.
#title (str): The title of the plot.
#ref (str): A reference to the plot.
#by_seq (bool): Whether to label each curve by sequence used in the corresponding extrapolation.
#max_points (int): The maximum number of points each curve should contain.
#folder (str): The folder to which the plot should be saved.
#
#Will save the plot as a png file named ref_steps in the folder given.
def plot_steps_error(results, title, ref, by_seq, max_points, folder):
	for result in results:
		my_label = result.seq_ref if by_seq else result.prob_ref
		ln_e = result.ln_e
		N = min(len(ln_e), max_points)
		steps = np.array([n+1 for n in range(N)])
		plt.plot(steps, ln_e[0:N], '.', label = my_label)
		plt.legend()
		x = np.linspace(1, N, min(100 * N, 1000))
		p = opt.curve_fit(fit_func, np.array([i+1 for i in range(len(ln_e))]), ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
		plt.plot(x, fit_func(x, *p), label = 'b = %.4g, c = %.4g, q = %.4g' % (p[0], p[1], p[2]))
		plt.legend()

	plt.xlabel('Extrapolation steps')
	plt.ylabel('Base $10$ logarithm of absolute error, $\log \epsilon $')
	plt.title(title)
	#plt.show()
	plt.savefig(folder + ref + '_steps.png')
	plt.clf()

#Does the same as plot_evals_error except it plots evals against ln of error and
#does a curve fitting on the results. 
#The plot will be saved as a png file named ref_trend under folder.
def plot_trend(results, title, ref, by_seq, folder):
	for result in results:
		my_label = result.seq_ref if by_seq else result.prob_ref
		evals = result.evals
		ln_e = result.ln_e
		plt.plot(evals, ln_e, '.', label = my_label)
		x = np.linspace(1, evals[-1], min(10 * evals[-1], 1000))
		p = opt.curve_fit(fit_func, evals, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
		validate_parameters(evals, ln_e, ref + "_" + result.seq_ref.lower(), folder)
		plt.plot(x, fit_func(x, *p), label = 'b = %.4g, c = %.4g, q = %.4g' % (p[0], p[1], p[2]))
		plt.legend()
    
	plt.xlabel('Number of function evaluations, $N$')
	plt.ylabel('Natural logarithm of absolute error, $\ln \epsilon $')
	plt.title(title)
	#plt.show()
	plt.savefig(folder + ref + '_trend.png')
	plt.clf()

#Plots log-log plot of evals against error.
#
#results (list<Results>): The results from the extrapolation.
#title (str): The title of the plot.
#ref (str): A reference to the plot.
#max_points (int): The maximum number of points each curve should contain.
#folder (str): The folder to which the plot should be saved.
#
#The plot will be saved as a png file named ref under folder.
def plot_log_log(results, title, ref, by_seq, folder): 
	for result in results:
		my_label = result.seq_ref if by_seq else result.prob_ref
		plt.plot(np.log(result.evals), result.ln_e, '.', label = my_label)
		plt.legend()

	plt.xlabel('Natural logarithm of number of function evaluations, $\ln N$')
	plt.ylabel('Natural logarithm of absolute error, $\ln \epsilon $')
	plt.title(title)
	#plt.show()
	plt.savefig(folder + ref + '.png')
	plt.clf()

#Does the same as plot_log_log except it also does curve fitting.
#
#The plot will be saved as a png file named ref_trend under folder.
def plot_log_log_lin_trend(results, title, ref, by_seq, folder):
	file = open(folder + ref + '.txt', 'w')
	for result in results:
		my_label = result.seq_ref if by_seq else result.prob_ref
		ln_evals = np.log(result.evals)
		ln_e = result.ln_e
		plt.plot(ln_evals, ln_e, '.', label = my_label)
		x = np.linspace(ln_evals[0], ln_evals[-1])
		m,b = np.polyfit(ln_evals, ln_e, 1)
		plt.plot(x, m * x + b, label = 'm = %.4g, b = %.4g' % (m, b))
		plt.legend()
		file.write('seq: %s, problem: %s, m=%.10g, b=%.10g\n' % (result.seq_ref, result.prob_ref, m, b))
    
	file.close()
	plt.xlabel('Number of function evaluations, $N$')
	plt.ylabel('Natural logarithm of absolute error, $\ln \epsilon $')
	plt.title(title)
	#plt.show()
	plt.savefig(folder + ref + '_trend.png')
	plt.clf()

#Does the same as plot_log_log except it also does curve fitting.
#
#The plot will be saved as a png file named ref_trend under folder.
def plot_log_log_power_trend(results, title, ref, by_seq, folder):
	for result in results:
		my_label = result.seq_ref if by_seq else result.prob_ref
		ln_evals = np.log(result.evals)
		ln_e = result.ln_e
		plt.plot(ln_evals, ln_e, '.', label = my_label)
		x = np.linspace(ln_evals[0], ln_evals[-1])
		p = opt.curve_fit(fit_func, ln_evals, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
		validate_parameters(ln_evals, ln_e, ref + "_" + result.seq_ref.lower(), folder)
		plt.plot(x, fit_func(x, *p), label = 'b = %.4g, c = %.4g, q = %.4g' % (p[0], p[1], p[2]))
		plt.legend()
		error = get_least_square_error(fit_func, p, ln_evals, ln_e)
    
	plt.xlabel('Natural logarithm of number of function evaluations, $\ln N$')
	plt.ylabel('Natural logarithm of absolute error, $\ln \epsilon $')
	plt.title(title)
	#plt.show()
	plt.savefig(folder + ref + '_trend.png')
	plt.clf()

def plot_by_param(param_prob, scheme, ps, title, seqs, ref, folder, cache_folder):
	qs_seq = []
	qs_seq_log_log = []
	for seq in seqs:
		qs = []
		qs_log_log = []
		for p in ps:
			prob = param_prob(p)
			result = analyze(prob, scheme, seq, True, "%s_%s_%s" % (ref, seq.ref.lower(), p), cache_folder)
			plt.clf()
			evals = result.evals
			ln_e = result.ln_e
			ln_evals = np.log(evals)
			q = opt.curve_fit(fit_func, evals, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0][-1]
			q_log_log = opt.curve_fit(fit_func, ln_evals, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0][-1]
			qs.append(q)
			qs_log_log.append(q_log_log)

		qs_seq.append(qs)
		qs_seq_log_log.append(qs_log_log)
	
	qs_seq = np.array(qs_seq)
	mln_ps = np.array([-float(mp.log(p)) for p in ps])

	for (qs, seq) in zip(qs_seq, seqs):
		plt.plot(mln_ps, qs, '.', label = seq.ref)
		plt.legend()

	plt.xlabel('Minus the natural logarithm of $a$')
	plt.ylabel('Optimal parameter $q$')
	plt.title(title)
	plt.savefig(folder + 'log_p_vs_q_%s.png' % ref)
	plt.clf()
	
	qs_seq_log_log = np.array(qs_seq_log_log)

	for (qs, seq) in zip(qs_seq_log_log, seqs):
		plt.plot(mln_ps, qs, '.', label = seq.ref)
		plt.legend()

	plt.xlabel('Minus the natural logarithm of $a$')
	plt.ylabel('Optimal parameter $q$')
	plt.title(title)
	plt.savefig(folder + 'log_p_vs_q_%s_log_log.png' % ref)
	plt.clf()

def validate_parameters(x, y, ref, folder):
	outfile = open(os.path.join(folder, ref + "_parameters_validation.txt"), 'w')
	bs = []
	cs = []
	qs = []
	for step in range(1, 4):
		for offset in range(step):
			xp = x[offset::step]
			yp = y[offset::step]
			try:
				p = opt.curve_fit(fit_func, xp, yp, [0, 1.0, 1.0], maxfev = 10000)[0]
				bs.append(p[0])
				cs.append(p[1])
				qs.append(p[2])
				outfile.write("b = %.5g,c = %.5g,q = %.5g,step = %d,offset = %d\n" % (p[0], p[1], p[2], step, offset))
			except:
				outfile.write("WARNING: OPTIMAL PARAMETERS NOT FOUND!!!\tstep = %d\toffset = %d\n" % (step, offset))

	outfile.write("################################################################################\n")
	outfile.write("################################################################################\n")

	new_len = max((len(x) + 1) // 2, 10)

	for offset in range(len(x) - new_len):
		xp = x[offset:(offset + new_len)]
		yp = y[offset:(offset + new_len)]
		try:
			p = opt.curve_fit(fit_func, xp, yp, [0, 1.0, 1.0], maxfev = 10000)[0]
			bs.append(p[0])
			cs.append(p[1])
			qs.append(p[2])
			outfile.write("b = %.5g,c = %.5g,q = %.5g,offset = %d,len = %d\n" % (p[0], p[1], p[2], offset, new_len))
		except: 
			outfile.write("WARNING: OPTIMAL PARAMETERS NOT FOUND!!!")

	bs = np.array(bs)
	cs = np.array(cs)
	qs = np.array(qs)
	bmin = np.min(bs)
	bmax = np.max(bs)
	cmin = np.min(cs)
	cmax = np.max(cs)
	qmin = np.min(qs)
	qmax = np.max(qs)
	outfile.write("################################################################################\n")
	outfile.write("################################################################################\n")
	outfile.write("bmin = %.5g, bmax = %.5g, cmin = %.5g, cmax = %.5g, qmin = %.5g, qmax = %.5g\n" % (bmin, bmax, cmin, cmax, qmin, qmax))
	outfile.close()

def get_least_square_error(f, p, x, y):
	return np.sum((y - f(x, *p))**2) / len(x)

def fit_func(x, b, c, q):
	return b - c * (x**q)
