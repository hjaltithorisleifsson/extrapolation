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
#returns: The extrapolation table as an np.array of np.arrays.
def extrapolate(sc, prob, seq, hp):
	n = len(seq)
	X = [[None] * (i+1) for i in range(n)]

	#X[i][j] = T_ij
	for i in range(n):
		X[i][0] = sc.apply(prob, seq[i])
		for j in range(1, i + 1):
			#r = h_{i-j} / h_i = seq[i] / seq[i-j]
			#rp = r^p.
			#Must cast the elements of seq to hp numbers if in hp mode.
			rp = ((mpf('1') * seq[i]) / (mpf('1') * seq[i-j]) if hp else seq[i] / seq[i-j]) ** sc.p
			X[i][j] = (rp * X[i][j-1] - X[i-1][j-1]) / (rp - 1)

	return np.array([np.array(X_i) for X_i in X])

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
		evals_path = os.path.join(my_folder, 'evals')
		ln_e_path = os.path.join(my_folder, 'ln_e')
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
	number_of_points_res = get_number_of_points(results)
	
	for (result, number_of_points) in zip(results, number_of_points_res):
		my_label = result.seq_ref if by_seq else result.prob_ref
		evals = result.evals
		log_e = result.log_e
		plt.plot(evals[0:number_of_points], result.log_e[0:number_of_points], '.', label = my_label)
		plt.legend()

	plt.xlabel('Number of function evaluations, $N$')
	plt.ylabel('Base $10$ logarithm of absolute error, $\log \epsilon $')
	plt.title(title)
	plt.savefig(os.path.join(folder, ref + '.png'))
	plt.close()

#results (list<Results>): The results from the extrapolation.
#
#Returns a list containing the number of points, which should be used when plotting the results together, evals against error.
def get_number_of_points(results):
	max_evals = np.array([result.evals[-1] for result in results])
	end = np.min(max_evals)

	number_of_points = []
	for result in results: 
		evals = result.evals
		i = 0
		while i < len(evals) and evals[i] < 2 * end:
			i += 1

		number_of_points.append(i)

	return number_of_points

#results (list<Results>): The results from the extrapolation.
#title (str): The title of the plot.
#ref (str): A reference to the plot.
#by_seq (bool): Whether to label each curve by sequence used in the corresponding extrapolation.
#max_points (int): The maximum number of points each curve should contain.
#folder (str): The folder to which the plot should be saved.
#
#Will save the plot as a png file named ref_steps in the folder given.
def plot_steps_error(results, title, ref, by_seq, max_points, folder):
	xlabel = 'Extrapolation steps'
	ylabel = 'Natural logarithm of absolute error, $\ln \epsilon $'
	acq_vars_by_result = []
	p_by_result = []
	for result in results:
		my_label = result.seq_ref if by_seq else result.prob_ref
		ln_e = result.ln_e
		N = min(len(ln_e), max_points)
		steps = np.array([n+1 for n in range(N)])
		steps_all = np.array([i+1 for i in range(len(ln_e))])
		plt.plot(steps, ln_e[0:N], '.', label = my_label)
		plt.legend()
		x = np.linspace(1, N, min(100 * N, 1000))
		p = opt.curve_fit(fit_func, steps_all, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
		p_by_result.append(p)
		acq_vars = get_fit_variance(steps_all, ln_e, *p)
		acq_vars_by_result.append(acq_vars)
		plt.plot(x, fit_func(x, *p), label = 'b = %.4g, c = %.4g, q = %.4g' % (p[0], p[1], p[2]))
		plt.legend()

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(os.path.join(folder, ref + '_steps.png'))
	plt.close()

	file = open(os.path.join(folder, ref + '_steps_variance.txt'), 'w')
	for (acq_vars, p, result) in zip(acq_vars_by_result, p_by_result, results):
		if acq_vars != None:
			(a_mean, a_var, c_mean, c_var, q_mean, q_var, bcq_mat) = acq_vars
			stack_title = title + '. Stack plot. Sequence: ' + result.seq_ref
			stack_ref = os.path.join(folder, ref + '_' + result.seq_ref.lower() + '_steps_stack.png')
			ln_e = result.ln_e
			steps_all = np.array([i+1 for i in range(len(ln_e))])
			plot_stack(steps_all, ln_e, bcq_mat, p, stack_title, stack_ref, xlabel, ylabel)
			file.write('%s & %s & \\(%.4g\\) & \\(%.4g\\) & \\(%.4g\\) & \\(%.4g\\) & \\(%.4g\\) & \\(%.4g\\) \\\\\n' % (result.seq_ref, "lin-ln steps-error", a_mean, a_var, c_mean, c_var, q_mean, q_var))
		else:
			file.write('%s & %s & . & . & . & . & . & . \\\\\n' % (result.seq_ref, "lin-ln steps-error"))

	file.close()

#Does the same as plot_evals_error except it plots evals against ln of error and
#does a curve fitting on the results. 
#The plot will be saved as a png file named ref_trend under folder.
def plot_trend(results, title, ref, by_seq, folder):
	number_of_points_res = get_number_of_points(results)

	xlabel = 'Number of function evaluations, $N$'
	ylabel = 'Natural logarithm of absolute error, $\ln \epsilon $'
	acq_vars_by_result = []
	p_by_result = []
	for (result, number_of_points) in zip(results, number_of_points_res):
		my_label = result.seq_ref if by_seq else result.prob_ref
		evals = result.evals
		ln_e = result.ln_e
		plt.plot(evals[0:number_of_points], ln_e[0:number_of_points], '.', label = my_label)
		p = opt.curve_fit(fit_func, evals, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
		p_by_result.append(p)
		acq_vars = get_fit_variance(evals, ln_e, *p)
		acq_vars_by_result.append(acq_vars)
		end_val = evals[number_of_points- 1]
		x = np.linspace(1, end_val, min(10 * (end_val - 1), 1000))
		plt.plot(x, fit_func(x, *p), label = 'b = %.4g, c = %.4g, q = %.4g' % (p[0], p[1], p[2]))
		plt.legend()

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(os.path.join(folder, ref + '_trend.png'))
	plt.close()

	file = open(os.path.join(folder, ref + '_variance.txt'), 'w')
	for (acq_vars, p, result) in zip(acq_vars_by_result, p_by_result, results):
		if acq_vars != None:
			(a_mean, a_var, c_mean, c_var, q_mean, q_var, bcq_mat) = acq_vars
			stack_title = title + '. Stack plot. Sequence: ' + result.seq_ref
			stack_ref = os.path.join(folder, ref + '_' + result.seq_ref.lower() + '_stack.png')
			evals = result.evals
			ln_e = result.ln_e
			plot_stack(evals, ln_e, bcq_mat, p, stack_title, stack_ref, xlabel, ylabel)
			file.write('%s & %s & \\(%.4g\\) & \\(%.4g\\) & \\(%.4g\\) & \\(%.4g\\) & \\(%.4g\\) & \\(%.4g\\) \\\\\n' % (result.seq_ref, "lin-ln evals-error", a_mean, a_var, c_mean, c_var, q_mean, q_var))
		else:
			file.write('%s & %s & . & . & . & . & . & . \\\\\n' % (result.seq_ref, "lin-ln evals-error"))

	file.close()


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
	plt.savefig(os.path.join(folder, ref + '.png'))
	plt.close()

#Does the same as plot_log_log except it also does linear fitting
#
#The plot will be saved as a png file named ref_trend under folder.
def plot_log_log_trend(results, title, ref, by_seq, folder):
	xlabel = 'Natural logarithm of number of function evaluations, $\ln N$'
	ylabel = 'Natural logarithm of absolute error, $\ln \epsilon $'
	acq_vars_by_result = []
	p_by_result = []
	for result in results:
		my_label = result.seq_ref if by_seq else result.prob_ref
		ln_evals = np.log(result.evals)
		ln_e = result.ln_e
		plt.plot(ln_evals, ln_e, '.', label = my_label)
		x = np.linspace(ln_evals[0], ln_evals[-1])
		k,c = np.polyfit(ln_evals, ln_e, 1)
		plt.plot(x, k * x + c)
		plt.legend()

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(os.path.join(folder,ref + '_trend.png'))
	plt.close()

def get_fit_variance(x, y, b0, c0, q0):
	bs = []
	cs = []
	qs = []

	new_len = min(len(x) - 3, max(10, (len(x) + 1) // 2))
	success = True
	for offset in range(3, len(x) - new_len):
		xp = x[offset:(offset + new_len)]
		yp = y[offset:(offset + new_len)]
		try:
			p = opt.curve_fit(fit_func, xp, yp, [b0, c0, q0], maxfev = 10000)[0]
			bs.append(p[0])
			cs.append(p[1])
			qs.append(p[2])
		except: 
			return None

	bs = np.array(bs)
	cs = np.array(cs)
	qs = np.array(qs)

	a = np.exp(bs)
	a_mean = np.mean(a)
	a_variance = np.sum((a - a_mean) ** 2) / (len(a) * (a_mean ** 2))

	c_mean = np.mean(cs)
	c_variance = np.sum((cs - c_mean) ** 2) / (len(cs) * (c_mean ** 2))

	q_mean = np.mean(qs)
	q_variance = np.sum((qs - q_mean) ** 2) / (len(qs) * (q_mean ** 2))

	bcq_mat = np.array([bs, cs, qs]).T
	return (a_mean, a_variance, c_mean, c_variance, q_mean, q_variance, bcq_mat)

def plot_stack(x, y, bcq_mat, bcq_0, title, ref, xlabel, ylabel):
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(x, y, '.')
	my_x = np.linspace(x[0], x[-1], min(int(10 * (x[-1] - x[0])), 1000))
	plt.plot(my_x, fit_func(my_x, *bcq_0))
	for bcq_i in bcq_mat:
		plt.plot(my_x, fit_func(my_x, *bcq_i))

	plt.savefig(ref)
	plt.close()

def get_rho_log(p, x, y):
	return get_relative_residual(fit_func, p, x, y)

def get_rho_lin(p, x, y):
	try:
		eps = np.exp(y)
		pp = (math.exp(p[0]), p[1], p[2])
		return get_relative_residual(fit_func_lin, pp, x, eps)
	except:
		return -1

def get_relative_residual(f, p, x, y):
	try:
		return np.sum((y - f(x, *p))**2) / np.sum(y**2)
	except:
		return -1

def fit_func(x, b, c, q):
	return b - c * (x**q)

def fit_func_lin(x, A, c, q):
	return A * np.exp(-c * (x**q))
