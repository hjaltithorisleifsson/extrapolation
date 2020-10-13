import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt 
from mpmath import *
import math
from extrapolation import *

mp.dps = 500

folder = os.path.join(os.path.abspath(''), '../results/diff_quot_plots')

if not os.path.isdir(folder):
    os.mkdir(folder)

cache_folder = os.path.join(folder, 'cache')

if not os.path.isdir(cache_folder):
    os.mkdir(cache_folder)

class SymmetricDifference(Scheme):

	def __init__(self):
		super(SymmetricDifference, self).__init__(2)

	def apply(self, quot, n):
		h = quot.h0 / n
		return (quot.f(quot.x + h) - quot.f(quot.x - h)) / (2 * h)

	def get_evals(self, n, m):
		return 2

class Quotient:
	def __init__(self, f, x, h0, dfx, tex, ref):
		self.f = f
		self.x = x
		self.m = 1
		self.h0 = h0
		self.ans = dfx
		self.tex = tex
		self.ref = ref

def plot_basic():
	quotients = []
	quotients.append(Quotient(lambda x: 0.0 if x < 0 else math.exp(-1.0 / x), 0.0, 0.5, 0.0, '$d/dx|_{x=0}r(x)$', 'rho'))
	quotients.append(Quotient(lambda x: x * math.exp(-1.0/x**2), 0.0, 0.5, 0.0, '$d/dx|_{x=0}xe^{-1/x^2}$. $h=1/2$', 'xemxm2'))
	quotients.append(Quotient(lambda x: math.sin(x), 0.0, 0.5, 1.0, '$d/dx|_{x=0}\sin x$. $h=1/2$', 'sin'))
	quotients.append(Quotient(lambda x: math.log(x + 1), 0.0, 0.5, 1.0, '$d/dx|_{x=0} \ln(x + 1)$. $h=1/2$', 'h_one'))
	quotients.append(Quotient(lambda x: x * x * math.sin(1 / x), 0.0, 1.0, 0.0, '$d/dx|_{x=0} x^2 \sin(1/x)$. $h=1$', 'xsin'))
	quotients.append(Quotient(lambda x: math.sqrt(x + 1), 0.0, 0.5, 0.5, '$d/dx|_{x=0} \sqrt{x + 1}$. $h=1/2$', 'sqrt_1'))
	quotients.append(Quotient(lambda x: math.exp(x), 0, 1.0, 1.0, '$d/dx|_{x=0}e^x$. $h=1$', 'exp_0'))

	#hp is for 'high precision'
	hp_quotients = []
	hp_quotients.append(Quotient(lambda x: 0 if x < 0 else mp.exp(-1/x), mpf('0'), mpf('0.5'), mpf('0'), '$d/dx|_{x=0}r(x)$', 'rho_hp'))
	hp_quotients.append(Quotient(lambda x: x * mp.exp(-1/mpf(x)**2), mpf('0'), mpf('0.5'), mpf('0'), '$d/dx|_{x=0}xe^{-1/x^2}$. $h=1/2$', 'xemxm2_hp'))
	hp_quotients.append(Quotient(lambda x: mp.sin(x), 0.0, mpf('0.5'), mpf('1'), '$d/dx|_{x=0}\sin x$. $h=1/2$', 'sin_hp'))
	hp_quotients.append(Quotient(lambda x: mp.log(mpf(x) + mpf(1)), mpf(0.0), mpf(0.5), mpf(1), '$d/dx|_{x=0} \ln(x + 1)$. $h=1/2$', 'h_one_hp'))
	hp_quotients.append(Quotient(lambda x: mpf(x) * mpf(x) * mp.sin(mpf(1) / mpf(x)), mpf(0.0), mpf(1.0), mpf(0.0), '$d/dx|_{x=0} x^2 \sin(1/x)$. $h=1$', 'xsin_hp'))
	hp_quotients.append(Quotient(lambda x: mp.sqrt(mpf(x) + 1), 0.0, mpf('0.5'), mpf('0.5'), '$d/dx|_{x=0} \sqrt{x + 1}$. $h=1/2$', 'sqrt_1_hp'))
	hp_quotients.append(Quotient(lambda x: mp.exp(x), 0.0, mpf(1.0), mpf(1.0), '$d/dx|_{x=0}e^x$', 'exp_0_hp'))

	results_quot_seq = []
	hp_results_quot_seq = []

	for quotient in quotients:
		results_seq = []
		for seq in seqs:
			results_seq.append(analyze(quotient, sdq, seq, False))

		results_quot_seq.append(results_seq)

	for hp_quotient in hp_quotients:
		hp_results_seq = []
		for seq in seqs:
			hp_results_seq.append(analyze(hp_quotient, sdq, seq, True, hp_quotient.ref + "_" + seq.ref.lower(), cache_folder))

		hp_results_quot_seq.append(hp_results_seq)

	for (results_seq, quotient) in zip(results_quot_seq, quotients):
		plot_eval_error(results_seq, 'Quotient: ' + quotient.tex + '. Double precision', quotient.ref, True, folder)

	for (results_seq, quotient) in zip(hp_results_quot_seq, hp_quotients):
		plot_eval_error(results_seq, 'Quotient: %s' % quotient.tex, quotient.ref, True, folder)
		plot_trend(results_seq, 'Quotient: %s' % quotient.tex, quotient.ref, True, folder)

	file = open(os.path.join(folder, 'all_results.txt'), 'w')

	for hp_results_seq in hp_results_quot_seq:
		for hp_result in hp_results_seq:
			ln_e = hp_result.ln_e
			p = opt.curve_fit(fit_func, hp_result.evals, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
			rho_lin = get_rho_lin(p, hp_result.evals, ln_e)
			rho_log = get_rho_log(p, hp_result.evals, ln_e)
			file.write('%s & %s & \\(%.2e\\) & \\(%.2e\\) & \\(%.2e\\) & \\(%.2e\\) & \\(%.2e\\) \\\\\n' % (hp_result.prob_ref, hp_result.seq_ref, p[0], p[1], p[2], rho_lin, rho_log))

	file.close()

seqs = []
seqs.append(romberg_seq(35))
seqs.append(harmonic_seq(35))

sdq = SymmetricDifference()

def main():
	plot_basic()

main()
