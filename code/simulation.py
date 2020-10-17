import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import math
from extrapolation import *

def simulate_exp(b, c, q, x):
	rel_noise = np.random.normal(0, 1, len(x)) * np.array([1.0 / (i+1) for i in range(len(x))])
	ln_error = (b - c * x**q) * (1.0 + rel_noise)
	return ln_error

def simulate_alg(m, k, x):
	rel_noise = np.random.normal(0, 1, len(x)) * 0.5
	ln_error = (m * np.log(x) + k) * (1 + rel_noise)
	return ln_error

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

def plot_stack(x, y, bcq_mat, title, ref, xlabel, ylabel):
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(x, y, '.')
	my_x = np.linspace(x[0], x[-1], min(int(10 * (x[-1] - x[0])), 1000))
	for bcq_i in bcq_mat:
		plt.plot(my_x, fit_func(my_x, *bcq_i))

	plt.show()

def main():
	b = 0.0
	c = 1.0
	q = 1.0
	x = np.array([i + 1 for i in range(100)])
	y = simulate_exp(b, c, q, x)
	(a_mean, a_var, c_mean, c_var, q_mean, q_var, bcq_mat) = get_fit_variance(x, y, b, c, q)
	print("%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e" % (a_mean, a_var, c_mean, c_var, q_mean, q_var))
	plot_stack(x, y, bcq_mat, "title", "", "x", "y")

main()


