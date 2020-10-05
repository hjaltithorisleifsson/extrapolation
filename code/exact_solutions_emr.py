import numpy as np
from mpmath import *
import math

mp.dps = 750

def emr(f, y0, a, b, n):
	h = (b-a) / (mpf(2)*n)
	y_sl = y0
	y_l = y0 + h * f(a, y0)

	for i in range(1, 2 * n):
		tmp = y_l
		y_l = y_sl + 2 * h * f(a + i * h, y_l)
		y_sl = tmp

	return y_l

def emr_all(f, y0, a, b, n):
	h = (b - a) / (2 * n)
	y = np.zeros((2*n + 1, len(y0)))

	y[0] = y0
	y[1] = y0 + h * f(a, y0)

	for i in range(1, 2 * n):
		y[i + 1] = y[i - 1] + 2*h*f(a + i*h, y[i])

	return y

def compute(f, y0, a, b, seq): 
	n = len(seq)
	X = [[0 for j in range(i + 1)] for i in range(n)]

	for i in range(n):
		X[i][0] = emr(f, y0, a, b, seq[i])
		for j in range(1, i + 1):
			r = (mpf(seq[i]) / mpf(seq[i-j])) ** 2
			X[i][j] = (r * X[i][j-1] - X[i-1][j-1]) / (r - 1)

	return X

def pendulum():
	f = lambda t,y: np.array([y[1], -mp.sin(y[0])])
	y0 = np.array([mpf(0), mpf('1')])
	a = mpf('0')
	b = mpf('1')
	seq = [(i + 1) for i in range(320)]
	X = compute(f, y0, a, b, seq)
	approx = [X_i[len(X_i) - 1] for X_i in X]
	return approx

def federpendel_2():
	y0 = np.array([mpf('1'), mpf('0'), mpf('0'), mpf('1')])
	a = mpf('0')
	b = mpf('2')
	seq = [2 * (i + 1) for i in range(510)]
	X = compute(fp, y0, a, b, seq)
	approx = [X_i[len(X_i) - 1] for X_i in X]
	return approx

def federpendel_10():
	y0 = np.array([mpf('1'), mpf('0'), mpf('0'), mpf('1')])
	a = mpf('0')
	b = mpf('10')
	seq = [100 * (i + 1) for i in range(100)]
	X = compute(fp, y0, a, b, seq)
	approx = [X_i[len(X_i) - 1] for X_i in X]
	return approx

def lorenz():
	f = lambda t,y: np.array([mpf('10') * (y[1] - y[0]), y[0] * (mpf('28') - y[2]) - y[1], y[0] * y[1] - mpf('8') / mpf('3') * y[2]])
	y0 = np.array([mpf('1'), mpf('1'), mpf('1')])
	a = mpf('0')
	b = mpf('0.2')
	seq = [(i+1) for i in range(400)]
	X = compute(f, y0, a, b, seq)
	approx = [X_i[len(X_i) - 1] for X_i in X]
	return approx

def fp(t, y): 
	q = y[0:2]
	p = y[2:4]
	q_n = np.linalg.norm(q)
	qp = -(q_n - 1) * q / q_n - np.array([mpf('0'), mpf('1')])
	return np.array([p[0], p[1], qp[0], qp[1]])

def main():
	approx = federpendel_10()
	print(approx[-1])
	print(mp.log10(np.linalg.norm(approx[-1] - approx[-2])))

main()
