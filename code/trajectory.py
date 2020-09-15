import matplotlib.pyplot as plt
import numpy as np
import math
from mpl_toolkits import mplot3d
import os

folder = os.path.join(os.path.abspath(''), '../results/trajectories')

def compute(f, y0, a, b, m, seq):
	n = len(seq)
	T = [[None] * (i + 1) for i in range(n)]

	for i in range(n):
		h = (b - a) / (2 * m * seq[i])
		T_i0 = [None] * m
		T_i0[0] = y0
		y_sl = y0
		y_l = y0 + h * f(a, y0)
		for j in range(1, 2 * seq[i]):
			tmp = y_l
			y_l = y_sl + 2 * h * f(a + i * h, y_l)
			y_sl = tmp

		T_i0[1] = y_l

		for k in range(2, m):
			offset = a + 2 * seq[i] * (k - 1)
			for j in range(2 * seq[i]):
				tmp = y_l
				y_l = y_sl + 2 * h * f(offset + i * h, y_l)
				y_sl = tmp

			T_i0[k] = y_l

		T[i][0] = np.array(T_i0)

		for j in range(1, i + 1):
			r = (seq[i] / seq[i-j]) ** 2
			T[i][j] = (r * T[i][j-1] - T[i-1][j-1]) / (r - 1)

	return T[n-1][n-1].T

def plot_mathematical_pendulum():
	f = lambda t,y: np.array([y[1], -math.sin(y[0])])
	y0 = np.array([0, 1])
	a = 0
	b = 10
	m = 1000
	seq = [2 ** i for i in range(10)]
	ys = compute(f, y0, a, b, m, seq)
	theta = ys[0]
	x = np.sin(theta)
	y = 1 - np.cos(theta)
	plot(x, y, a, b, m, "x", "y", "Mathematical pendulum", "mathematical_pendulum_trajectory")

def plot_federpendel():
	y0 = np.array([1, 0, 0, 1])
	a = 0
	b = 20
	m = 1000
	seq = [2 ** i for i in range(10)]
	ys = compute(fp, y0, a, b, m, seq)
	x = ys[0]
	y = ys[1]
	plot(x, y, a, b, m, "x", "y", "Federpendel", "federpendel_trajectory")

def fp(t, y): 
	q = y[0:2]
	p = y[2:4]
	q_n = np.linalg.norm(q)
	qp = -(q_n - 1) * q / q_n - np.array([0, 1])
	return np.array([p[0], p[1], qp[0], qp[1]])

def get_t(a, b, m):
	h = (b - a) / m
	return np.array([a + i * h for i in range(m)])
	
def plot(x, y, a, b, m, x_label, y_label, title, ref):
	t = get_t(a, b, m)
	plot_explicit(t, x, y, x_label, y_label, title, ref)

def plot_explicit(t, x, y, x_label, y_label, title, ref):
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(t, x, y)
	ax.set_xlabel("Time")
	ax.set_ylabel(x_label)
	ax.set_zlabel(y_label)
	ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
	file_name = os.path.join(folder, ref + ".png")
	if not os.path.isdir(folder):
		os.mkdir(folder)

	plt.savefig(file_name)
	plt.close()

def main():
	plot_mathematical_pendulum()
	plot_federpendel()

main()

