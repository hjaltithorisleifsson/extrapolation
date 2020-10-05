import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import math
from extrapolation import *

folder = os.path.join(os.path.abspath(''), '../results/simulation_plots')

def simulate_exp(b, c, q, x):
	rel_noise = np.random.normal(0, 1, len(x)) * 0.5
	ln_error = (b - c * x**p) * (1.0 + rel_noise)
	return ln_error

def simulate_alg(m, k, x):
	rel_noise = np.random.normal(0, 1, len(x)) * 0.5
	ln_error = (m * np.log(x) + k) * (1 + rel_noise)
	return ln_error

def main():