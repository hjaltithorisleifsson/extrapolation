import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt 
from mpmath import *
import math
import os
from extrapolation import *

import multiprocessing

mp.dps = 500

folder = os.path.join(os.path.abspath(''), '../results/romberg_plots')

if not os.path.isdir(folder):
    os.mkdir(folder)

cache_folder = os.path.join(folder, 'cache')

if not os.path.isdir(cache_folder):
    os.mkdir(cache_folder)

class TrapezoidalRule(Scheme):
    def __init__(self):
        super(TrapezoidalRule, self).__init__(2)

    def apply(self, inte, m):
        (a,b) = inte.interval
        h = (b - a) / m
        I = 0.5 * (inte.f(a) + inte.f(b))
        for i in range(1, m):
            I += inte.f(a + i * h)

        return I * h

    def get_evals(self, n):
        return n + 1

class Integration:

    def __init__(self, f, F, a, b, tex, ref):
        self.f = f 
        self.ans = F(b) - F(a)
        self.interval = (a,b)
        self.tex = tex
        self.ref = ref

def process_examples():
    results_int_seq = []
    integrands = []

    integrand = Integration(lambda x: math.cos(x)**2, lambda x: math.sin(2*x) / 4 + x / 2, 0, math.pi, '$f$', 'cos_squared')
    integrands.append(integrand)
    results_int_seq.append(process_integrand(integrand))

    integrand = Integration(lambda x: 1 / (0.0001 + x**2), lambda x: 100 * math.atan(100 * x), -1, 1, '$g_{10^{-2}}$', 'g_hundredth')
    integrands.append(integrand)
    results_int_seq.append(process_integrand(integrand))

    integrand = Integration(lambda x: 1 / (0.01 + x**2), lambda x: 10 * math.atan(10 * x), -1, 1, '$g_{10^{-1}}$','g_tenth')
    integrands.append(integrand)
    results_int_seq.append(process_integrand(integrand))

    integrand = Integration(lambda x: 1 / (1 + x**2), lambda x: math.atan(x), -1, 1, '$g_1$', 'g_one')
    integrands.append(integrand)
    results_int_seq.append(process_integrand(integrand))

    integrand = Integration(lambda x: math.log(x + 0.0001), lambda x: (x + 0.0001) * math.log(x + 0.0001) - x, 0, 1, '$h_{10^{-4}}$','h_tenthousandth')
    integrands.append(integrand)
    results_int_seq.append(process_integrand(integrand))

    integrand = Integration(lambda x: math.log(x + 0.01), lambda x: (x + 0.01) * math.log(x + 0.01) - x, 0, 1, '$h_{10^{-2}}$', 'h_hundredth')
    integrands.append(integrand)
    results_int_seq.append(process_integrand(integrand))

    integrand = Integration(lambda x: math.log(x + 1), lambda x: (x + 1) * math.log(x + 1) - x, 0, 1, '$h_1$', 'h_one')
    integrands.append(integrand)
    results_int_seq.append(process_integrand(integrand))

    integrand = Integration(lambda x: math.sqrt(1 - x**2), lambda x: (x * math.sqrt(1 - x**2) + math.asin(x)) / 2, -1, 1, '$i$', 'circle_area')
    integrands.append(integrand)
    results_int_seq.append(process_integrand(integrand))

    integrand = Integration(lambda x: 2 * math.exp(-(x**2)) / math.sqrt(math.pi), lambda x: math.erf(x), 0, 1, '$j$', 'gaussian')
    integrands.append(integrand)
    results_int_seq.append(process_integrand(integrand))

    return (results_int_seq, integrands)

def process_integrand(integrand):
    results_harmonic = analyze(integrand, tr, harmonic_short, False, integrand.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg = analyze(integrand, tr, romberg_short, False, integrand.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch = analyze(integrand, tr, bulirsch_short, False, integrand.ref + "_" + bulirsch.ref.lower(), cache_folder)
    return [results_harmonic, results_romberg, results_bulirsch]

def process_hp_integrand(integrand): 
    results_harmonic = analyze(integrand, tr, harmonic, True, integrand.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg = analyze(integrand, tr, romberg, True, integrand.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch = analyze(integrand, tr, bulirsch, True, integrand.ref + "_" + bulirsch.ref.lower(), cache_folder)
    return [results_harmonic, results_romberg, results_bulirsch]

def cos_squared_hp(x):
    return mp.cos(x)**2

def cos_squared_int_hp(x):
    return mp.sin(2*x) / 4 + x / 2

def g_hundredth_hp(x):
    return 1 / (mpf('0.0001') + mpf(x)**2)

def g_hundredth_int_hp(x):
    return 100 * mp.atan(mpf(100) * x)

def g_tenth_hp(x):
    return 1 / (mpf('0.01') + mpf(x)**2)

def g_tenth_int_hp(x):
    return 10 * mp.atan(10 * mpf(x))

def g_one_hp(x):
    return 1 / (1 + mpf(x)**2)

def g_one_int_hp(x):
    return mp.atan(x)

def h_tenthousandth_hp(x):
    return mp.log(mpf(x) + mpf('0.0001'))

def h_tenthousandth_int_hp(x):
    return (mpf(x) + mpf('0.0001')) * mp.log(mpf(x) + mpf('0.0001')) - mpf(x)

def h_hundredth_hp(x):
    return mp.log(mpf(x) + mpf('0.01'))

def h_hundredth_int_hp(x):
    return (mpf(x) + mpf('0.01')) * mp.log(mpf(x) + mpf('0.01')) - mpf(x)

def h_one_hp(x):
    return mp.log(mpf(x) + 1)

def h_one_int_hp(x):
    return (mpf(x) + 1) * mp.log(mpf(x) + 1) - mpf(x)

def circle_area_hp(x):
    return mp.sqrt(1 - mpf(x)**2)

def circle_area_int_hp(x):
    return (x * mp.sqrt(1 - mpf(x)**2) + mp.asin(x)) / 2

def gaussian_hp(x):
    return 2 * mp.exp(-(mpf(x)**2)) / mp.sqrt(pi_hp)

def gaussian_int_hp(x):
    return mp.erf(x)

pi_hp = mpf('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819644288109756659334461284756482337867831652712019091456485669234603486104543266482133936072602491412737245870066063155881748815209209628292540917153643678925903600113305305488204665213841469519415116094330572703657595919530921861173819326117931051185480744623799627495673518857527248912279381830119491')

def process_hp_examples():
    integrands_hp = []

    integrand_hp = Integration(cos_squared_hp, cos_squared_int_hp, mpf('0'), pi_hp, '$f$', 'cos_squared_hp')
    integrands_hp.append(integrand_hp)

    integrand_hp = Integration(g_hundredth_hp, g_hundredth_int_hp, mpf('-1'), mpf('1'), '$g_{10^{-2}}$', 'g_hundredth_hp')
    integrands_hp.append(integrand_hp)

    integrand_hp = Integration(g_tenth_hp, g_tenth_int_hp, mpf('-1'), mpf('1'), '$g_{10^{-1}}$','g_tenth_hp')
    integrands_hp.append(integrand_hp)

    integrand_hp = Integration(g_one_hp, g_one_int_hp, mpf('-1'), mpf('1'), '$g_1$', 'g_one_hp')
    integrands_hp.append(integrand_hp)

    integrand_hp = Integration(h_tenthousandth_hp, h_tenthousandth_int_hp, mpf('0'), mpf('1'), '$h_{10^{-4}}$','h_tenthousandth_hp')
    integrands_hp.append(integrand_hp)

    integrand_hp = Integration(h_hundredth_hp, h_hundredth_int_hp, mpf('0'), mpf('1'), '$h_{10^{-2}}$', 'h_hundredth_hp')
    integrands_hp.append(integrand_hp)

    integrand_hp = Integration(h_one_hp, h_one_int_hp, mpf('0'), mpf('1'), '$h_1$', 'h_one_hp')
    integrands_hp.append(integrand_hp)

    integrand_hp = Integration(circle_area_hp, circle_area_int_hp, mpf('-1'), mpf('1'), '$i$', 'circle_area_hp')
    integrands_hp.append(integrand_hp)

    integrand_hp = Integration(gaussian_hp, gaussian_int_hp, mpf('0'), mpf('1'), '$j$', 'gaussian_hp')
    integrands_hp.append(integrand_hp)

    pool = multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 2))
    results_int_seq_hp = pool.map(process_hp_integrand, integrands_hp)
    pool.close()

    return (results_int_seq_hp, integrands_hp)

def plot_basic(results_int_seq, integrands):
    #Plot for each integrand the results by sequence
    for (results_seq, integrand) in zip(results_int_seq, integrands): 
        title = "Integrand: %s. Double precision." % integrand.tex
        ref = integrand.ref.lower()
        plot_eval_error(results_seq, title, ref, True, folder)

def plot_basic_hp(results_int_seq_hp, integrands_hp):
    #Plot for each integrand the results by sequence
    for (results_seq_hp, integrand_hp) in zip(results_int_seq_hp, integrands_hp):
        title = "Integrand: %s" % integrand_hp.tex
        ref = integrand_hp.ref.lower()
        plot_eval_error(results_seq_hp, title, ref, True, folder)
        plot_trend(results_seq_hp, title, ref, True, folder)
        plot_steps_error(results_seq_hp, 'Integral: %s' % integrand_hp.tex, integrand_hp.ref, True, 20, folder)

    #Write out all results as latex table
    file1 = open(os.path.join(folder, 'all_results_evals_error_exp_conv.txt'), 'w')
    file2 = open(os.path.join(folder, 'all_results_steps_error_exp_conv.txt'), 'w')
    for results_seq_hp in results_int_seq_hp:
        for result_hp in results_seq_hp: 
            ln_e = result_hp.ln_e
            steps = np.array([i+1 for i in range(len(ln_e))])
            p1 = opt.curve_fit(fit_func, result_hp.evals, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
            e1 = get_least_square_error(fit_func, p1, result_hp.evals, ln_e)
            p2 = opt.curve_fit(fit_func, steps, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
            e2 = get_least_square_error(fit_func, p2, steps, ln_e)
            file1.write('%s & %s & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) \\\\\n' % (result_hp.prob_ref, result_hp.seq_ref, p1[0], p1[1], p1[2], e1))
            file2.write('%s & %s & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) \\\\\n' % (result_hp.prob_ref, result_hp.seq_ref, p2[0], p2[1], p2[2], e2))

    file1.close()
    file2.close()

def plot_strange_results(results, integrands):
    for (results_seq, integrand) in zip(results, integrands):
        title = "Integrand: %s" % integrand.tex
        ref = "%s_log_log" % integrand.ref.lower()
        plot_log_log_trend(results_seq, title, ref, True, folder)


harmonic = harmonic_seq(120)
romberg = romberg_seq(20)
bulirsch = bulirsch_seq(25)

harmonic_short = harmonic_seq(100)
romberg_short = romberg_seq(12)
bulirsch_short = bulirsch_seq(21)

seqs = [harmonic, romberg, bulirsch]

tr = TrapezoidalRule()

def main():
    (results_int_seq, integrands) = process_examples()
    (results_int_seq_hp, integrands_hp) = process_hp_examples()
    plot_basic(results_int_seq, integrands)
    plot_basic_hp(results_int_seq_hp, integrands_hp)

main()
