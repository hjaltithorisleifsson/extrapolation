import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt 
from mpmath import *
import math
import os
from extrapolation import *

mp.dps = 500

folder = '/Users/hjaltithorisleifsson/Documents/extrapolation/romberg_plots/' #The folder to which results are written.
cache_folder = os.path.join(folder, 'cache')

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

def define_examples():
    results_int_seq = []
    integrands = []

    integrand = Integration(lambda x: math.cos(x)**2, lambda x: math.sin(2*x) / 4 + x / 2, 0, math.pi, '$f$', 'cos_squared')
    integrands.append(integrand)
    results_harmonic = analyze(integrand, tr, harmonic_short, False)
    results_romberg = analyze(integrand, tr, romberg_short, False)
    results_bulirsch = analyze(integrand, tr, bulirsch_short, False)
    results_int_seq.append([results_harmonic, results_romberg, results_bulirsch])

    integrand = Integration(lambda x: 1 / (0.0001 + x**2), lambda x: 100 * math.atan(100 * x), -1, 1, '$g_{10^{-2}}$', 'g_hundredth')
    integrands.append(integrand)
    results_harmonic = analyze(integrand, tr, harmonic_short, False)
    results_romberg = analyze(integrand, tr, romberg_short, False)
    results_bulirsch = analyze(integrand, tr, bulirsch_short, False)
    results_int_seq.append([results_harmonic, results_romberg, results_bulirsch])

    integrand = Integration(lambda x: 1 / (0.01 + x**2), lambda x: 10 * math.atan(10 * x), -1, 1, '$g_{10^{-1}}$','g_tenth')
    integrands.append(integrand)
    results_harmonic = analyze(integrand, tr, harmonic_short, False)
    results_romberg = analyze(integrand, tr, romberg_short, False)
    results_bulirsch = analyze(integrand, tr, bulirsch_short, False)
    results_int_seq.append([results_harmonic, results_romberg, results_bulirsch])

    integrand = Integration(lambda x: 1 / (1 + x**2), lambda x: math.atan(x), -1, 1, '$g_1$', 'g_one')
    integrands.append(integrand)
    results_harmonic = analyze(integrand, tr, harmonic_short, False)
    results_romberg = analyze(integrand, tr, romberg_short, False)
    results_bulirsch = analyze(integrand, tr, bulirsch_short, False)
    results_int_seq.append([results_harmonic, results_romberg, results_bulirsch])

    integrand = Integration(lambda x: math.log(x + 0.0001), lambda x: (x + 0.0001) * math.log(x + 0.0001) - x, 0, 1, '$h_{10^{-4}}$','h_tenthousandth')
    integrands.append(integrand)
    results_harmonic = analyze(integrand, tr, harmonic_short, False)
    results_romberg = analyze(integrand, tr, romberg_short, False)
    results_bulirsch = analyze(integrand, tr, bulirsch_short, False)
    results_int_seq.append([results_harmonic, results_romberg, results_bulirsch])

    integrand = Integration(lambda x: math.log(x + 0.01), lambda x: (x + 0.01) * math.log(x + 0.01) - x, 0, 1, '$h_{10^{-2}}$', 'h_hundredth')
    integrands.append(integrand)
    results_harmonic = analyze(integrand, tr, harmonic_short, False)
    results_romberg = analyze(integrand, tr, romberg_short, False)
    results_bulirsch = analyze(integrand, tr, bulirsch_short, False)
    results_int_seq.append([results_harmonic, results_romberg, results_bulirsch])

    integrand = Integration(lambda x: math.log(x + 1), lambda x: (x + 1) * math.log(x + 1) - x, 0, 1, '$h_1$', 'h_one')
    integrands.append(integrand)
    results_harmonic = analyze(integrand, tr, harmonic_short, False)
    results_romberg = analyze(integrand, tr, romberg_short, False)
    results_bulirsch = analyze(integrand, tr, bulirsch_short, False)
    results_int_seq.append([results_harmonic, results_romberg, results_bulirsch])

    integrand = Integration(lambda x: math.sqrt(1 - x**2), lambda x: (x * math.sqrt(1 - x**2) + math.asin(x)) / 2, -1, 1, '$i$', 'circle_area')
    integrands.append(integrand)
    results_harmonic = analyze(integrand, tr, harmonic_short, False)
    results_romberg = analyze(integrand, tr, romberg_short, False)
    results_bulirsch = analyze(integrand, tr, bulirsch_short, False)
    results_int_seq.append([results_harmonic, results_romberg, results_bulirsch])

    integrand = Integration(lambda x: 2 * math.exp(-(x**2)) / math.sqrt(math.pi), lambda x: math.erf(x), 0, 1, '$j$', 'gaussian')
    integrands.append(integrand)
    results_harmonic = analyze(integrand, tr, harmonic_short, False)
    results_romberg = analyze(integrand, tr, romberg_short, False)
    results_bulirsch = analyze(integrand, tr, bulirsch_short, False)
    results_int_seq.append([results_harmonic, results_romberg, results_bulirsch])

    return (results_int_seq, integrands)

def define_hp_examples():
    results_int_seq_hp = []
    integrands_hp = []
    strange_results = []
    strange_integrands = []

    integrand_hp = Integration(lambda x: mp.cos(x)**2, lambda x: mp.sin(2*x) / 4 + x / 2, mpf('0'), mp.pi, '$f$', 'cos_squared_hp')
    integrands_hp.append(integrand_hp)
    results_harmonic_hp = analyze(integrand_hp, tr, harmonic, True, integrand_hp.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg_hp = analyze(integrand_hp, tr, romberg, True, integrand_hp.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch_hp = analyze(integrand_hp, tr, bulirsch, True, integrand_hp.ref + "_" + bulirsch.ref.lower(), cache_folder)
    results_int_seq_hp.append([results_harmonic_hp, results_romberg_hp, results_bulirsch_hp])

    integrand_hp = Integration(lambda x: 1 / (mpf('0.0001') + mpf(x)**2), lambda x: 100 * mp.atan(mpf(100) * x), mpf('-1'), mpf('1'), '$g_{10^{-2}}$', 'g_hundredth_hp')
    integrands_hp.append(integrand_hp)
    results_harmonic_hp = analyze(integrand_hp, tr, harmonic, True, integrand_hp.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg_hp = analyze(integrand_hp, tr, romberg, True, integrand_hp.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch_hp = analyze(integrand_hp, tr, bulirsch, True, integrand_hp.ref + "_" + bulirsch.ref.lower(), cache_folder)
    results_int_seq_hp.append([results_harmonic_hp, results_romberg_hp, results_bulirsch_hp])

    integrand_hp = Integration(lambda x: 1 / (mpf('0.01') + mpf(x)**2), lambda x: 10 * mp.atan(10 * mpf(x)), mpf('-1'), mpf('1'), '$g_{10^{-1}}$','g_tenth_hp')
    integrands_hp.append(integrand_hp)
    results_harmonic_hp = analyze(integrand_hp, tr, harmonic, True, integrand_hp.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg_hp = analyze(integrand_hp, tr, romberg, True, integrand_hp.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch_hp = analyze(integrand_hp, tr, bulirsch, True, integrand_hp.ref + "_" + bulirsch.ref.lower(), cache_folder)
    results_int_seq_hp.append([results_harmonic_hp, results_romberg_hp, results_bulirsch_hp])

    integrand_hp = Integration(lambda x: 1 / (1 + mpf(x)**2), lambda x: mp.atan(x), mpf('-1'), mpf('1'), '$g_1$', 'g_one_hp')
    integrands_hp.append(integrand_hp)
    results_harmonic_hp = analyze(integrand_hp, tr, harmonic, True, integrand_hp.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg_hp = analyze(integrand_hp, tr, romberg, True, integrand_hp.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch_hp = analyze(integrand_hp, tr, bulirsch, True, integrand_hp.ref + "_" + bulirsch.ref.lower(), cache_folder)
    results_int_seq_hp.append([results_harmonic_hp, results_romberg_hp, results_bulirsch_hp])

    integrand_hp = Integration(lambda x: mp.log(mpf(x) + mpf('0.0001')), lambda x: (mpf(x) + mpf('0.0001')) * mp.log(mpf(x) + mpf('0.0001')) - mpf(x), mpf('0'), mpf('1'), '$h_{10^{-4}}$','h_tenthousandth_hp')
    integrands_hp.append(integrand_hp)
    results_harmonic_hp = analyze(integrand_hp, tr, harmonic, True, integrand_hp.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg_hp = analyze(integrand_hp, tr, romberg, True, integrand_hp.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch_hp = analyze(integrand_hp, tr, bulirsch, True, integrand_hp.ref + "_" + bulirsch.ref.lower(), cache_folder)
    results_int_seq_hp.append([results_harmonic_hp, results_romberg_hp, results_bulirsch_hp])

    integrand_hp = Integration(lambda x: mp.log(mpf(x) + mpf('0.01')), lambda x: (mpf(x) + mpf('0.01')) * mp.log(mpf(x) + mpf('0.01')) - mpf(x), mpf('0'), mpf('1'), '$h_{10^{-2}}$', 'h_hundredth_hp')
    integrands_hp.append(integrand_hp)
    results_harmonic_hp = analyze(integrand_hp, tr, harmonic, True, integrand_hp.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg_hp = analyze(integrand_hp, tr, romberg, True, integrand_hp.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch_hp = analyze(integrand_hp, tr, bulirsch, True, integrand_hp.ref + "_" + bulirsch.ref.lower(), cache_folder)
    results_int_seq_hp.append([results_harmonic_hp, results_romberg_hp, results_bulirsch_hp])

    integrand_hp = Integration(lambda x: mp.log(mpf(x) + 1), lambda x: (mpf(x) + 1) * mp.log(mpf(x) + 1) - mpf(x), mpf('0'), mpf('1'), '$h_1$', 'h_one_hp')
    integrands_hp.append(integrand_hp)
    results_harmonic_hp = analyze(integrand_hp, tr, harmonic, True, integrand_hp.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg_hp = analyze(integrand_hp, tr, romberg, True, integrand_hp.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch_hp = analyze(integrand_hp, tr, bulirsch, True, integrand_hp.ref + "_" + bulirsch.ref.lower(), cache_folder)
    results_int_seq_hp.append([results_harmonic_hp, results_romberg_hp, results_bulirsch_hp])

    integrand_hp = Integration(lambda x: mp.sqrt(1 - mpf(x)**2), lambda x: (x * mp.sqrt(1 - mpf(x)**2) + mp.asin(x)) / 2, mpf('-1'), mpf('1'), '$i$', 'circle_area_hp')
    integrands_hp.append(integrand_hp)
    strange_integrands.append(integrand_hp)
    results_harmonic_hp = analyze(integrand_hp, tr, harmonic, True, integrand_hp.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg_hp = analyze(integrand_hp, tr, romberg, True, integrand_hp.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch_hp = analyze(integrand_hp, tr, bulirsch, True, integrand_hp.ref + "_" + bulirsch.ref.lower(), cache_folder)
    strange_results.append([results_harmonic_hp, results_romberg_hp, results_bulirsch_hp])
    results_int_seq_hp.append([results_harmonic_hp, results_romberg_hp, results_bulirsch_hp])

    integrand_hp = Integration(lambda x: 2 * mp.exp(-(mpf(x)**2)) / mp.sqrt(mp.pi), lambda x: mp.erf(x), mpf('0'), mpf('1'), '$j$', 'gaussian_hp')
    integrands_hp.append(integrand_hp)
    results_harmonic_hp = analyze(integrand_hp, tr, harmonic, True, integrand_hp.ref + "_" + harmonic.ref.lower(), cache_folder)
    results_romberg_hp = analyze(integrand_hp, tr, romberg, True, integrand_hp.ref + "_" + romberg.ref.lower(), cache_folder)
    results_bulirsch_hp = analyze(integrand_hp, tr, bulirsch, True, integrand_hp.ref + "_" + bulirsch.ref.lower(), cache_folder)
    results_int_seq_hp.append([results_harmonic_hp, results_romberg_hp, results_bulirsch_hp])

    return (results_int_seq_hp, integrands_hp, strange_results, strange_integrands)

def plot_basic(results_int_seq, integrands):
    #Plot for each integrand the reults by sequence
    for (results_seq, integrand) in zip(results_int_seq, integrands): 
        title = "Integrand: %s. Double precision." % integrand.tex
        ref = integrand.ref.lower()
        plot_eval_error(results_seq, title, ref, True, folder)

def plot_basic_hp(results_int_seq_hp, integrands_hp):
    #Plot for each integrand the reults by sequence
    for (results_seq_hp, integrand_hp) in zip(results_int_seq_hp, integrands_hp):
        title = "Integrand: %s" % integrand_hp.tex
        ref = integrand_hp.ref.lower()
        plot_eval_error(results_seq_hp, title, ref, True, folder)
        plot_trend(results_seq_hp, title, ref, True, folder)
        plot_steps_error(results_seq_hp, 'Integral: %s' % integrand_hp.tex, integrand_hp.ref, True, 20, folder)
        plot_log_log_trend(results_seq_hp, 'Integral: %s' % integrand_hp.tex, '%s_log_log_pow_fit' % integrand_hp.ref, True, folder)

    #Write out all results as latex table
    file1 = open(folder + 'all_results_evals_error_exp_conv.txt', 'w')
    file2 = open(folder + 'all_results_steps_error_exp_conv.txt', 'w')
    file3 = open(folder + 'all_results_evals_error_log_log_pow_fit.txt', 'w')
    for results_seq_hp in results_int_seq_hp:
        for result_hp in results_seq_hp: 
            ln_e = result_hp.ln_e
            ln_evals = np.log(result_hp.evals)
            steps = np.array([i+1 for i in range(len(ln_e))])
            p1 = opt.curve_fit(fit_func, result_hp.evals, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
            e1 = get_least_square_error(fit_func, p1, result_hp.evals, ln_e)
            p2 = opt.curve_fit(fit_func, steps, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
            e2 = get_least_square_error(fit_func, p2, steps, ln_e)
            p3 = opt.curve_fit(fit_func, ln_evals, ln_e, [0, 1.0, 1.0], maxfev = 10000)[0]
            e3 = get_least_square_error(fit_func, p3, ln_evals, ln_e)
            file1.write('%s & %s & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) \\\\\n' % (result_hp.prob_ref, result_hp.seq_ref, p1[0], p1[1], p1[2], e1))
            file2.write('%s & %s & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) \\\\\n' % (result_hp.prob_ref, result_hp.seq_ref, p2[0], p2[1], p2[2], e2))
            file3.write('%s & %s & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) & \\(%.5g\\) \\\\\n' % (result_hp.prob_ref, result_hp.seq_ref, p3[0], p3[1], p3[2], e3))

    file1.close()
    file2.close()
    file3.close()

def plot_strange_results(results, integrands):
    for (results_seq, integrand) in zip(results, integrands):
        title = "Integrand: %s" % integrand.tex
        ref = "%s_log_log" % integrand.ref.lower()
        plot_log_log_trend(results_seq, title, ref, True, folder)

def plot_q_g():
    param_integrand = lambda q: Integration(lambda x: mpf('1') / (q**2 + x**2), lambda x: 1 / q * mp.atan(1 / q * x), mpf('-1'), mpf('1'), 'g_%.5g' % q, 'g_%.5g' % q)
    ps = np.array([mpf(0.5) ** i for i in range(15)])
    title = 'Integrand: $g_a$'
    plot_by_param(param_integrand, tr, ps, title, seqs, 'g_a_by_param', folder, cache_folder)

def plot_q_h():
    param_integrand = lambda q: Integration(lambda x: mp.log(x + q), lambda x: (x + q) * mp.log(x + q) - x, mpf('0'), mpf('1'), 'h_%.5g' % q, 'h_%.5g' % q)
    ps = np.array([mpf(0.5) ** i for i in range(15)])
    title = 'Integrand: $h_a$'
    plot_by_param(param_integrand, tr, ps, title, seqs, 'h_a_by_param', folder, cache_folder)


harmonic = harmonic_seq(120) #120
romberg = romberg_seq(14) #14
bulirsch = bulirsch_seq(25)

harmonic_short = harmonic_seq(100)
romberg_short = romberg_seq(12)
bulirsch_short = bulirsch_seq(21)

seqs = [harmonic, romberg, bulirsch]

tr = TrapezoidalRule()

def main():
    (results_int_seq, integrands) = define_examples()
    (results_int_seq_hp, integrands_hp, strange_results, strange_integrands) = define_hp_examples()
    plot_basic(results_int_seq, integrands)
    plot_basic_hp(results_int_seq_hp, integrands_hp)
    plot_strange_results(strange_results, strange_integrands)
    plot_q_g()
    plot_q_h()

main()
