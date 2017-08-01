import pandas as pd
from pudb import set_trace
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pkl
import datetime
import os
import scipy.optimize

import splines
import enrichment_model_data as ed
import smooth_vonmises as sv


#
# Position recurrence
#

def determine_position_recurrence_parameters(
        data, logpmin, logpmax, method='cv', max_iter=1000,
        thresh=1e-12, repeats=20, n_knots=50,
        savedir='/analysis/Jeff/Df16A/Df_remap_paper/data/enrichment_model'):
    distance_recur = ed.recurrence_by_position(data, method=method)
    distance_recur = distance_recur[np.argsort(distance_recur[:, 0])]

    if method == 'cv':
        knots = np.linspace(-np.pi, np.pi, n_knots)
    elif method == 'si':
        knots = np.linspace(-0.5, 0.5, n_knots)

    repeat_range = range(len(distance_recur))
    repeat_indices = [np.random.permutation(repeat_range) for _ in range(repeats)]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    time_str = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
    fn = 'recur_optimize_results_{}.pdf'.format(time_str)

    print 'Saving data to: ' + os.path.join(savedir, fn)

    for log_penalty in np.linspace(logpmin, logpmax, 10):
        fits, logL = splines.nonparameteric_logistic_regression(
            distance_recur, knots, np.exp(log_penalty), max_iter=max_iter, thresh=thresh,
            repeats=repeat_indices, single_value_test=True)
        ax.plot(distance_recur[:, 0], np.mean(fits, axis=0), label=str(log_penalty))
        print log_penalty, logL

    ax.plot(distance_recur[:, 0], distance_recur[:, 1] + (np.random.random(len(distance_recur)) - 0.5) / 10., 'ko')
    ax.set_ylim(-0.055, 1.055)
    ax.set_ylabel('Recurrence')
    ax.set_xlabel('Distance from reward')
    ax.set_title('logpmin={}, logpmax={}'.format(logpmin, logpmax))
    ax.legend()

    fig.savefig(os.path.join(savedir, fn), format='pdf')
    plt.close('all')


def return_position_recurrence_fit(
        data, log_penalty, method='cv',
        max_iter=1000, thresh=1e-12, n_boots=1000, n_knots=50):

    distance_recur = ed.recurrence_by_position(data, method=method)

    if method == 'cv':
        knots = np.linspace(-np.pi, np.pi, n_knots)
    elif method == 'si':
        knots = np.linspace(-0.5, 0.5, n_knots)

    spline = splines.CyclicSpline(knots)
    N = spline.design_matrix(distance_recur[:, 0])
    Omega = spline.omega

    theta = splines.fit_model(
        distance_recur[:, 1], N, Omega, knots, np.exp(log_penalty),
        max_iter=max_iter, thresh=thresh)

    boots_theta = []
    n_samples = len(distance_recur)
    for _ in range(n_boots):
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_theta = splines.fit_model(
            distance_recur[idxs, 1], N[idxs], Omega, knots,
            np.exp(log_penalty), max_iter=max_iter, thresh=thresh)
        boots_theta.append(boot_theta)

    return theta, np.array(boots_theta)

#
# Novel position
#


def determine_novel_position_parameters(data, kmin, kmax, method='cv', repeats=1):

    novel_pc_distances, recur_pc_distances = ed.new_recur_pc_distances(data, method=method)

    if method == 'cv':
        x = np.linspace(-np.pi, np.pi, 100)
    elif method == 'si':
        x = np.linspace(-0.5, 0.5, 100)

    repeat_range = range(len(novel_pc_distances))
    repeat_indices = [np.random.permutation(repeat_range) for _ in range(repeats)]

    for kappa in np.exp(np.linspace(kmin, kmax, 10)):
        print kappa, sv.cross_val(kappa, novel_pc_distances, repeats=repeat_indices, single_value_test=True)
        y = [sv.density(v, novel_pc_distances, kappa) for v in x]
        plt.plot(x, y)
    plt.show()


def return_novel_position_fit(data, kappa, method='cv', n_boots=1000, x_vals=None):

    novel_pc_distances, recur_pc_distances = ed.new_recur_pc_distances(data, method=method)

    if x_vals is None:
        if method == 'cv':
            x_vals = np.linspace(-np.pi, np.pi, 100)
        elif method == 'si':
            x_vals = np.linspace(-0.5, 0.5, 100)

    fit = np.array([sv.density(x, novel_pc_distances, kappa) for x in x_vals])

    boots = []
    n_samples = len(novel_pc_distances)
    for _ in range(n_boots):
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        fit_b = [sv.density(x, novel_pc_distances[idxs], kappa) for x in x_vals]
        boots.append(fit_b)

    return fit, x_vals, np.array(boots)

#
# Position stability
#


def prep_position_stability_data(data, method, pairing, n_knots):
    if method == 'cv':
        if pairing == 'all_pairs':
            distances = ed.paired_activity_centroid_distance_to_reward(data)
            distances = distances.dropna()
            fit_data = np.array(distances[['first', 'second']])
        elif 'skip_one' in pairing:
            if 'npc' in pairing:
                distances = ed.tripled_activity_centroid_distance_to_reward(data, prev_imaged=True)
                distances = distances.dropna(subset=['first', 'third'])
                distances = distances[np.isnan(distances['second'])]
            else:
                distances = ed.tripled_activity_centroid_distance_to_reward(data, prev_imaged=False)
                distances = distances.dropna(subset=['first', 'third'])
            fit_data = np.array(distances[['first', 'third']])
        elif 'skip_two' in pairing:
            if 'npc' in pairing:
                distances = ed.quad_activity_centroid_distance_to_reward(data, prev_imaged=True)
                distances = distances.dropna(subset=['first', 'fourth'])
                distances = distances[np.isnan(distances['third'])]
            else:
                distances = ed.quad_activity_centroid_distance_to_reward(data, prev_imaged=False)
                distances = distances.dropna(subset=['first', 'fourth'])
            fit_data = np.array(distances[['first', 'fourth']])
        elif 'skip_three' in pairing:
            if 'npc' in pairing:
                distances = ed.quint_activity_centroid_distance_to_reward(data, prev_imaged=True)
                distances = distances.dropna(subset=['first', 'fifth'])
                distances = distances[np.isnan(distances['fourth'])]
            else:
                distances = ed.quint_activity_centroid_distance_to_reward(data, prev_imaged=False)
                distances = distances.dropna(subset=['first', 'fifth'])
            fit_data = np.array(distances[['first', 'fifth']])
        elif 'skip_four' in pairing:
            if 'npc' in pairing:
                distances = ed.hex_activity_centroid_distance_to_reward(data, prev_imaged=True)
                distances = distances.dropna(subset=['first', 'sixth'])
                distances = distances[np.isnan(distances['fifth'])]
            else:
                distances = ed.hex_activity_centroid_distance_to_reward(data, prev_imaged=False)
                distances = distances.dropna(subset=['first', 'sixth'])
            fit_data = np.array(distances[['first', 'sixth']])
        # knots = np.linspace(-np.pi, np.pi, n_knots)
        knots = np.array([np.percentile(fit_data[:, 0], x) for x in np.linspace(0, 100, n_knots)])
        knots[0] = -np.pi
        knots[-1] = np.pi
    elif method == 'si':
        pass
    elif method is None:
        # Assume data is already formated
        fit_data = data.copy()
        knots = np.array([np.percentile(fit_data[:, 0], x) for x in np.linspace(0, 100, n_knots)])
        knots[0] = -np.pi
        knots[-1] = np.pi

    return knots, fit_data


def determine_position_stability_parameters(
        data, initial_p=None, method='cv', pairing='consecutive', n_knots=20, repeats=20,
        savedir='/analysis/Jeff/Df16A/Df_remap_paper/data/enrichment_model',
        options=None):

    if options is None:
        options = {}

    if initial_p is None:
        initial_p = (0, 0)

    knots, fit_data = prep_position_stability_data(data, method, pairing, n_knots)

    repeat_range = range(len(fit_data))
    repeat_perm = np.random.permutation(repeat_range)
    repeat_indices = []
    for idx in repeat_perm[:repeats]:
        repeat_indices.append(
            np.hstack([np.setdiff1d(repeat_perm, [idx]), idx]))
    # repeat_indices = [np.random.permutation(repeat_range) for _ in range(repeats)]

    time_str = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
    savedir = os.path.join(savedir, 'bk_optimize_{}'.format(time_str))
    os.mkdir(savedir)

    print 'Saving data to: ' + savedir

    def objective(log_penalties):
        logL, b, k = splines.bk_cross_val(
            fit_data, knots, np.exp(log_penalties), repeats=repeat_indices,
            single_value_test=True)

        # Status PDFs
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(fit_data[:, 0], b, '.')
        axs[0].set_title('b')
        axs[1].plot(fit_data[:, 0], k, '.')
        axs[1].set_title('k')
        fig.suptitle('ps={}, logL={}'.format(log_penalties, logL))

        time_str = datetime.datetime.strftime(
            datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
        fn = 'bk_optimize_results_{}.pdf'.format(time_str)
        fig.savefig(os.path.join(savedir, fn), format='pdf')
        plt.close('all')

        print '{}: {}'.format(log_penalties, logL)

        return -logL

    # result = scipy.optimize.minimize(objective, initial_p, method='BFGS', options=options)
    result = scipy.optimize.fmin_bfgs(objective, initial_p, **options)

    print "Optimized penalties: " + str(result)


def determine_position_stability_offset(
        data, initial_p=None, method='cv', pairing='consecutive', n_knots=20,
        repeats=20,
        savedir='/analysis/Jeff/Df16A/Df_remap_paper/data/enrichment_model',
        fit_method='auto', options=None):

    if options is None:
        options = {}

    if initial_p is None:
        initial_p = 0

    knots, fit_data = prep_position_stability_data(data, method, pairing, n_knots)

    repeat_range = range(len(fit_data))
    repeat_perm = np.random.permutation(repeat_range)
    repeat_indices = []
    for idx in repeat_perm[:repeats]:
        repeat_indices.append(
            np.hstack([np.setdiff1d(repeat_perm, [idx]), idx]))
    # repeat_indices = [np.random.permutation(repeat_range) for _ in range(repeats)]

    time_str = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
    savedir = os.path.join(savedir, 'b_optimize_{}'.format(time_str))
    os.mkdir(savedir)

    data = {}
    data['log_p'] = []
    data['err'] = []
    pkl.dump(data, open(os.path.join(savedir, 'data.pkl'), 'w'), pkl.HIGHEST_PROTOCOL)

    print 'Saving data to: ' + savedir

    def objective(log_penalty):
        err, b = splines.b_cross_val(
            fit_data, knots, np.exp(log_penalty), repeats=repeat_indices,
            single_value_test=True)

        # Status PDFs
        fig, ax = plt.subplots(1, 1)
        ax.plot(fit_data[:, 0], b, '.')
        ax.set_title('b')
        fig.suptitle('p={}, err={}'.format(log_penalty, err))

        time_str = datetime.datetime.strftime(
            datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
        fn = 'b_optimize_results_{}.pdf'.format(time_str)
        fig.savefig(os.path.join(savedir, fn), format='pdf')
        plt.close('all')

        data = pkl.load(open(os.path.join(savedir, 'data.pkl'), 'r'))
        data['log_p'].append(log_penalty)
        data['err'].append(err)
        pkl.dump(data, open(os.path.join(savedir, 'data.pkl'), 'w'), pkl.HIGHEST_PROTOCOL)

        print '{}: {}'.format(log_penalty, err)

        return err

    if fit_method == 'auto':
        # result = scipy.optimize.minimize(objective, initial_p, method='BFGS', options=options)
        result = scipy.optimize.fmin_bfgs(objective, initial_p, **options)
    else:
        errs = []
        for log_p in fit_method:
            errs.append(objective(log_p))
        result = [fit_method[np.argmin(errs)]]

    all_data = pkl.load(open(os.path.join(savedir, 'data.pkl'), 'r'))
    fig, ax = plt.subplots(1, 1)
    ax.plot(all_data['log_p'], all_data['err'], '.')
    ax.set_ylabel('err')
    ax.set_xlabel('log_p')
    ax.set_title('b')
    fig.savefig(os.path.join(savedir, 'final_fits.pdf'), format='pdf')
    plt.close('all')

    print "Optimized penalties: " + str(result)

    return result


def determine_position_stability_kappa(
        data, theta_b, initial_p=None, method='cv', pairing='consecutive', n_knots=20, repeats=20,
        savedir='/analysis/Jeff/Df16A/Df_remap_paper/data/enrichment_model', fit_method='auto',
        options=None):

    if options is None:
        options = {}

    if initial_p is None:
        initial_p = 0

    knots, fit_data = prep_position_stability_data(data, method, pairing, n_knots)

    repeat_range = range(len(fit_data))
    repeat_perm = np.random.permutation(repeat_range)
    repeat_indices = []
    for idx in repeat_perm[:repeats]:
        repeat_indices.append(
            np.hstack([np.setdiff1d(repeat_perm, [idx]), idx]))
    # repeat_indices = [np.random.permutation(repeat_range) for _ in range(repeats)]

    time_str = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
    savedir = os.path.join(savedir, 'k_optimize_{}'.format(time_str))
    os.mkdir(savedir)

    data = {}
    data['log_p'] = []
    data['nLL'] = []
    pkl.dump(data, open(os.path.join(savedir, 'data.pkl'), 'w'), pkl.HIGHEST_PROTOCOL)

    print 'Saving data to: ' + savedir

    def objective(log_penalty):
        logL, k = splines.k_cross_val(
            fit_data, knots, np.exp(log_penalty), theta_b,
            repeats=repeat_indices, single_value_test=True)

        # Status PDFs
        fig, ax = plt.subplots(1, 1)
        ax.plot(fit_data[:, 0], k, '.')
        ax.set_title('k')
        fig.suptitle('p={}, neg_logL={}'.format(log_penalty, - logL))

        time_str = datetime.datetime.strftime(
            datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
        fn = 'k_optimize_results_{}.pdf'.format(time_str)
        fig.savefig(os.path.join(savedir, fn), format='pdf')
        plt.close('all')

        data = pkl.load(open(os.path.join(savedir, 'data.pkl'), 'r'))
        data['log_p'].append(log_penalty)
        data['nLL'].append(-logL)
        pkl.dump(data, open(os.path.join(savedir, 'data.pkl'), 'w'), pkl.HIGHEST_PROTOCOL)

        print '{}: {}'.format(log_penalty, - logL)

        return - logL

    if fit_method == 'auto':
        # result = scipy.optimize.minimize(objective, initial_p, method='BFGS', options=options)
        result = scipy.optimize.fmin_bfgs(objective, initial_p, **options)
    else:
        nLL = []
        for log_p in fit_method:
            nLL.append(objective(log_p))
        result = [fit_method[np.argmin(nLL)]]

    all_data = pkl.load(open(os.path.join(savedir, 'data.pkl'), 'r'))
    fig, ax = plt.subplots(1, 1)
    ax.plot(all_data['log_p'], all_data['nLL'], '.')
    ax.set_ylabel('negative LL')
    ax.set_xlabel('log_p')
    ax.set_title('k')
    fig.savefig(os.path.join(savedir, 'final_fits.pdf'), format='pdf')
    plt.close('all')

    print "Optimized penalties: " + str(result)

    return result


def return_position_stability_parameters(
        data, penalties, method='cv', pairing='consecutive', n_knots=20, n_boots=0):

    knots, fit_data = prep_position_stability_data(data, method, pairing, n_knots)

    theta_b, N_b, theta_k, N_k, ll = splines.fit_transitions(
        fit_data, knots, penalties)

    b = np.dot(N_b, theta_b)
    k = splines.get_k(theta_k, N_k)

    boots_b, boots_k, boots_data = [], [], []

    for _ in range(n_boots):
        idxs = np.random.choice(len(fit_data), size=len(fit_data), replace=True)
        theta_b, N_b, theta_k, N_k, ll = splines.fit_transitions(
            fit_data[idxs], knots, penalties)
        boot_b = np.dot(N_b, theta_b)
        boot_k = splines.get_k(theta_k, N_k)
        boots_b.append(boot_b)
        boots_k.append(boot_k)
        boots_data.append(fit_data[idxs])

    return b, k, fit_data[:, 0], fit_data[:, 1], np.array(boots_b), np.array(boots_k), np.array(boots_data)


def return_position_stability_offset(
        data, penalty, method='cv', pairing='consecutive', n_knots=20, n_boots=0):

    knots, fit_data = prep_position_stability_data(data, method, pairing, n_knots)

    theta_b, N_b, ll = splines.fit_biases(fit_data, knots, penalty)

    boots_theta_b = []

    for _ in range(n_boots):
        idxs = np.random.choice(len(fit_data), size=len(fit_data), replace=True)
        boot_theta_b, boot_N_b, ll = splines.fit_biases(
            fit_data[idxs], knots, penalty)
        boots_theta_b.append(boot_theta_b)

    return theta_b, np.array(boots_theta_b)


def return_position_stability_kappa(
        data, theta_b, penalty, method='cv', pairing='consecutive', n_knots=20, n_boots=0):

    knots, fit_data = prep_position_stability_data(data, method, pairing, n_knots)

    theta_k, N_k, ll = splines.fit_kappas(
        fit_data, knots, penalty, theta_b)

    boots_theta_k = []

    for _ in range(n_boots):
        idxs = np.random.choice(len(fit_data), size=len(fit_data), replace=True)
        boot_theta_k, boot_N_k, ll = splines.fit_kappas(
            fit_data[idxs], knots, penalty, theta_b)
        boots_theta_k.append(boot_theta_k)

    return theta_k, np.array(boots_theta_k)


if __name__ == '__main__':

    raw_data, data = ed.load_data('wt')

    # determine_position_recurrence_parameters(
    #     data, -4, 0, method='cv', max_iter=5000, thresh=1e-10, repeats=100)

    # determine_novel_position_parameters(data, 3, 5, method='cv', repeats=2000)

    # determine_position_stability_parameters(
    #     data, [0, -2], method='cv', n_knots=10, repeats=20,
    #     options={'epsilon': 0.1})
    result = determine_position_stability_offset(
        data, initial_p=0, method='cv', pairing='skip_one', n_knots=20, repeats=20, options=None)
    # params = pkl.load(open('Df_model_params.pkl'))
    # theta_b = params['skip_position_stability']['theta_b']
    # n_knots = params['skip_position_stability']['n_knots']
    # result = determine_position_stability_kappa(
    #     data, theta_b, initial_p=None, method='cv', n_knots=n_knots, repeats=100,
    #     savedir='/analysis/Jeff/Df16A/Df_remap_paper/data/enrichment_model',
    #     options=None)

    # Df
    # b, k, x, y, boots_b, boots_k, boots_data = return_position_stability_parameters(
    #     data, np.exp([4.49710256e-06, 3.50258634e-06]), method='cv', n_knots=10, n_boots=200)

    # WT
    # b, k, x, y, boots_b, boots_k, boots_data = return_position_stability_parameters(
    #     data, np.exp([-1.77038246e-07, -4.64224294e-07]), method='cv', n_knots=10, n_boots=200, options={'epsilon': 0.1}, initial_p=(0, -2))
    set_trace()
