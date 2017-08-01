import matplotlib.pyplot as plt
import cPickle as pkl
import numpy as np
import seaborn.apionly as sns

from lab.plotting import histogram


def enrichment(positions):
    distances = np.abs(positions[np.isfinite(positions)])
    return np.mean(distances), np.std(distances) / np.sqrt(len(distances))


def calc_enrichment(pos, masks):
        enrich = []
        for rep_positions, rep_masks in zip(pos, masks):
            enrich.append(
                [np.pi / 2 - enrichment(iter_positions[iter_mask])[0]
                    for iter_positions, iter_mask in zip(
                        rep_positions, rep_masks)])
        return enrich


def calc_final_distributions(pos, masks):
    final_dist = []
    for rep_positions, rep_masks in zip(pos, masks):
        final_dist.extend(rep_positions[-1][rep_masks[-1]].tolist())
    return final_dist


def plot_enrichment(ax, enrichment, color, title='', rad=True):
    ax.plot(range(9), np.mean(enrichment, axis=0), color=color)
    ax.plot(range(9), np.percentile(enrichment, 5, axis=0), ls='--',
            color=color)
    ax.plot(range(9), np.percentile(enrichment, 95, axis=0), ls='--',
            color=color)
    ax.fill_between(
        range(9), np.percentile(enrichment, 5, axis=0),
        np.percentile(enrichment, 95, axis=0), facecolor=color, alpha=0.5)

    sns.despine(ax=ax)
    ax.tick_params(length=3, pad=2, direction='out')
    ax.set_xlim(-0.5, 8.5)
    if rad:
        ax.set_ylim(-0.15, 0.5)
        ax.set_ylabel('Enrichment (rad)')
    else:
        ax.set_ylim(-0.15, 0.10 * 2 * np.pi)
        y_ticks = np.array(['0', '0.05', '0.10'])
        ax.set_yticks(y_ticks.astype('float') * 2 * np.pi)
        ax.set_yticklabels(y_ticks)
        ax.set_ylabel('Enrichment (fraction of belt)')
    ax.set_xlabel("Iteration ('session' #)")
    ax.set_title(title)


def plot_final_distributions(
        ax, final_dists, colors, labels=None, title='', rad=True):
    if labels is None:
        labels = [None] * len(final_dists)
    for final_dist, color, label in zip(final_dists, colors, labels):
        histogram(
            ax, final_dist, bins=50, range=(-np.pi, np.pi),
            color=color, filled=False, plot_mean=False, normed=True,
            label=label)
    ax.tick_params(length=3, pad=2, direction='out')
    ax.axvline(ls='--', color='0.3')
    ax.set_xlim(-np.pi, np.pi)
    if rad:
        ax.set_xlabel('Distance from reward (rad)')
    else:
        ax.set_xlabel('Distance from reward (fraction of belt)')
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax.set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])

    ax.set_ylim(0, 0.3)
    ax.set_ylabel('Normalized density')
    ax.set_title(title)


def plot_parameters(axs, model, enrich):

    positions = np.linspace(-np.pi, np.pi, 1000)

    bs, ks = model.shift_mean_var(positions)
    recur = model.recur_by_position(positions)

    axs[0].plot(positions, recur)
    axs[0].set_xlim(-np.pi, np.pi)
    axs[0].set_ylim(0., 1.)
    axs[0].set_xlabel('Position')
    axs[0].set_ylabel('Recurrence probability')

    axs[1].plot(positions, bs)
    axs[1].set_xlim(-np.pi, np.pi)
    axs[1].set_xlabel('Position')
    axs[1].set_ylabel('Offset')

    axs[2].plot(positions, 1 / ks)
    axs[2].set_xlim(-np.pi, np.pi)
    axs[2].set_xlabel('Position')
    axs[2].set_ylabel('Variance')

    axs[3].plot(range(9), np.mean(enrich, axis=0), color='b')
    axs[3].fill_between(
        range(9), np.percentile(enrich, 5, axis=0),
        np.percentile(enrich, 95, axis=0), facecolor='b', alpha=0.5)
    axs[3].set_xlabel('Iteration')
    axs[3].set_ylabel('Enrichment (rad)')


def plot_models(
        models, model_labels=None, n_cells=1000, n_runs=100, n_iterations=8):

    if model_labels is None:
        model_labels = ['Model {}'.format(idx) for idx in range(len(models))]

    fig, axs = plt.subplots(4, len(models), figsize=(10, 10))

    models[0].initialize(n_cells=n_cells)
    for model in models[1:]:
        model.initialize_like(models[0])

    initial_mask = models[0].mask
    initial_positions = models[0].positions

    masks = []
    positions = []
    enrichment = []

    for model, model_axs in zip(models, axs.T):
        masks.append([])
        positions.append([])
        for _ in range(n_runs):
            model.initialize(
                initial_mask=initial_mask, initial_positions=initial_positions)

            model.run(n_iterations)

            masks[-1].append(model._masks)
            positions[-1].append(model._positions)

        enrichment.append(calc_enrichment(positions[-1], masks[-1]))

        plot_parameters(model_axs, model, enrichment[-1])

    for ax in axs[:, 1:].flat:
        ax.set_ylabel('')
    for ax in axs[:2, :].flat:
        ax.set_xlabel('')

    for label, ax in zip(model_labels, axs[0]):
        ax.set_title(label)

    offset_min, offset_max = np.inf, -np.inf
    for ax in axs[1]:
        offset_min = min(offset_min, ax.get_ylim()[0])
        offset_max = max(offset_max, ax.get_ylim()[1])
    for ax in axs[1]:
        ax.set_ylim(offset_min, offset_max)

    var_min, var_max = np.inf, -np.inf
    for ax in axs[2]:
        var_min = min(var_min, ax.get_ylim()[0])
        var_max = max(var_max, ax.get_ylim()[1])
    for ax in axs[2]:
        ax.set_ylim(var_min, var_max)

    enrich_min, enrich_max = np.inf, -np.inf
    for ax in axs[3]:
        enrich_min = min(enrich_min, ax.get_ylim()[0])
        enrich_max = max(enrich_max, ax.get_ylim()[1])
    for ax in axs[3]:
        ax.set_ylim(enrich_min, enrich_max)

    return fig


if __name__ == '__main__':
    import enrichment_model as em
    import enrichment_model_theoretical as emt

    params_path_A = '/analysis/Jeff/Df16A/Df_remap_paper_v2/data/enrichment_model/Df_model_params_A.pkl'
    params_path_B = '/analysis/Jeff/Df16A/Df_remap_paper_v2/data/enrichment_model/Df_model_params_B.pkl'
    params_path_C = '/analysis/Jeff/Df16A/Df_remap_paper_v2/data/enrichment_model/Df_model_params_C.pkl'

    #
    # WT to theoretical
    #

    # WT_params_path = params_path_C

    # WT_params = pkl.load(open(WT_params_path, 'r'))

    # WT_model = em.EnrichmentModel2(**WT_params)
    # recur_model = emt.EnrichmentModel2_recur(
    #     kappa=1, span=0.8, mean_recur=0.4, **WT_params)
    # offset_model = emt.EnrichmentModel2_offset(alpha=0.25, **WT_params)
    # var_model = emt.EnrichmentModel2_var(
    #     kappa=1, alpha=10, mean_k=3, **WT_params)
    # models = [WT_model, recur_model, offset_model, var_model]
    # model_labels = ['WT model', 'Stable recurrence', 'Shift towards reward',
    #                 'Stable position']

    params_A = pkl.load(open(params_path_A, 'r'))
    params_B = pkl.load(open(params_path_B, 'r'))
    params_C = pkl.load(open(params_path_C, 'r'))

    model_A = em.EnrichmentModel2(**params_A)
    model_B = em.EnrichmentModel2(**params_B)
    model_C = em.EnrichmentModel2(**params_C)

    models = [model_A, model_B, model_C]
    model_labels = ['A', 'B', 'C']

    fig = plot_models(
        models, model_labels, n_cells=1000, n_runs=100, n_iterations=8)

    fig.savefig('Df_model_parameters.pdf')

    from pudb import set_trace
    set_trace()
