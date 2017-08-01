"""Additional Enrichment models"""

import enrichment_model as em

from scipy.special import i0
import numpy as np


class EnrichmentModel2_recur(em.EnrichmentModel2):

    def __init__(self, kappa=1, span=0.2, mean_recur=None, **kwargs):
        super(EnrichmentModel2_recur, self).__init__(**kwargs)

        self.params['kappa'] = kappa
        self.params['span'] = span
        self.params['mean_recur'] = mean_recur

        self.flatten()

        self.recur_by_position = self._recur_by_position_theoretical

    def copy(self):
        new_model = super(EnrichmentModel2_recur, self).copy()

        new_model.recur_by_position = new_model._recur_by_position_theoretical

        return new_model

    def _recur_by_position_theoretical(self, positions):

        """Model as a vonMises distribution centered at the reward with
        dispersion kappa

        """

        kappa = self.params['kappa']
        span = self.params['span']
        mean_recur = self.params['mean_recur']

        if mean_recur is None:
            mean_recur = self._recur_by_position(positions).mean()

        def vm(x):
            return np.exp(kappa * np.cos(x)) / (2 * np.pi * i0(kappa))

        prob = np.array(map(vm, positions))

        # Normalize prob such that max - min = span and
        # mean = mean of original recur by position

        prob -= prob.min()
        prob /= prob.max()
        prob *= span

        return prob - prob.mean() + mean_recur


class EnrichmentModel2_offset(em.EnrichmentModel2):

    def __init__(self, alpha=1, mean_b=None, **kwargs):
        super(EnrichmentModel2_offset, self).__init__(**kwargs)

        self.params['alpha'] = alpha
        self.params['mean_b'] = mean_b

        self.flatten()

        self.shift_mean_var = self._shift_mean_var_theoretical_offset

    def copy(self):
        new_model = super(EnrichmentModel2_offset, self).copy()

        new_model.shift_mean_var = new_model._shift_mean_var_theoretical_offset

        return new_model

    def _shift_mean_var_theoretical_offset(self, positions):

        bs, ks = self._shift_mean_var_flat(positions)

        alpha = self.params['alpha']
        mean_b = self.params['mean_b']

        if mean_b is None:
            mean_b = bs.mean()

        def sin(x):
            return -1 * alpha * np.sin(x) + mean_b

        offset = np.array(map(sin, positions))

        return offset, ks


class EnrichmentModel2_var(em.EnrichmentModel2):

    def __init__(self, kappa=1, alpha=0.2, mean_k=None, **kwargs):
        super(EnrichmentModel2_var, self).__init__(**kwargs)

        self.params['kappa'] = kappa
        self.params['alpha'] = alpha
        self.params['mean_k'] = mean_k

        self.flatten()

        self.shift_mean_var = self._shift_mean_var_theoretical_var

    def copy(self):
        new_model = super(EnrichmentModel2_var, self).copy()

        new_model.shift_mean_var = new_model._shift_mean_var_theoretical_var

        return new_model

    def _shift_mean_var_theoretical_var(self, positions):

        bs, ks = self._shift_mean_var_flat(positions)

        kappa = self.params['kappa']
        alpha = self.params['alpha']
        mean_k = self.params['mean_k']

        if mean_k is None:
            mean_k = ks.mean()

        def vm(x):
            return alpha * np.exp(kappa * np.cos(x)) / (2 * np.pi * i0(kappa))

        var = np.array(map(vm, positions))

        var -= var.mean()
        var += mean_k

        return bs, var

if __name__ == '__main__':

    import sys
    sys.path.insert(0, '/home/jeff/code/df/reward_remap_paper/supplementals')
    import matplotlib.pyplot as plt
    import cPickle as pkl

    import model_swap_parameters as msp

    WT_params_path = '/analysis/Jeff/reward_remap/data/enrichment_model/WT_model_params_C.pkl'

    WT_params = pkl.load(open(WT_params_path, 'r'))

    WT_model_orig = em.EnrichmentModel2(**WT_params)
    recur_model_orig = EnrichmentModel2_recur(kappa=1, span=0.8, mean_recur=0.4, **WT_params)
    offset_model_orig = EnrichmentModel2_offset(alpha=0.25, **WT_params)
    var_model_orig = EnrichmentModel2_var(kappa=1, alpha=10, mean_k=3, **WT_params)

    WT_model = WT_model_orig.copy()
    th_model = var_model_orig.copy()

    # WT_model.interpolate(Df_model_orig, shift_b=1)
    # Df_model.interpolate(WT_model_orig, shift_b=1)

    WT_model.initialize(n_cells=1000)
    th_model.initialize_like(WT_model)
    initial_mask = WT_model.mask
    initial_positions = WT_model.positions

    WT_masks, WT_positions = [], []
    th_masks, th_positions = [], []

    n_runs = 100

    for _ in range(n_runs):
        WT_model.initialize(initial_mask=initial_mask, initial_positions=initial_positions)
        th_model.initialize(initial_mask=initial_mask, initial_positions=initial_positions)

        WT_model.run(8)
        th_model.run(8)

        WT_masks.append(WT_model._masks)
        WT_positions.append(WT_model._positions)

        th_masks.append(th_model._masks)
        th_positions.append(th_model._positions)

    WT_enrich = msp.calc_enrichment(WT_positions, WT_masks)
    th_enrich = msp.calc_enrichment(th_positions, th_masks)

    msp.plot_enrichment(plt.axes(), WT_enrich, th_enrich, 'b', 'r', 'Enrich')

    from pudb import set_trace; set_trace()

