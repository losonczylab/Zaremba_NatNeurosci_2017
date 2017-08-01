"""Enrichment model."""

import numpy as np
import cPickle as pkl
from scipy.interpolate import interp1d
from numpy.random import vonmises as vm
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

import enrichment_model_plotting as emp
from lab.misc import splines


class EnrichmentModel(with_metaclass(ABCMeta, object)):
    """Base Enrichment model."""

    def __init__(self, pc_fraction, **kwargs):
        """Enrichment model

        Parameters
        ----------
        p_on : float
            Probability of a place cell to turn off to on

        """

        if not hasattr(self, 'params'):
            self.params = {}

        self.params['pc_fraction'] = pc_fraction
        self.params.update(kwargs)

        self._masks = None
        self._positions = None

        self._interp = None
        self._interp_values = {}

    def copy(self):
        new_model = type(self)(**self.params)
        new_model._masks = self._masks
        new_model._positions = self._positions
        new_model._interp = self._interp
        new_model._interp_vales = self._interp_values.copy()

        return new_model

    def reset(self):
        self._masks = None
        self._positions = None

    def initialize(
            self, initial_positions=None, initial_mask=None, n_cells=1000,
            flat_initial=True, flat_tol=1e-4):

        if initial_positions is None or initial_mask is None:
            if not flat_initial:
                raise NotImplemented

            if self._interp is None:
                initial_mask = np.random.random(n_cells) < self.params['pc_fraction']
            else:
                s = self._interp_values['pc_fraction']
                interp = interp1d(
                    [0, 1],
                    [self.params['pc_fraction'], self._interp.params['pc_fraction']])
                initial_mask = np.random.random(n_cells) < interp(s)

            is_flat = False
            while not is_flat:
                initial_positions = (np.random.random(n_cells) * 2 * np.pi) - np.pi
                flatness = np.abs(np.mean(np.abs(
                    initial_positions[initial_mask])) - np.pi / 2)
                is_flat = flatness < flat_tol

        self._positions = [initial_positions]
        self._masks = [initial_mask]

    def initialize_like(self, other):
        self._positions = [other._positions[0]]
        self._masks = [other._masks[0]]

    def interpolate(self, other, pc_fraction=0):
        self._interp = other
        self._interp_values['pc_fraction'] = pc_fraction

    def un_interpolate(self):
        self._interp = None
        self._interp_values['pc_fraction'] = 0

    @abstractmethod
    def run(self, iterations=1):
        pass

    @property
    def n_cells(self):
        return len(self.mask)

    @property
    def mask(self):
        return self._masks[-1]

    @property
    def positions(self):
        return self._positions[-1]


class EnrichmentModel2(EnrichmentModel):
    """Enrichment model #2.

    All cells have an underlying tuning each iteration that shifts with a
    position-dependent probability.
    Some non-place cells become place cells.
    Some place-cells become non-place cells with a position-dependent
    probability.

    """

    def __init__(
            self, p_on, position_recurrence, position_stability, **kwargs):

        if not hasattr(self, 'params'):
            self.params = {}

        self.params['p_on'] = p_on
        self.params['position_recurrence'] = position_recurrence
        self.params['position_stability'] = position_stability

        self.recur_by_position = self._recur_by_position
        self.shift_mean_var = self._shift_mean_var

        super(EnrichmentModel2, self).__init__(**kwargs)

    def copy(self):

        new_model = super(EnrichmentModel2, self).copy()

        if self.recur_by_position == self._recur_by_position_flat:
            new_model.recur_by_position = new_model._recur_by_position_flat

        if self.shift_mean_var == self._shift_mean_var_flat:
            new_model.shift_mean_var = new_model._shift_mean_var_flat

        return new_model

    def flatten(self):

        self.recur_by_position = self._recur_by_position_flat
        self.shift_mean_var = self._shift_mean_var_flat

    def un_flatten(self):

        self.recur_by_position = self._recur_by_position
        self.shift_mean_var = self._shift_mean_var

    def npc_to_pc(self):
        """Given a mask, returns the indices of cells that flip from non-PC to
        PC

        """

        non_pcs = ~self.mask
        arange = np.arange(self.n_cells)

        if self._interp is None:
            turn_on = np.random.rand(non_pcs.sum()) < self.params['p_on']
        else:
            s = self._interp_values['on']
            interp = interp1d(
                [0, 1], [self.params['p_on'], self._interp.params['p_on']])
            turn_on = np.random.rand(non_pcs.sum()) < interp(s)

        return arange[non_pcs][turn_on]

    def _recur_by_position(self, positions):
        recur_theta = self.params['position_recurrence']['theta']
        recur_knots = np.linspace(-np.pi, np.pi, self.params['position_recurrence']['n_knots'])
        recur_spline = splines.CyclicSpline(recur_knots)
        N = recur_spline.design_matrix(positions)

        return splines.prob(recur_theta, N)

    def pc_to_pc(self):

        pcs = self.mask
        arange = np.arange(self.n_cells)

        self_p = self.recur_by_position(self.positions)

        if self._interp is None:
            p = self_p
        else:
            s = self._interp_values['recur']

            other_p = self._interp.recur_by_position(self.positions)

            p = interp1d([0, 1], [self_p, other_p], axis=0)(s)

        turn_off = np.random.rand(self.n_cells) > p

        return arange[pcs & turn_off]

    def _recur_by_position_flat(self, positions):

        p = self._recur_by_position(positions)
        p.fill(p.mean())

        return p

    def _shift_mean_var_flat(self, positions):

        bs, ks = self._shift_mean_var(positions)

        bs.fill(bs.mean())
        ks.fill(ks.mean())

        return bs, ks

    def _shift_mean_var(self, positions):
        shift_knots = self.params['position_stability']['all_pairs']['knots']
        shift_spline = splines.CyclicSpline(shift_knots)

        shift_theta_b = self.params['position_stability']['all_pairs']['theta_b']
        shift_theta_k = self.params['position_stability']['all_pairs']['theta_k']

        N = shift_spline.design_matrix(positions)
        bs = np.dot(N, shift_theta_b)
        ks = splines.get_k(shift_theta_k, N)

        return bs, ks

    def pc_shift(self):

        self_bs, self_ks = self.shift_mean_var(self.positions)

        if self._interp is None:
            bs = self_bs
            ks = self_ks

        else:
            s_b = self._interp_values['shift_b']
            s_k = self._interp_values['shift_k']
            other_bs, other_ks = self._interp.shift_mean_var(self.positions)

            bs = (1 - s_b) * self_bs + s_b * other_bs
            ks = (1 - s_k) * self_ks + s_k * other_ks

        new_positions = []
        for x, b, k in zip(self.positions, bs, ks):
            d = vm(0, k)
            y = d + x + b
            new_positions.append(y)
        new_positions = np.array(new_positions)
        new_positions[new_positions < -np.pi] += 2 * np.pi
        new_positions[new_positions >= np.pi] -= 2 * np.pi

        return new_positions

    def interpolate(
            self, other, on=0, recur=0, shift_b=0, shift_k=0, **kwargs):

        self._interp_values['on'] = on
        self._interp_values['recur'] = recur
        self._interp_values['shift_b'] = shift_b
        self._interp_values['shift_k'] = shift_k

        super(EnrichmentModel2, self).interpolate(other, **kwargs)

    def un_interpolate(self):

        self._interp_values['on'] = 0
        self._interp_values['recur'] = 0
        self._interp_values['shift_b'] = 0
        self._interp_values['shift_k'] = 0

        super(EnrichmentModel2, self).un_interpolate()

    def run(self, iterations=1):

        for x in range(iterations):
            new_mask = self.mask.copy()

            # Shift all PC positions to new location
            new_positions = self.pc_shift()

            # Decide which off PCs will turn on
            turn_on = self.npc_to_pc()

            # Decide which PCs will stay on
            turn_off = self.pc_to_pc()

            # Put it all together
            new_mask[turn_on] = True
            new_mask[turn_off] = False

            self._masks.append(new_mask)
            self._positions.append(new_positions)

    def show_parameters(self):

        positions = np.linspace(-np.pi, np.pi, 1000)

        bs, ks = self.shift_mean_var(positions)
        recur = self.recur_by_position(positions)

        fig, axs = plt.subplots(1, 3, figsize=(10, 6))

        axs[0].plot(positions, recur)
        axs[0].set_xlim(-np.pi, np.pi)
        axs[0].set_ylim(0., 1.)
        axs[0].set_xlabel('Position')
        axs[0].set_ylabel('Recurrence probability')
        axs[0].set_title('Recurrence by position')

        axs[1].plot(positions, bs)
        axs[1].set_xlim(-np.pi, np.pi)
        axs[1].set_xlabel('Position')
        axs[1].set_ylabel('Offset')
        axs[1].set_title('Mean shift')

        axs[2].plot(positions, ks)
        axs[2].set_xlim(-np.pi, np.pi)
        axs[2].set_xlabel('Position')
        axs[2].set_ylabel('Variance')
        axs[2].set_title('Shift variance')

        return fig


class EnrichmentModel4(EnrichmentModel2):
    """Enrichment model #4.

    All cells have an underlying tuning each iteration that shifts with a
    position-dependent probability.
    All cells are place cells with a position-dependent probability determined
    by their latent or expressed spatial tuning on the previous iteration.

    """

    def pc_on(self):

        self_p = self.recur_by_position(self.positions)

        if self._interp is None:
            p = self_p
        else:
            s = self._interp_values['recur']

            other_p = self._interp.recur_by_position(self.positions)

            p = interp1d([0, 1], [self_p, other_p], axis=0)(s)

        return np.random.rand(self.n_cells) < p

    def run(self, iterations=1):

        for _ in range(iterations):

            # Shift all PC positions to new location
            new_positions = self.pc_shift()

            # Determine new mask
            new_mask = self.pc_on()

            self._masks.append(new_mask)
            self._positions.append(new_positions)


class EnrichmentModel4_2(EnrichmentModel4):
    """Enrichment model #4-2.

    All cells have an underlying tuning each iteration that shifts with a
    position-dependent probability.
    All cells are place cells with a position-dependent probability determined
    by their latent or expressed spatial tuning for the current iteration
    (after shifting).

    """
    def run(self, iterations=1):

        for _ in range(iterations):

            # Shift all PC positions to new location
            new_positions = self.pc_shift()

            self._positions.append(new_positions)

            # Determine new mask
            new_mask = self.pc_on()

            self._masks.append(new_mask)


def novel_pc_pos(s, n_rois=1):
    fit = novel_fit_interp(s)
    cdf = np.cumsum(fit) / np.sum(fit)
    place_interp = interp1d(cdf, novel_x_vals, assume_sorted=True, bounds_error=False, fill_value=-np.pi)
    return place_interp(np.random.random(n_rois))


def novel_pc_pos_random(n_rois):
    return (np.random.random(n_rois) * 2 * np.pi) - np.pi
