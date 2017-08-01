"""Statistics helper functions"""

import numpy as np
from scipy.special import ndtr
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import ttest_ind, levene, kstest, ks_2samp, mannwhitneyu
import itertools as it
import pandas as pd


def poisson_binomial_distribution(probabilities):
    """Return the probability mass function of a poisson binomial distribution
    with the given probabilities

    """

    Pr = np.empty(len(probabilities) + 1)

    Pr[0] = np.product(1.0 - np.array(probabilities))

    def T(i):
        return sum([(prob / (1 - prob)) ** i for prob in probabilities])

    for k in range(1, len(probabilities) + 1):
        Pr[k] = 1.0 / k * sum(
            [(-1) ** (i - 1) * Pr[k - i] * T(i) for i in range(1, k + 1)])

    return Pr


def z_test(x1, x2, n1, n2, d=0):
    """Compare two binomial frequency counts.

    Returns the z-statistic and two-tailed p-value

    Parameters
    ----------
    x1, x2 : int
        Counts for each sample.
    n1, n2 : int
        Sample sizes.
    d : float
        Expected difference.

    """
    f = (x1 + x2) / float(n1 + n2)

    z = ((x1 / float(n1)) - (x2 / float(n2)) - d) \
        / np.sqrt(f * (1 - f) * (1 / float(n1) + 1 / float(n2)))

    return z, 2 * ndtr(-z)


def full_anova(
        data, ind_vars=None, dep_var='value', file_path=None,
        ignore_shuffle=False):
    """Run an ANOVA and post-hoc tests on a dataframe.

    Parameters
    ----------
    data : dict or pd.DataFrame
        Data to be analyzed. Either a single long-form dataframe (i.e. genotype
        or task condition as a categorical column) or data saved using
        lab.misc.save_data
    ind_vars : list or None
        List of columns names that are independent variables in the model. If
        'None' attempt to infer which columns are categorical.
    dep_var : str
        Name of column containing dependent variable.
    file_path : str or None
        Path to save results to. If 'None', just print to standard output.
    ignore_shuffle : bool
        If True, drop shuffle data before ANOVA. Always include in 1v1 tests.

    """
    data = prep_data(data)

    if ignore_shuffle:
        anova_data = data[data['grp'] != 'shuffle']
    else:
        anova_data = data

    if ind_vars is None:
        ind_vars = _infer_ind_vars(data)

    def xprint(xstr):
        if file_path is None:
            print(xstr)
        else:
            with open(file_path, 'a') as f:
                f.write(str(xstr))
                f.write('\n')

    formula = '{}~C({})'.format(dep_var, ind_vars[0])
    for var in ind_vars[1:]:
        formula += '*C({})'.format(var)

    xprint("# Formula: {}".format(formula))
    for ind_var in ind_vars:
        xprint("# {}: {}".format(ind_var, set(anova_data[ind_var])))
    xprint("")

    lm = ols(formula, anova_data).fit()

    xprint(lm.summary2())

    xprint(anova_lm(lm))

    xprint("\n# Post-hoc t-tests\n# No correction for multiple comparisons, means+-sem\n")
    for var1, var2 in it.permutations([None] + list(ind_vars), 2):
        if var2 is None:
            continue
        var1_vals = data[var1].unique() if var1 is not None else [None]
        for var1_val in var1_vals:
            for val1, val2 in it.combinations(data[var2].unique(), 2):
                if var1 is None:
                    data1 = data.ix[
                        (data[var2] == val1), dep_var].dropna()
                    data2 = data.ix[
                        (data[var2] == val2), dep_var].dropna()
                else:
                    data1 = data.ix[
                        (data[var1] == var1_val) &
                        (data[var2] == val1), dep_var].dropna()
                    data2 = data.ix[
                        (data[var1] == var1_val) &
                        (data[var2] == val2), dep_var].dropna()
                str1 = "{} = {}, ".format(
                    var1, var1_val) if var1 is not None else ""
                str2 = "{}: {} ({:.5f}+-{:.5f}, n={}) vs. {} ({:.5f}+-{:.5f}, n={})".format(
                    var2, val1, data1.mean(), data1.sem(), data1.shape[0],
                    val2, data2.mean(), data2.sem(), data2.shape[0])
                post_hoc_str = str1 + str2
                xprint(post_hoc_str)
                xprint('-' * len(post_hoc_str))

                # Equal variance
                levene_result = levene(data1, data2)
                equal_var = levene_result[1] > 0.05
                xprint("{} Levene test for equal variance: W={}, p={}".format(
                    '*' if not equal_var else ' ', levene_result[0], levene_result[1]))

                # Normality
                norm_1_result = kstest(
                    data1, 'norm', args=(data1.mean(), data1.std()))
                norm_2_result = kstest(
                    data2, 'norm', args=(data2.mean(), data2.std()))
                val1_normal = norm_1_result[1] > 0.05
                val2_normal = norm_2_result[1] > 0.05
                xprint(
                    "{} KS Normality test; {}: D={}, p={}; {}: D={}, p={}".format(
                        '*' if not val1_normal or not val2_normal else ' ',
                        val1, norm_1_result[0], norm_1_result[1],
                        val2, norm_2_result[0], norm_2_result[1]))

                # t-test
                if val1_normal and val2_normal:
                    ttest_result = ttest_ind(data1, data2, equal_var=equal_var)
                    xprint("  {}(t={}, p={})".format(
                        'Ttest_ind' if equal_var else 'WelchsTtest_ind',
                        ttest_result[0], ttest_result[1]))
                else:
                    mann_whitney_result = mannwhitneyu(data1, data2)
                    xprint("  Mann-WhitneyU(U={}, p={}".format(
                        mann_whitney_result[0], mann_whitney_result[1]))

                # K-S test
                ks_result = ks_2samp(data1, data2)
                xprint('  Ks_2samp(D={}, p={})'.format(
                    ks_result[0], ks_result[1]))

                xprint("")


def prep_data(data):
    """Convert data from the serialized format to a single long-form dataframe.

    Parameters
    ----------
    data : dict
        Data should be loaded from lab.misc.save_data. If the format is not
        recognized, just pass through unmodified.

    """
    result = data
    if isinstance(data, dict):
        data_dfs = []
        for grp in data:
            if isinstance(data[grp], pd.DataFrame):
                grp_data = data[grp]
                grp_data['grp'] = grp
                data_dfs.append(grp_data)
            elif 'dataframe' in data[grp]:
                grp_data = data[grp]['dataframe']
                grp_data['grp'] = grp
                data_dfs.append(grp_data)
                shuffle_data = data[grp].get('shuffle')
                if shuffle_data is not None:
                    shuffle_data['grp'] = 'shuffle'
                    data_dfs.append(shuffle_data)
            else:
                # Unrecognized data format, skipping
                pass
        result = pd.concat(data_dfs, ignore_index=True)
    return result


def _infer_ind_vars(data, max_unique_vals=3, exclude_cols=('order',)):
    """Attempt to infer which columns are categorical.

    Parameters
    ----------
    data : pd.DataFrame
    max_unique_vals : int
        Maximum number of unique values in a column for it to still be
        considered categorical.
    exclude_cols : sequence of str
        Sequence of column names to never include.

    """
    ind_vars = [col for col in data.columns
                if 1 < len(data[col].unique()) <= max_unique_vals]
    return filter(lambda col: col not in exclude_cols, ind_vars)
