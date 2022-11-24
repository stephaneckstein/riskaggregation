from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('text', usetex=True)
plt.rcParams.update({
    'font.size': 14,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

# Input parameters
ALPHA = 0.9  # probability level for expected shortfall
D_LIST = [2, 5, 10]  # dimension
C_TYPE = [1, 3]  # 1 normal covariance, 2 cut-off-covariance (not used), 3 conditional covariance
M_TYPE = [1]  # 1 standard normal, 2 student t, 3 uniform ; Currently marginals are assumed to be equal
UB_OR_LB = [-1, 1]  # -1 is lower bound, 1 is upper bound ; determines covariance constraint type
LB_UB_RANGE = 5  # how many different levels of bound to consider
OBJ_H = 2  # we look at expected shortfall
USE_INDEXES = 1  # flexible execution of testcases
QSF = 0  # if QSF == 1, then the conditional bounds will start at 0.75 quantile, and not at 0.5
N_SMALL_YN = 0  # if N_SMALL_YN == 1, then some other number of intervals other than 200 may be used
N_SMALL_VAL = 100  # specifies the number of intervals if N_SMALL_YN = 1

# Plot specific parameters
PLOT_MARGINALS = 0  # whether to plot marginal fits or not
PLOT_OPTIMIZERS = 0  # whether to plot two-dimensional scatter plots of optimizers
PLOT_SUM_DIST = 0  # whether to plot distribution of sum of optimizer as histogram

# get sizes of: dimension, constraint-type, marginal-type, upper or lower, constraint val, VaR or ES
n_1, n_2, n_3, n_4, n_5 = len(D_LIST), len(C_TYPE), len(M_TYPE), len(UB_OR_LB), LB_UB_RANGE

val_list = np.zeros([n_1, n_2, n_3, n_4, n_5])
bound_list = np.zeros([n_1, n_2, n_3, n_4, n_5])
absolute_max_list = np.zeros([n_1, n_2, n_3, n_4, n_5])
feas_list = np.zeros([n_1, n_2, n_3, n_4, n_5])
th_list = np.zeros([n_1, n_2, n_3, n_4, n_5])

ind = 0
for d_ind, D in enumerate(D_LIST):
    for c_ind, C_T in enumerate(C_TYPE):
        for m_ind, M_T in enumerate(M_TYPE):
            for ul_ind, UL in enumerate(UB_OR_LB):
                for uol in range(LB_UB_RANGE):
                    if UL == -1:
                        cur_bound = 0.95 - 0.5 * uol / (LB_UB_RANGE - 1)
                        # 0.5 is the lowest value we look at for lower bound, and 0.95 the highest
                    if UL == 1:
                        cur_bound = -1 / (D - 1) + 0.05 + (0.4 + 1 / (D - 1) - 0.05) * uol / (LB_UB_RANGE - 1)
                        # -1/(D-1) is the minimum possible, and 0.05 is added for numerical feasibility;
                        # 0.4 is the maximum value that we look at for upper bound
                    ind += 1
                    print('Current index: ' + str(ind))
                    print('Current dimension: ' + str(D))
                    print('Current constraint type: ' + str(C_T))
                    print('Current marginal: ' + str(M_T))
                    print('Upper or lower bound: ' + str(UL))
                    print('Current level of bound: ' + str(uol) + ',' + str(cur_bound))
                    print('Current objective: ' + str(OBJ_H))

                    # path for loading values
                    save_path = '../data/' + str(ALPHA) + '_' + str(D) + '_' + str(C_T) + '_' + str(M_T) + '_' + str(
                        UL) + '_' + str(uol) + '_' + str(OBJ_H)
                    if QSF == 1:
                        save_path = save_path + '_Q75'
                    if N_SMALL_YN == 1:
                        save_path = save_path + str(N_SMALL_VAL)

                    # path for saving images
                    save_path_images = '../data/figures/' + str(ALPHA) + '_' + str(D) + '_' + str(C_T) + '_' + \
                                       str(M_T) + '_' + str(UL) + '_' + str(uol) + '_' + str(OBJ_H)
                    if QSF == 1:
                        save_path_images = save_path + '_Q75'
                    if N_SMALL_YN == 1:
                        save_path_images = save_path + str(N_SMALL_VAL)

                    # index for saving values
                    tot_ind = np.array([d_ind, c_ind, m_ind, ul_ind, uol], dtype=int)
                    tot_ind = tuple(tot_ind)

                    # try loading values, otherwise input put in some placeholder value and skip case
                    try:
                        val = np.load(save_path + '_ES_value.npy')
                        print(val)
                        opti_x = np.load(save_path + '_ES_optx.npy')
                        opti_w = np.load(save_path + '_ES_optw.npy')
                    except:
                        try:
                            # for cov lower bound, sometimes skipped cases because obvious comonotone is optimizer
                            val = np.load(save_path + '_comonotone_value.npy')
                        except:
                            val = 0
                        val_list[tot_ind] = val
                        bound_list[tot_ind] = cur_bound
                        continue

                    if PLOT_MARGINALS:
                        marg_1_x = opti_x[:, 0]
                        x_min = np.min(marg_1_x)
                        x_max = np.max(marg_1_x)
                        n_marg = len(marg_1_x)
                        n_samp = 10 ** 6
                        np.random.seed(0)  # for reproducability
                        x_big_sample = np.random.choice(marg_1_x, p=opti_w, replace=True, size=n_samp)
                        x_normal_sample = np.random.randn(n_samp)
                        xns_s = np.sort(x_normal_sample)
                        xbs_s = np.sort(x_big_sample)
                        fig, ax1 = plt.subplots()
                        ax2 = ax1.twinx()
                        lns1 = ax1.plot(xns_s, xbs_s, '.', label='optimizer marginal quantile')
                        lns2 = ax1.plot(xns_s, xns_s, label='diagonal')
                        # ax2.plot(xns_s, norm.pdf(xns_s))
                        # ax2.plot(xns_s, np.zeros(len(xns_s)), color='black', alpha=0.5)
                        lns3 = ax2.fill_between(xns_s, norm.pdf(xns_s), alpha=0.5, label='$\mathcal{N}(0, 1)$ density')
                        ax1.set_xlim(-5, 5)
                        ax1.set_xlabel('$\mathcal{N}(0, 1)$ quantiles')
                        ax2.set_ylabel('density level')
                        ax1.set_ylabel('marginal quantile level')
                        ax2.set_ylim(-0.1, 1)
                        if N_SMALL_YN == 0:
                            plt.title('$N=200$')
                        else:
                            plt.title('$N={}$'.format(N_SMALL_VAL))
                        lines, labels = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
                        plt.savefig(save_path_images + '.png', dpi=300, format='png')
                        plt.show()
                        # plt.clf()

                        # # Alternative way to visualize marginals
                        # marg_1_x_sorted = np.sort(marg_1_x)
                        # plt.plot(marg_1_x_sorted, norm.pdf(marg_1_x_sorted))
                        # plt.hist(marg_1_x, weights=opti_w, density=True, bins=26)
                        # plt.show()
                        # sns.kdeplot(marg_1_x, weights=opti_w)
                        # plt.show()

                    if PLOT_OPTIMIZERS:
                        plt.scatter(opti_x[:, 0], opti_x[:, 1], s=100 * np.sqrt(len(opti_w)) * opti_w, alpha=0.75)
                        if C_T == 1:
                            plt.title('Expected Shortfall optimizer, unconditional constraint')
                        elif C_T == 3:
                            plt.title('Expected Shortfall optimizer, conditional constraint')
                        else:
                            plt.title('Expected Shortfall optimizer, $x={bound:.3f}$'.format(bound=cur_bound))
                        plt.savefig(save_path_images + '_optimizer_joint.png', dpi=300, format='png')
                        plt.show()
                        # plt.clf()

                    if PLOT_SUM_DIST:
                        opti_s = np.sum(opti_x, axis=1)
                        sns.distplot(opti_s, hist_kws={'weights': opti_w}, bins=50, kde=False)
                        plt.title('$d={dim}$, $x={bound:.3f}$'.format(dim=D, bound=cur_bound))
                        plt.savefig(save_path_images + '_sum.png', dpi=300, format='png')
                        plt.show()
                        # plt.clf()

                    val_list[tot_ind] = val
                    bound_list[tot_ind] = cur_bound

# Plot values for different levels of bounds in one figure
for d_ind, D in enumerate(D_LIST):
    for c_ind, C_T in enumerate(C_TYPE):
        for m_ind, M_T in enumerate(M_TYPE):
            print('dimension and constraint type:', d_ind, c_ind)

            plt.plot(bound_list[d_ind, c_ind, m_ind, 0, :], val_list[d_ind, c_ind, m_ind, 0, :], 'o',
                     label='using lower bound')
            plt.plot(bound_list[d_ind, c_ind, m_ind, 1, :], val_list[d_ind, c_ind, m_ind, 1, :], '*',
                     label='using upper bound', color='C1')
            plt.ylim(ymin=0)
            plt.legend(loc='center right')
            if C_T == 1:
                plt.title(r'Expected shortfall, $d={}$'.format(D))
            if C_T == 3:
                plt.title(r'Expected shortfall, $d={}$'.format(D))
            plt.xlabel('Level of bound')
            plt.ylabel('Worst-case ES')
            plt.savefig(
                '../data/figures/es_' + str(round(100 * ALPHA)) + '_' + str(d_ind) + '_' + str(c_ind) + '_' + str(
                    m_ind)
                + str(QSF) + '.png', dpi=500, format='png')
            plt.show()
