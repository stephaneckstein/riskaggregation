from src.outer_optim import convex_opt, solve_total
from scipy.stats import norm, uniform, t
import rearrangement_algorithm as ra
import numpy as np

"""
This file includes testcases for expected shortfall for different dimensions, constraints, etc.
For expected shortfall, additionally to piecewise constant test functions, we also use piecewise linear functions,
This has some significant costs in terms of running times, but we also found that it can help decrease the approximation
error. To change this, one can set use_higher_order to zero.
"""

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
OUTPUTFLAG = 0  # outputflag for gurobi
stop_crit_eps = 0.5 * 10**-5  # stopping criteria epsilon for solve_total
N_SMALL_YN = 1  # if N_SMALL_YN == 1, then some other number of intervals other than 200 may be used
N_SMALL_VAL = 100  # specifies the number of intervals if N_SMALL_YN = 1
N_BISEC = 13  # number of trisection steps for ES outer optimization. Precision roughly ~(b-a) (3/2)^{-N_BISEC}

# Note: At the moment ES does not use early stopping;
# Using a different convex optimization approach one may be able to use early stopping...

# determine which cases are evaluated in this run
if USE_INDEXES:
    start_ind = int(input('Start index?'))
    end_ind = int(input('End index'))
else:
    start_ind = 0
    end_ind = 10**5


# define functions for constraint functions

# unconditional covariance functions
def make_cov_constr_fun(ind, mean_h, sig_h):
    mhe = np.reshape(mean_h, (1, -1))

    def constr_fun_cov_ub(x):
        return np.prod(x[ind]-mean_h[ind])/sig_h

    def constr_fun_vec_cov_ub(x):
        # x should be [n, d] size input. Returns [n] sized output
        return np.prod(x[:, ind]-mhe[:, ind], axis=1)/sig_h

    def constr_fun_cov_lb(x):
        return -np.prod(x[ind]-mean_h[ind])/sig_h

    def constr_fun_vec_cov_lb(x):
        # x should be [n, d] size input. Returns [n] sized output
        return -np.prod(x[:, ind]-mhe[:, ind], axis=1)/sig_h

    return constr_fun_cov_ub, constr_fun_cov_lb, constr_fun_vec_cov_ub, constr_fun_vec_cov_lb


# Only relevant for C_TYPE value of 2
def make_tail_constr_fun(ind, mean_h, sig_h, tail_start):
    def constr_fun_lb_tail(x):
        return -np.prod(np.maximum(x[ind]-tail_start[ind], 0)-mean_h[ind])/sig_h

    def constr_fun_ub_tail(x):
        return np.prod(np.maximum(x[ind]-tail_start[ind], 0)-mean_h[ind])/sig_h
    tse = np.reshape(tail_start, (1, -1))
    mhe = np.reshape(mean_h, (1, -1))

    def constr_fun_vec_lb_tail(x):
        return -np.prod(np.maximum(x[:, ind]-tse[:, ind], 0)-mhe[:, ind], axis=1)/sig_h

    def constr_fun_vec_ub_tail(x):
        return np.prod(np.maximum(x[:, ind]-tse[:, ind], 0)-mhe[:, ind], axis=1)/sig_h
    return constr_fun_ub_tail, constr_fun_lb_tail, constr_fun_vec_ub_tail, constr_fun_vec_lb_tail


# Conditional covariance constraint functions, conditioned on box that starts at tail_start
def make_cond_constr_fun(ind, mean_h, sig_h, tail_start, cost_h):
    def constr_fun_lb_cond(x):
        if np.all(x[ind] >= tail_start[ind]):
            return -((np.prod(x[ind])-np.prod(mean_h[ind]))/sig_h-cost_h)
        return 0

    def constr_fun_ub_cond(x):
        if np.all(x[ind] >= tail_start[ind]):
            return ((np.prod(x[ind])-np.prod(mean_h[ind]))/sig_h-cost_h)
        return 0

    tse = np.reshape(tail_start, (1, -1))
    mhe = np.reshape(mean_h, (1, -1))

    def constr_fun_vec_lb_cond(x):
        # x should be [n, d] size input. Returns [n] sized output
        inds_tail = np.all(x[:, ind] >= tse[:, ind], axis=1)
        out = np.zeros(len(x))
        out[inds_tail] = -((np.prod(np.array([x[inds_tail, i] for i in ind]), axis=0) - np.prod(mhe[:, ind],
                                                                                                axis=1))/sig_h - cost_h)
        return out

    def constr_fun_vec_ub_cond(x):
        # x should be [n, d] size input. Returns [n] sized output
        inds_tail = np.all(x[:, ind] >= tse[:, ind], axis=1)
        out = np.zeros(len(x))
        out[inds_tail] = ((np.prod(np.array([x[inds_tail, i] for i in ind]), axis=0) - np.prod(mhe[:, ind],
                                                                                               axis=1))/sig_h - cost_h)
        return out

    return constr_fun_ub_cond, constr_fun_lb_cond, constr_fun_vec_ub_cond, constr_fun_vec_lb_cond


# conditional first and second moment constraint functions
def make_cond_mom_constr(ind, mean_h, sig_h, tail_start):
    def first_mom_one(x):
        if np.all(x[ind] >= tail_start[ind]):
            return x[ind[0]] - mean_h[ind[0]]
        return 0

    def first_mom_two(x):
        if np.all(x[ind] >= tail_start[ind]):
            return x[ind[1]] - mean_h[ind[1]]
        return 0

    tse = np.reshape(tail_start, (1, -1))
    mhe = np.reshape(mean_h, (1, -1))

    def first_mom_one_vec(x):
        # x should be [n, d] size input. Returns [n] sized output
        inds_tail = np.all(x[:, ind] >= tse[:, ind], axis=1)
        out = np.zeros(len(x))
        out[inds_tail] = x[inds_tail, ind[0]] - mhe[:, ind[0]]
        return out

    def first_mom_two_vec(x):
        # x should be [n, d] size input. Returns [n] sized output
        inds_tail = np.all(x[:, ind] >= tse[:, ind], axis=1)
        out = np.zeros(len(x))
        out[inds_tail] = x[inds_tail, ind[1]] - mhe[:, ind[1]]
        return out

    def second_mom_one(x):
        if np.all(x[ind] >= tail_start[ind]):
            return (x[ind[0]]-mean_h[ind[0]])**2 - sig_h
        return 0

    def second_mom_two(x):
        if np.all(x[ind] >= tail_start[ind]):
            return (x[ind[1]]-mean_h[ind[1]])**2 - sig_h
        return 0

    def second_mom_one_vec(x):
        # x should be [n, d] size input. Returns [n] sized output
        inds_tail = np.all(x[:, ind] >= tse[:, ind], axis=1)
        out = np.zeros(len(x))
        out[inds_tail] = (x[inds_tail, ind[0]]-mhe[:, ind[0]])**2 - sig_h
        return out

    def second_mom_two_vec(x):
        # x should be [n, d] size input. Returns [n] sized output
        inds_tail = np.all(x[:, ind] >= tse[:, ind], axis=1)
        out = np.zeros(len(x))
        out[inds_tail] = (x[inds_tail, ind[1]]-mhe[:, ind[1]])**2 - sig_h
        return out

    return first_mom_one, first_mom_two, first_mom_one_vec, first_mom_two_vec, second_mom_one, second_mom_two, \
           second_mom_one_vec, second_mom_two_vec


# Functions that allow us to add as constraints the first and second moments of the considered measures;
# TODO: Perhaps have to add in the paper that we are doing this... It's basically one more function in dual relaxation
#       It's impossible this leads to worse bounds, but should be added for reproducability
def make_second(ind):  # second moment fixed
    def constr_fun_second(x):
        return np.square(x[ind])
    return constr_fun_second


def make_second_vec(ind):  # second moment fixed vector
    def constr_fun_second_vec(x):
        return np.square(x[:, ind])
    return constr_fun_second_vec


def make_first(ind):  # first moment fixed
    def constr_fun_first(x):
        return x[ind]
    return constr_fun_first


def make_first_vec(ind):  # first moment fixed vector
    def constr_fun_first_vec(x):
        return x[:, ind]
    return constr_fun_first_vec


ind = 0
for D in D_LIST:
    for C_T in C_TYPE:
        for M_T in M_TYPE:
            for UL in UB_OR_LB:
                skip_next = 0
                for uol in range(LB_UB_RANGE):
                    if UL == -1:
                        cur_bound = 0.95 - 0.5 * uol/(LB_UB_RANGE-1)
                        # 0.5 is the lowest value we look at for lower bound, and 0.95 the highest
                    if UL == 1:
                        cur_bound = -1/(D-1) + 0.05 + (0.4+1/(D-1)-0.05)*uol/(LB_UB_RANGE-1)
                        # -1/(D-1) is the minimum possible, and 0.05 is added for numerical feasibility;
                        # 0.4 is the maximum value that we look at for upper bound
                    ind += 1
                    print('Current index: ' + str(ind))
                    if (ind < start_ind) or (ind > end_ind):
                        continue

                    # Print what the current case is and get path for saving values
                    print('Current dimension: ' + str(D))
                    print('Current constraint type: ' + str(C_T))
                    print('Current marginal: ' + str(M_T))
                    print('Upper or lower bound: ' + str(UL))
                    print('Current level of bound: ' + str(uol) + ',' + str(cur_bound))
                    print('Current objective: ' + str(OBJ_H))
                    save_path = '../data/' + str(ALPHA) + '_' + str(D) + '_' + str(C_T) + '_' + str(M_T) + '_' + str(
                        UL) + '_' + str(uol) + '_' + str(OBJ_H)
                    if QSF == 1:
                        save_path = save_path + '_Q75'
                    if N_SMALL_YN == 1:
                        save_path = save_path + str(N_SMALL_VAL)

                    if skip_next:
                        print('Case skipped because the previous (more constrained) case already achieved the maximum '
                              'possible value given by the comonotone coupling')
                        np.save(save_path + '_ES_value', es_val)
                        np.save(save_path + '_ES_alpha', ALPHA)
                        np.save(save_path + '_ES_optx', xo)
                        np.save(save_path + '_ES_optw', wo)
                        continue

                    # initialize parameters of marginals
                    mean_vec = np.array([0] * D)
                    scale_vec = np.array([1] * D)
                    if M_T == 1:
                        margs = [norm(loc=mean_vec[i], scale=scale_vec[i]) for i in range(D)]
                    if M_T == 2:
                        DF = 5
                        margs = [t(loc=mean_vec[i], scale=scale_vec[i] / (np.sqrt(DF / (DF - 2))), df=DF) for i in
                                 range(D)]
                    if M_T == 3:
                        margs = [uniform() for i in range(D)]

                    # get relevant functions for marginals
                    margs_cdfs = [margs[i].cdf for i in range(D)]
                    marg_sampler = [margs[i].rvs for i in range(D)]
                    marg_ppf = [margs[i].ppf for i in range(D)]

                    # initialize tails for conditional covariance constraint
                    RETURN_PRIMAL = 1
                    tail_start = np.zeros([D])
                    tail_start_vec = np.zeros([1, D])
                    if QSF == 1:
                        tail_start += marg_ppf[1](0.75)
                        tail_start_vec += marg_ppf[1](0.75)
                    else:
                        tail_start += marg_ppf[1](0.5)
                        tail_start_vec += marg_ppf[1](0.5)

                    # Calculate means and variances for additionally imposing constraints on first and second marginal
                    # moments in the dual formulation
                    print('Calculating means and variances for constraints ...')
                    if M_T == 1 or M_T == 2:
                        mean_tot = mean_vec
                        sec_mom = np.ones(D)
                    if M_T == 3:
                        mean_tot = mean_vec + 0.5
                        sec_mom = np.ones(D) * 1/3

                    # determine number of interval parts and number of initial points in constraint
                    n_interval = 200
                    n_constraint_start = n_interval + 5000 + 2500 * D

                    # determine bounds for compactification of the problem
                    use_bounds = 1
                    use_equal_grid = 1

                    bounds = np.zeros([2, D])
                    if M_T == 1 or M_T == 2:
                        for i in range(D):
                            bounds[0, i] = marg_ppf[i](min(0.01, 0.1 / n_interval))
                            bounds[1, i] = marg_ppf[i](max(0.99, 1 - 0.1 / n_interval))
                    else:
                        for i in range(D):
                            bounds[0, i] = 0
                            bounds[1, i] = 1
                    bound_rep_l = np.mean(bounds[0, :])
                    bound_rep_u = np.mean(bounds[1, :])

                    # mean and sigma for constraints depending on which marginal is used
                    # used to normalize to correlation levels instead of covariances
                    if C_T == 1:
                        if M_T == 1 or M_T == 2:
                            mean_for_constr = mean_vec
                            sig_for_constr = 1.
                        if M_T == 3:
                            mean_for_constr = mean_vec + 0.5
                            sig_for_constr = (1/12)

                    if C_T == 2:
                        samp_h = marg_sampler[0](10 ** 8)  # Assumption: All marginals are the same!!!
                        samp_h = np.minimum(np.maximum(samp_h, bounds[0, 0]), bounds[1, 0])
                        mean_for_constr = np.mean(np.maximum(samp_h-tail_start[0], 0))
                        sig_for_constr = np.mean((np.maximum(samp_h-tail_start[0], 0) - mean_for_constr)**2)

                    if C_T == 3:
                        samp_h = marg_sampler[0](10 ** 8)
                        samp_h = np.minimum(np.maximum(samp_h, bounds[0, 0]), bounds[1, 0])
                        mean_for_constr = np.mean(samp_h[samp_h >= tail_start[0]])
                        sig_for_constr = np.mean((samp_h[samp_h >= tail_start[0]] - mean_for_constr) ** 2)
                    mean_for_constr = np.ones(D) * mean_for_constr

                    # determine comonotone values for worst case expected shortfall without additional constraints
                    samp_0 = marg_sampler[0](10 ** 8)
                    samp_sort = np.sort(samp_0)
                    threshold = samp_sort[int(np.round(ALPHA * 10 ** 8))] * D
                    STOP_VAL = D * np.mean(samp_sort[int(np.round(ALPHA * 10 ** 8)):])
                    print('Comonotone ES here: ' + str(STOP_VAL))
                    np.save(save_path + '_comonotone_value', STOP_VAL)

                    # determine parameters for solve_total algorithm
                    SUP_USE = 1
                    n_new = 75
                    total_steps = 1000
                    n_test = (7+D) * 10 ** 5  # (2+D) * 10 ** 5
                    n_local = 10 ** 4
                    two_step_n = 10 ** 4
                    n_line = 10 ** 2
                    eps_ball = 2 * (bound_rep_u - bound_rep_l) / n_interval
                    marginal_ball = 0  # marginal_ball = 1 works better if use_equal_grid = 0
                    use_countermonotone = 1
                    use_higher_order = 1    # If this is nonzero, one has to be careful with using bounds when
                                            # calculating integral values
                    use_shifted_points = 0
                    print_values = 1
                    show_time = 1
                    two_step_mult = 5
                    line_mult = 2

                    # determine concrete constraints for this test case and the given level
                    constraints = []
                    cfuns = []
                    cmults = []
                    czs = []
                    constraint_funs_vec = []
                    for i in range(D):
                        cfuns.append(make_second(i))
                        cmults.append(sec_mom[i])
                        czs.append(1)
                        constraint_funs_vec.append(make_second_vec(i))
                        cfuns.append(make_first(i))
                        cmults.append(mean_tot[i])
                        czs.append(1)
                        constraint_funs_vec.append(make_first_vec(i))
                    for i in range(D - 1):
                        for j in range(i + 1, D):
                            ind_1 = np.array([i, j], dtype=int)
                            if C_T == 1:
                                if UL == -1:
                                    _, con_fun, _, con_fun_vec = make_cov_constr_fun(ind_1, mean_for_constr,
                                                                                     sig_for_constr)
                                else:
                                    con_fun, _, con_fun_vec, _ = make_cov_constr_fun(ind_1, mean_for_constr,
                                                                                     sig_for_constr)
                            if C_T == 2:
                                if UL == -1:
                                    _, con_fun, _, con_fun_vec = make_tail_constr_fun(ind_1, mean_for_constr,
                                                                                      sig_for_constr, tail_start)
                                else:
                                    con_fun, _, con_fun_vec, _ = make_tail_constr_fun(ind_1, mean_for_constr,
                                                                                      sig_for_constr, tail_start)
                            if C_T == 3:
                                if UL == -1:
                                    _, con_fun, _, con_fun_vec = make_cond_constr_fun(ind_1, mean_for_constr,
                                                                                      sig_for_constr, tail_start,
                                                                                      cur_bound)
                                else:
                                    con_fun, _, con_fun_vec, _ = make_cond_constr_fun(ind_1, mean_for_constr,
                                                                                      sig_for_constr, tail_start,
                                                                                      cur_bound)

                            cfuns.append(con_fun)
                            if C_T != 3:
                                cmults.append(UL * cur_bound)
                            else:
                                cmults.append(0)
                            czs.append(0)
                            constraint_funs_vec.append(con_fun_vec)

                            if C_T == 3:
                                fcm, scm, fcmv, scmv, hfcm, hscm, hfcmv, hscmv = make_cond_mom_constr(ind_1,
                                                                                                      mean_for_constr,
                                                                                                      sig_for_constr,
                                                                                                      tail_start,
                                                                                                      cur_bound)
                                cfuns.append(fcm)
                                cfuns.append(scm)
                                cmults.append(0)
                                cmults.append(0)
                                czs.append(1)
                                czs.append(1)
                                constraint_funs_vec.append(fcmv)
                                constraint_funs_vec.append(scmv)
                                cfuns.append(hfcm)
                                cfuns.append(hscm)
                                cmults.append(0)
                                cmults.append(0)
                                czs.append(1)
                                czs.append(1)
                                constraint_funs_vec.append(hfcmv)
                                constraint_funs_vec.append(hscmv)

                    constraints = [cfuns, cmults, czs]

                    # set up function to make cost function for expected shortfall
                    def make_cost_fun(th_h):
                        def cost_fun_avar(x):
                            return np.maximum(np.sum(x) - th_h, 0) / (1 - ALPHA) + th_h

                        def cost_fun_vec_avar(x):
                            return np.maximum(np.sum(x, axis=1) - th_h, 0) / (1 - ALPHA) + th_h

                        return cost_fun_avar, cost_fun_vec_avar

                    # precompute integral values for first order test functions.
                    # More precisely, we simply calculate the mean of the marginals conditional on each interval
                    use_precomp_higher = 1
                    n1 = n_interval
                    price_high = np.zeros([n1+1, D, use_higher_order])
                    hpoints = np.concatenate([np.linspace(bounds[0, i], bounds[1, i], n1).reshape(-1, 1) for i in
                                              range(D)], axis=1)
                    if use_bounds == 1 and use_equal_grid == 1:
                        hpoints = np.concatenate(
                            [np.linspace(bounds[0, i], bounds[1, i], n1).reshape(-1, 1) for i in range(D)], axis=1)
                    else:
                        hpoints = np.concatenate(
                            [marg_ppf[i](np.linspace(1 / (n1 * 2), 1 - 1 / (n1 * 2), n1).reshape(-1, 1))
                             for i in range(D)], axis=1)

                    n_calc_higher = 1000
                    for i in range(D):
                        probs = margs_cdfs[i](hpoints[:, i])
                        probs = np.insert(probs, [0, len(probs)], [0, 1])
                        for j in range(n1+1):
                            lhs_h = probs[j]
                            rhs_h = probs[j + 1]
                            p_h = np.linspace(lhs_h, rhs_h, n_calc_higher + 1)
                            e_h = marg_ppf[i]((p_h[1:] + p_h[:-1]) / 2)
                            e_h = np.maximum(np.minimum(e_h, bounds[1, i])+0.025, bounds[0, i]-0.025)
                            for k in range(use_higher_order):
                                price_high[j, i, k] = np.mean(e_h ** (k + 1))

                    # this is the precise univariate convex function we want to minimize
                    def f_here(th):
                        cost_fun, cost_fun_vec = make_cost_fun(th)

                        value, pi_opt = solve_total(margs_cdfs, marg_ppf, marg_sampler, cost_fun, cost_fun_vec,
                                                    constraints, constraint_funs_vec, n_interval=n_interval,
                                                    n_constr=n_constraint_start, n_step_constraint=total_steps,
                                                    n_pot_constr=n_test, n_new_constr=n_new,
                                                    n_local=n_local, n_line=n_line, eps_ball=eps_ball, two_step=1,
                                                    two_step_mult=two_step_mult,
                                                    line_mult=line_mult, use_bounds=use_bounds, bounds=bounds,
                                                    use_countermonotone=use_countermonotone, outputflag=OUTPUTFLAG,
                                                    print_values=print_values,
                                                    show_time=show_time, marginal_ball=marginal_ball,
                                                    use_higher_order=use_higher_order,
                                                    use_precomp_higher=use_precomp_higher,
                                                    price_high=price_high, use_shifted_points=use_shifted_points,
                                                    return_primal=RETURN_PRIMAL, equal_space_inter=use_equal_grid,
                                                    new_points_from_sup=SUP_USE, sparse_init=1, min_steps=19,
                                                    stop_value=0, stop_crit=stop_crit_eps)

                        return value

                    # determine bounds for convex optimization algorithm.
                    # Clearly between 0 and highest possible value at risk (without constraint),
                    # since for the worst-case measure, the optimal level is its value at risk
                    start_lb = 0.
                    ra_lower, ra_upper = ra.bounds_VaR(ALPHA, marg_ppf, num_steps=100000, method='upper')
                    ra_threshold = (ra_lower[0]+ra_upper[0])/2
                    start_ub = ra_threshold + 0.1

                    # run the convex optimization and determine optimizer
                    conv_pl, conv_vl = convex_opt(f_here, a=start_lb, b=start_ub, N=N_BISEC)
                    es_val = np.min(conv_vl)
                    es_val_arg = np.argmin(conv_vl)
                    th_min = conv_pl[es_val_arg]
                    cost_fun, cost_fun_vec = make_cost_fun(th_min)

                    # run again for the optimal level to get optimizer
                    print('Running final value for ' + str(th_min) + ' to obtain optimizer')
                    value, pi_opt = solve_total(margs_cdfs, marg_ppf, marg_sampler, cost_fun, cost_fun_vec,
                                                constraints, constraint_funs_vec, n_interval=n_interval,
                                                n_constr=n_constraint_start, n_step_constraint=total_steps,
                                                n_pot_constr=n_test, n_new_constr=n_new,
                                                n_local=n_local, n_line=n_line, eps_ball=eps_ball, two_step=1,
                                                two_step_mult=two_step_mult,
                                                line_mult=line_mult, use_bounds=use_bounds, bounds=bounds,
                                                use_countermonotone=use_countermonotone, outputflag=OUTPUTFLAG,
                                                print_values=print_values,
                                                show_time=show_time, marginal_ball=marginal_ball,
                                                use_higher_order=use_higher_order,
                                                use_precomp_higher=use_precomp_higher,
                                                price_high=price_high, use_shifted_points=use_shifted_points,
                                                return_primal=RETURN_PRIMAL, equal_space_inter=use_equal_grid,
                                                new_points_from_sup=SUP_USE, sparse_init=1, min_steps=19,
                                                stop_value=0, stop_crit=stop_crit_eps)

                    # read out optimizer and save
                    xo = pi_opt[0]
                    wo = np.array(pi_opt[1])
                    if es_val > STOP_VAL-0.005:
                        skip_next = 1
                    np.save(save_path + '_ES_value', es_val)
                    np.save(save_path + '_ES_alpha', ALPHA)
                    np.save(save_path + '_ES_optx', xo)
                    np.save(save_path + '_ES_optw', wo)
