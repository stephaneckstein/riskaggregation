import numpy as np
from src.generate_points import get_new_const_points_from_old, get_new_const_points_dist, get_new_const_points
from src.linearprograms import dual_lp_new, solve_given_model_new_constraint
from time import time

"""
this file includes outer algorithms:
1) to solve the semi-infinite dual problem by iteratively increasing the number of constraint points
2) bisection to solve value at risk outer problem (find zero of monotone univariate function)
3) trisection to solve Avar outer problem (find minimum of univariate convex function)
"""


def get_primal_sol(model, points, indices=()):
    """
    supplementary function which turns dual optimizer into primal optimizer
    :param model: optimized gurobi model
    :param points: constraint points that were used for dual inequalit constraint
    :param indices: indices of constraint where the actual dual inequalit constraint is tested
    (included because there may also be shortselling constraint etc)
    :return: points and weights of optimal measure
    """
    npoints, D = points.shape
    if len(indices) > 0:
        mparr = np.array(model.Pi)[indices]
    else:
        mparr = np.array(model.Pi)
    nnz = np.sum(mparr > 0)
    opt_x = np.zeros([nnz, D])
    opt_w = []
    ind = 0
    for i in range(npoints):
        if mparr[i] > 0:
            opt_x[ind, :] = points[i, :]
            ind += 1
            opt_w.append(mparr[i])
    return opt_x, opt_w


def solve_total(margs_cdfs, margs_ppf, margs_rvs, cost_fun, cost_fun_vec, constraints, constraint_funs_vec,
                n_interval=50, n_constr=200, n_step_constraint=50, n_pot_constr=10 ** 3, n_new_constr=10,
                n_local=10 ** 3, n_line=3 * 10 ** 2, eps_ball=0.1, two_step=1, two_step_mult=1, line_mult=1,
                use_bounds=0, use_countermonotone=0, bounds=(), outputflag=0, print_values=0, show_time=0,
                marginal_ball=0, use_higher_order=0, use_precomp_higher=0, price_high=(),
                n_calc_higher=1000, use_shifted_points=0, eps_shifted_scale=1 / 50, return_primal=0,
                equal_space_inter=0, new_points_from_sup=0, stop_crit=10 ** -5, min_steps=15, sparse_init=0,
                stop_value=0,
                max_time=0, stop_crit_steps=6, eps_bounds=10 ** -7):
    """
    main function to solve dual problem for optimal transport with additional constraints on discrete grid
    :param margs_cdfs: list with d entries, each being cumulative distribution function of one marginals
    :param margs_ppf: list with d entries, each being quantile function of one marginal
    :param margs_rvs: list with d entries, each being function which generates points for one of marginals
    :param cost_fun: cost function taking inputs of shape [d]
    :param cost_fun_vec:  vectorized cost function taking inputs of shape [n, d]
    :param constraints: list of 3 elements, each being list of same size, [f_1, ...], [c_1, ...], [b_1, ...]
    enables dual function trading alpha_i f_i for price alpha_i c_i,
    where alpha_i <= 0, if b_i == -1, alpha_i >= 0, if b_i == 0, and alpha_i arbitrary, else
    functions f_i take inputs of shape [d]
    :param constraint_funs_vec: as first entry of constraints, but vectorized (taking inputs of shape [n, d])
    :param n_interval: number of intervals for each marginal
    :param n_constr: randomly sampled points to test inequality constraint for initilization of problem
    :param n_step_constraint: maximum number of steps for updating the set of points where constraint is tested
    :param n_pot_constr: argument searchsize (number of global points) for generation of point functions
    :param n_new_constr: number of points among global points chosen
    :param n_local: number of local points to search
    :param n_line: number of points on line segment to search
    :param eps_ball: size of ball for local point search
    :param two_step: 0/1 whether to use local search or not
    :param two_step_mult: how many local points to use for each global good point
    :param line_mult: how many points on line segment to use
    :param use_bounds: whether to cut tails off or not
    :param use_countermonotone: whether, for initializaiton purposes, use pairwise countermonotone points as well
    :param bounds: if use_bounds, determines what those bounds are
    :param outputflag: 0/1 whether to display output from gurobi
    :param print_values: 0/1 whether to print values during iterations of outer algorithm
    :param show_time: 0/1 whether to show time of certain parts of program
    :param marginal_ball: whether to use uniformly shaped balls for local search or product measure of marginal balls
    :param use_higher_order: integer, if > 0, then use of higher order polynomials as well as piece-wise constant funs
    :param use_precomp_higher: 0/1 if higher order poly are used, this says whether integral values are precomputed
    :param price_high: the corresponding values of the integrals for higher order polymoniamls on intervals
    :param n_calc_higher: if not precomputed, this says how many sample points are used to calculate higher order
    :param use_shifted_points: 0/1 whether to initialize also some points on boundaries of intervals
    :param eps_shifted_scale: determines how far away from boundary the generated points should be
    :param return_primal: whether to return primal optimizer
    :param equal_space_inter: 0/1, if one, then the intervals for marginals are equally spaced (recommended),
    otherwise, the intervals are spaced to have equal mass on each subinterval
    :param new_points_from_sup: 0/1 if one, then generate local points around current optimizers as well
    :param stop_crit: epsilon for stopping criteria
    :param min_steps: minimum number of updating steps for constraint points before stopping criteria can apply
    :param sparse_init: 0/1, if one, then runs new program with conservative parameter for generating initial solution,
    afterwards deletes all points except support of optimizer
    Note that this sometimes leads to numerical errors, and in this case, for instance increasing NumericFocus
    variable of gurobi helps
    :param stop_value: if nonzero, then if this value is surpassed, the program stops and outputs the current value
    particularly useful when calculating value at risk
    :param max_time: optionally specifies maximum time for gurobi to use.
    :param stop_crit_steps: specifies number of points which are measured for stopping criteria (std across this number
    of points is taken when deciding whether to stop or not)
    :param eps_bounds:
    :return:
    """

    # read out input
    n1 = n_interval
    n2 = n_constr
    D = len(margs_cdfs)

    # initialize intervals for marginals
    if use_bounds == 1 and equal_space_inter == 1:
        hpoints = np.concatenate([np.linspace(bounds[0, i], bounds[1, i], n1).reshape(-1, 1) for i in range(D)], axis=1)
    else:
        hpoints = np.concatenate([margs_ppf[i](np.linspace(1 / (n1 * 2), 1 - 1 / (n1 * 2), n1).reshape(-1, 1))
                                  for i in range(D)], axis=1)

    # if sparse init, call new program
    if sparse_init == 1:
        # try to generate initial solution
        print('Initializing solution ... ')
        value_start, pi_opt_start = solve_total(margs_cdfs, margs_ppf, margs_rvs, cost_fun, cost_fun_vec, constraints,
                                                constraint_funs_vec,
                                                n_interval=n_interval, n_constr=5 * 10 ** 4, n_step_constraint=100,
                                                n_pot_constr=10 ** 5, n_new_constr=5 * 10 ** 4,
                                                n_local=100, n_line=10, eps_ball=eps_ball, two_step=1, two_step_mult=1,
                                                line_mult=1,
                                                use_bounds=use_bounds, use_countermonotone=1, bounds=bounds,
                                                outputflag=0, print_values=1, show_time=1,
                                                marginal_ball=0, use_higher_order=use_higher_order,
                                                use_precomp_higher=use_precomp_higher, price_high=price_high,
                                                n_calc_higher=n_calc_higher, use_shifted_points=1,
                                                eps_shifted_scale=eps_shifted_scale, return_primal=1,
                                                equal_space_inter=equal_space_inter, new_points_from_sup=0, stop_crit=1,
                                                min_steps=100, sparse_init=2, max_time=max_time)
        if value_start == -1000000000:
            print('No initial solution found...')
            return value_start, pi_opt_start

        # build initial set of constraint points out of this initial solution
        xo = pi_opt_start[0]

        # # In case of numerical instabilities when sparse initializiation is used, the code below can lead to more
        # # robust initialization by adding some other perturbed points around optimizers
        # multpertub_start = 5
        # eps_perturb_start = 0.005
        # xo_pert = np.tile(xo, (multpertub_start, 1))
        # xo_pert += (np.random.random_sample(xo_pert.shape)*2-1) * eps_perturb_start
        # rand_points = np.concatenate([margs_rvs[i](size=(n2 - n1, 1)) for i in range(D)], axis=1)
        # const_test_points = np.concatenate([hpoints, xo, xo_pert, rand_points])

        const_test_points = np.concatenate([hpoints, xo])
        print('Sparse initiation of constraint points, numerical instabilities are possible \n'
              '(if infeasible afterwards: change code above this message or increase NumericFocus variable of Gurobi)')

    else:  # if no sparse initialization is used, otherwise initialize constraint points
        rand_points = np.concatenate([margs_rvs[i](size=(n2 - n1, 1)) for i in range(D)], axis=1)
        const_test_points = np.concatenate([hpoints, rand_points], axis=0)
        if use_shifted_points:
            eps_shifted = 1 / (n1 + 2) * eps_shifted_scale
            if use_bounds == 1 and equal_space_inter == 1:
                hpoints_2 = hpoints + 0.1 * (bounds[1, :] - bounds[0, :]) / n1
                hpoints_3 = hpoints - 0.1 * (bounds[1, :] - bounds[0, :]) / n1
            else:
                hpoints_2 = np.concatenate(
                    [margs_ppf[i](np.linspace(eps_shifted, 1 - 1 / (n1 + 2) + eps_shifted, n1).reshape(-1, 1)) for i in
                     range(D)], axis=1)
                hpoints_3 = np.concatenate(
                    [margs_ppf[i](np.linspace(1 / (n1 + 2) - eps_shifted, 1 - eps_shifted, n1).reshape(-1, 1)) for i in
                     range(D)], axis=1)
            const_test_points = np.concatenate([const_test_points, hpoints_2, hpoints_3], axis=0)
            n2 += len(hpoints_3) + len(hpoints_2)

        if use_countermonotone:
            for i in range(D):
                hch = hpoints.copy()
                hch[:, i] = np.flipud(hch[:, i])
                const_test_points = np.concatenate([const_test_points, hch], axis=0)
                n2 += len(hch)
                if use_shifted_points:
                    hch2 = hpoints_2.copy()
                    hch2[:, i] = np.flipud(hch2[:, i])
                    const_test_points = np.concatenate([const_test_points, hch2], axis=0)
                    n2 += len(hch2)
                    hch3 = hpoints_3.copy()
                    hch3[:, i] = np.flipud(hch3[:, i])
                    const_test_points = np.concatenate([const_test_points, hch3], axis=0)
                    n2 += len(hch2)

    # optionally cut points around bounds
    if use_bounds:
        bounds[0, :] = bounds[0, :] - eps_bounds
        bounds[1, :] = bounds[1, :] + eps_bounds
        if use_higher_order:
            const_test_points = np.maximum(const_test_points, bounds[0:1, :] - 0.1)
            const_test_points = np.minimum(const_test_points, bounds[1:2, :] + 0.1)
        else:
            const_test_points = np.maximum(const_test_points, bounds[0:1, :])
            const_test_points = np.minimum(const_test_points, bounds[1:2, :])

    # sort constraint points into the intervals
    sort_into = np.zeros(const_test_points.shape)
    for i in range(D):
        sort_into[:, i] = np.searchsorted(hpoints[:, i], const_test_points[:, i])

    # all_points summarize all points ever considered
    all_points = const_test_points.copy()
    indices = list(range(len(all_points)))

    # n_up will be the number of constraint points that are added at each iteration of the algorithm
    n_up = int(n_new_constr * two_step_mult * line_mult)
    if new_points_from_sup:
        n_up *= 2

    if sparse_init == 2:
        MT = max_time
    else:
        MT = 0

    # solve dual problem for the initial set of points
    if not use_higher_order:
        val, hvar, coefvar, const, m, h_var, coeff_var, constant = dual_lp_new(margs_cdfs, cost_fun, constraints,
                                                                               hpoints, const_test_points, sort_into,
                                                                               outputflag=outputflag, model_out=1,
                                                                               use_higher_order=use_higher_order,
                                                                               use_precomp_higher=use_precomp_higher,
                                                                               price_high=price_high,
                                                                               n_calc_higher=n_calc_higher,
                                                                               marg_ppf=margs_ppf, mt=MT)
    else:
        val, hvar, coefvar, const, m, h_var, coeff_var, constant, h_higher, h_higher_val = dual_lp_new(margs_cdfs,
                                                                                                       cost_fun,
                                                                                                       constraints,
                                                                                                       hpoints,
                                                                                                       const_test_points,
                                                                                                       sort_into,
                                                                                                       outputflag=outputflag,
                                                                                                       model_out=1,
                                                                                                       use_higher_order=use_higher_order,
                                                                                                       use_precomp_higher=use_precomp_higher,
                                                                                                       price_high=price_high,
                                                                                                       n_calc_higher=n_calc_higher,
                                                                                                       marg_ppf=margs_ppf,
                                                                                                       mt=MT)

    # number of constraint points so far; used for labeling of new constraint points
    n_old = len(const_test_points)

    if return_primal or new_points_from_sup:
        start = len(m.Pi)
    if print_values:
        print(val)

    # initialize value list etc for stopping criteria, also initialize times for current iterations to determine
    # whether it makes sense to reinitialize the problem
    val_list = [val]
    tlp = 0
    tlp_hist = [tlp]
    tsamp = 1000
    val_list_stopping_crit = []
    if val > -100000 and val != 0:
        val_list_stopping_crit.append(val)

    # start iteration
    for i in range(n_step_constraint):
        print('Current size: ' + str(n_old))

        # check early stopping criteria
        if stop_value:
            if val_list[-1] > stop_value:
                print('We reached the maximum possible value (theoretically...)')
                break
        if sparse_init == 2:
            if val_list[-1] != -1000000000:
                print('Initial solution found!')
                break
        if len(val_list_stopping_crit) > min_steps:
            if np.std(val_list_stopping_crit) > 0 and np.std(val_list_stopping_crit[
                                                             -stop_crit_steps:]) <= stop_crit:  # and np.mean(viol_hist[min(i-6, 0):]) <= stop_crit:
                print('Early stopping criteria met!')
                break
            if len(val_list) > 9 and np.std(val_list) == 0 and val_list[-1] == -1000000000:
                print('I think this will stay infeasible...')
                break

        t0 = time()

        # if it is worth it in terms of time, then reinitialize problem
        if len(tlp_hist) > 6 and tlp_hist[-1] > np.mean(tlp_hist[:6]) * 5 and tlp_hist[-1] > 3.5 * tsamp:
            print('Reinitializing LP...')
            t0 = time()
            x_p, w_p = get_primal_sol(m, all_points, np.array(indices + list(range(start, len(m.Pi))), dtype=int))
            const_test_points = np.concatenate([hpoints, x_p])
            n_old = len(const_test_points)
            sort_into = np.zeros(const_test_points.shape)
            for i in range(D):
                sort_into[:, i] = np.searchsorted(hpoints[:, i], const_test_points[:, i])

            if not use_higher_order:
                val, hvar, coefvar, const, m, h_var, coeff_var, constant = dual_lp_new(margs_cdfs, cost_fun,
                                                                                       constraints,
                                                                                       hpoints, const_test_points,
                                                                                       sort_into,
                                                                                       outputflag=outputflag,
                                                                                       model_out=1,
                                                                                       use_higher_order=use_higher_order,
                                                                                       use_precomp_higher=use_precomp_higher,
                                                                                       price_high=price_high,
                                                                                       n_calc_higher=n_calc_higher,
                                                                                       marg_ppf=margs_ppf, mt=MT)
            else:
                val, hvar, coefvar, const, m, h_var, coeff_var, constant, h_higher, h_higher_val = dual_lp_new(
                    margs_cdfs, cost_fun, constraints,
                    hpoints, const_test_points, sort_into,
                    outputflag=outputflag, model_out=1,
                    use_higher_order=use_higher_order,
                    use_precomp_higher=use_precomp_higher,
                    price_high=price_high,
                    n_calc_higher=n_calc_higher,
                    marg_ppf=margs_ppf, mt=MT)
            tlp = np.mean(tlp_hist[:5])
            tlp_hist = [tlp]
            all_points = const_test_points.copy()
            indices = list(range(len(all_points)))
            start = len(m.Pi)
            print('Took ' + str(time() - t0) + ' many seconds')
            print('Current value:', val)
            continue

        if not use_higher_order:  # no higher order polynomials
            # generate new points globally
            nph = get_new_const_points(margs_rvs, cost_fun_vec, hpoints, hvar, constraint_funs_vec, coefvar, const,
                                       size=n_new_constr, searchsize=n_pot_constr, two_step=two_step,
                                       two_step_n=n_local,
                                       two_step_mult=two_step_mult, line_n=n_line, line_mult=line_mult,
                                       eps_ball=eps_ball,
                                       use_bounds=use_bounds, bounds=bounds, marginal_ball=marginal_ball,
                                       marg_cdf=margs_cdfs, marg_ppf=margs_ppf)

            # generate points that are far away from points generated so far
            x_p, w_p = get_primal_sol(m, all_points, np.array(indices + list(range(start, len(m.Pi))), dtype=int))
            nph3 = get_new_const_points_dist(margs_rvs, cost_fun_vec, hpoints, hvar, constraint_funs_vec, coefvar,
                                             const, all_points,
                                             searchsize=n_pot_constr, out_size=4 * n_new_constr, use_bounds=use_bounds,
                                             bounds=bounds, use_higher_order=0, h_higher_val=())
            nph4 = get_new_const_points_dist(margs_rvs, cost_fun_vec, hpoints, hvar, constraint_funs_vec, coefvar,
                                             const, x_p,
                                             searchsize=n_pot_constr, out_size=4 * n_new_constr, use_bounds=use_bounds,
                                             bounds=bounds, use_higher_order=0, h_higher_val=())

            if new_points_from_sup and val_list[-1] != -1000000000:
                # generate points locally around optimizer
                nph2 = get_new_const_points_from_old(x_p, cost_fun_vec, hpoints, hvar, constraint_funs_vec, coefvar,
                                                     const,
                                                     size=n_new_constr, two_step_n=n_local, two_step_mult=two_step_mult,
                                                     line_n=n_line, line_mult=line_mult, eps_ball=eps_ball,
                                                     use_bounds=use_bounds, bounds=bounds, marginal_ball=marginal_ball,
                                                     marg_cdf=margs_cdfs, marg_ppf=margs_ppf)
                nph = np.concatenate([nph, nph2, nph3, nph4], axis=0)
            else:
                # if no points around optimizer are generated, we generated some more global points
                nph2 = get_new_const_points(margs_rvs, cost_fun_vec, hpoints, hvar, constraint_funs_vec, coefvar, const,
                                            size=n_new_constr, searchsize=n_pot_constr, two_step=two_step,
                                            two_step_n=n_local,
                                            two_step_mult=two_step_mult, line_n=n_line, line_mult=line_mult,
                                            eps_ball=eps_ball,
                                            use_bounds=use_bounds, bounds=bounds, marginal_ball=marginal_ball,
                                            marg_cdf=margs_cdfs, marg_ppf=margs_ppf)
                nph = np.concatenate([nph, nph2, nph3, nph4], axis=0)

        else:  # we use higher order polynomials
            # generate global points
            nph = get_new_const_points(margs_rvs, cost_fun_vec, hpoints, hvar, constraint_funs_vec, coefvar, const,
                                       size=n_new_constr, searchsize=n_pot_constr, two_step=two_step,
                                       two_step_n=n_local,
                                       two_step_mult=two_step_mult, line_n=n_line, line_mult=line_mult,
                                       eps_ball=eps_ball,
                                       use_bounds=use_bounds, bounds=bounds, marginal_ball=marginal_ball,
                                       marg_cdf=margs_cdfs, marg_ppf=margs_ppf, use_higher_order=use_higher_order,
                                       h_higher_val=h_higher_val)

            x_p, w_p = get_primal_sol(m, all_points, np.array(indices + list(range(start, len(m.Pi))), dtype=int))

            # generate points which are far away from current optimizer
            nph3 = get_new_const_points_dist(margs_rvs, cost_fun_vec, hpoints, hvar, constraint_funs_vec, coefvar,
                                             const, all_points,
                                             searchsize=n_pot_constr, out_size=4 * n_new_constr,
                                             use_bounds=use_bounds, bounds=bounds, use_higher_order=use_higher_order,
                                             h_higher_val=h_higher_val)
            nph4 = get_new_const_points_dist(margs_rvs, cost_fun_vec, hpoints, hvar, constraint_funs_vec, coefvar,
                                             const, x_p,
                                             searchsize=n_pot_constr, out_size=4 * n_new_constr,
                                             use_bounds=use_bounds, bounds=bounds, use_higher_order=use_higher_order,
                                             h_higher_val=h_higher_val)

            if new_points_from_sup and val_list[-1] != -1000000000:
                nph2 = get_new_const_points_from_old(x_p, cost_fun_vec, hpoints, hvar, constraint_funs_vec, coefvar,
                                                     const,
                                                     size=n_new_constr, two_step_n=n_local, two_step_mult=two_step_mult,
                                                     line_n=n_line, line_mult=line_mult, eps_ball=eps_ball,
                                                     use_bounds=use_bounds, bounds=bounds, marginal_ball=marginal_ball,
                                                     marg_cdf=margs_cdfs, marg_ppf=margs_ppf,
                                                     use_higher_order=use_higher_order, h_higher_val=h_higher_val)
                nph = np.concatenate([nph, nph2, nph3, nph4], axis=0)
            else:
                nph2 = get_new_const_points(margs_rvs, cost_fun_vec, hpoints, hvar, constraint_funs_vec, coefvar, const,
                                            size=n_new_constr, searchsize=n_pot_constr, two_step=two_step,
                                            two_step_n=n_local, two_step_mult=two_step_mult, line_n=n_line,
                                            line_mult=line_mult, eps_ball=eps_ball, use_bounds=use_bounds,
                                            bounds=bounds, marginal_ball=marginal_ball, marg_cdf=margs_cdfs,
                                            marg_ppf=margs_ppf, use_higher_order=use_higher_order,
                                            h_higher_val=h_higher_val)
                nph = np.concatenate([nph, nph2, nph3, nph4], axis=0)

        # # Include optional step of getting sparse representation of all the points that are added
        # tquant = time()
        # nph = get_representatives(nph, int(round(len(nph) / 10)))

        # append set of all constraint points
        if return_primal or new_points_from_sup:
            all_points = np.concatenate([all_points, nph], axis=0)

        # sort them into marginal intervals
        ss_new = np.zeros(nph.shape)
        for j in range(D):
            ss_new[:, j] = np.searchsorted(hpoints[:, j], nph[:, j])
        ss_new = ss_new.astype(int)
        tsamp = time() - t0

        # solve the model with the new points, starting from the previous iteration
        t0 = time()
        if not use_higher_order:
            val, hvar, coefvar, const, m, h_var, coeff_var, constant = solve_given_model_new_constraint(cost_fun,
                                                                                                        constraints, m,
                                                                                                        h_var,
                                                                                                        coeff_var,
                                                                                                        constant, nph,
                                                                                                        ss_new, n_old,
                                                                                                        n1)
        else:
            val, hvar, coefvar, const, m, h_var, coeff_var, constant, h_higher, h_higher_val = \
                solve_given_model_new_constraint(cost_fun, constraints, m, h_var, coeff_var, constant, nph, ss_new,
                                                 n_old, n1, use_higher_order=use_higher_order,h_higher=h_higher)

        # append value list and other parameters for stopping criteria
        val_list.append(val)
        if val > -100000 and val != 0:
            val_list_stopping_crit.append(val)

        n_old += len(nph)
        tlp = time() - t0
        tlp_hist.append(tlp)
        if print_values:
            if show_time:
                print('Update iteration = ' + str(i) + ', Current value: ' + str(val) +
                      ', Time LP: ' + str(tlp) + ', Time Sampling: ' + str(tsamp))
            else:
                print('Update iteration = ' + str(i) + ', Current value: ' + str(val))

    # return optimal value and optionally optimizer
    if not return_primal:
        return val
    else:
        x_p, w_p = get_primal_sol(m, all_points, np.array(indices + list(range(start, len(m.Pi))), dtype=int))
        return val, [x_p, w_p]


# A simple bisection method, mainly copy-pasted from somewhere on the internet
def bisection(f, a, b, N):
    """Approximate solution of f(x)=0 on interval [a,b] by bisection method.

    Parameters
    ----------
    f : function
        The function for which we are trying to approximate a solution f(x)=0.
    a,b : numbers
        The interval in which to search for a solution. The function returns
        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
    N : (positive) integer
        The number of iterations to implement.

    Returns
    -------
    x_N : number
        The midpoint of the Nth interval computed by the bisection method. The
        initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
        midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iteration, the bisection method fails and return None.

    Examples
    --------
    f = lambda x: x**2 - x - 1
    bisection(f,1,2,25)
    1.618033990263939
    f = lambda x: (2*x - 1)*(x - 3)
    bisection(f,0,1,10)
    0.5
    """

    a_n = a
    b_n = b
    fan = f(a_n)
    fbn = f(b_n)
    for n in range(1, N+1):
        print('Current iteration: ' + str(n) + ' out of ' + str(N), 'Bounds: ', a_n, b_n, 'Values:', fan, fbn)
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if fan*f_m_n < 0:
            a_n = a_n
            b_n = m_n
            fbn = f_m_n
        elif fbn*f_m_n < 0:
            a_n = m_n
            b_n = b_n
            fan = f_m_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        else:
            print("Bisection method is getting inexact due to approximate evaluations of function (i.e., monotonicity is violated)...")
            continue
    return (a_n + b_n)/2, a_n, b_n


# a simple trisection method, mainly copy-pasted from somewhere on the internet
def convex_opt(f, a, b, N):
    """

    :param f: function to minimize, univariate and convex
    :param a: left interval
    :param b: right interval
    :param N: number of steps
    :return: list of points and values calculated
    """
    ml = 2*a/3+b/3
    mr = a/3 +2*b/3
    point_list = [ml, mr]
    print('Step 1')

    print('Initializing first value for trisection')
    vl = f(ml)
    print('Initializing second value for trisection')
    vr = f(mr)
    value_list = [vl, vr]

    for i in range(1, N):
        print('Step ' + str(i+1) + ' out of ' + str(N), 'Current points:', ml, mr, 'Current values:', vl, vr)
        if vr < vl:
            a = ml
            if mr - a > b - mr:
                ml = (a+mr)/2
                print('Calculating for point = ' + str(ml))
                vl = f(ml)
                point_list.append(ml)
                value_list.append(vl)
            else:
                ml = mr
                mr = (ml+b)/2
                vl = vr
                print('Calculating for point = ' + str(mr))
                vr = f(mr)
                point_list.append(mr)
                value_list.append(vr)
        else:
            b = mr
            if ml - a > b - ml:
                mr = ml
                ml = (a+mr)/2
                vr = vl
                print('Calculating for point = ' + str(ml))
                vl = f(ml)
                point_list.append(ml)
                value_list.append(vl)

            else:
                mr = (ml+b)/2
                print('Calculating for point = ' + str(mr))
                vr = f(mr)
                point_list.append(mr)
                value_list.append(vr)
    return point_list, value_list
