import numpy as np
from scipy.spatial import distance_matrix

"""
The programs herein take output from a relaxed linear program with finitely many inequality constraints, and try to 
find points that violate these inequality constraints on some other parts of the space
"""


def eval_points(cost_fun_vec, hvals, constraint_funs_vec, coeff_vals, constant_val, new_points, new_points_into,
                use_higher_order=0, h_higher_val=()):
    """
    evaluates values occurring in dual constraint
    sum_{i=1, ..., d} h_i(x_{j, i}) + sum_{i=1, ..., K} coeff_i f_i(x_j) + constant - f_cost(x_j)
    (for use_higher_order=0, otherwise includes h_{i, p} x^p for p >= 1 as well)
    :param cost_fun_vec: cost function vectorized, i.e., takes input [n, d] and returns [n]-array
    :param hvals: values as output from dual_lp_new; i.e., multiplier for piecewise constant function on each interval
    :param constraint_funs_vec: vectorized constraint functions
    :param coeff_vals: multiplier values for constraint functions
    :param constant_val: constant for inequality constraint
    :param new_points: points to evaluate constraint term
    :param new_points_into: integer positions of points inside the intervals
    :param use_higher_order: integer, if > 0, uses higher order polynomials on intervals as well
    :param h_higher_val: corresponding multipliers for the higher order polynomials
    :return: returns values of size len(new_points); should be >= zero if constraint is satisfied
    """

    # read out input
    (n1, d) = hvals.shape
    n2 = len(new_points)
    k = len(coeff_vals)

    # evaluate h functions
    hplugvals = np.zeros([n2, d])
    for i in range(d):
        hplugvals[:, i] = hvals[new_points_into[:, i], i]
    if use_higher_order:
        h_higher_plug = np.zeros([n2, d, use_higher_order])
        for i in range(d):
            for kind in range(use_higher_order):
                h_higher_plug[:, i, kind] = h_higher_val[new_points_into[:, i], i, kind] * new_points[:, i] ** (
                        kind + 1)

    # evaluate constraint functions, objective function, and return values
    const_plug_vals = np.zeros([n2, k])
    for i in range(k):
        const_plug_vals[:, i] = coeff_vals[i] * constraint_funs_vec[i](new_points)
    fun_vals = cost_fun_vec(new_points)
    if not use_higher_order:
        return constant_val - fun_vals + np.sum(hplugvals, axis=1) + np.sum(const_plug_vals, axis=1)
    else:
        return constant_val - fun_vals + np.sum(hplugvals, axis=1) + np.sum(h_higher_plug, axis=(1, 2)) + np.sum(
            const_plug_vals, axis=1)


def gen_local_product_around(points, marg_cdf, marg_ppf, eps_ball, eps_cut_off_quantile=10 ** -6, multiplier=1):
    """
    generates points according to a local product measure with respect to the marginals around given points.
    In other words, generates randomly uniform points around the probability levels (i.e., on the copula level) for
    each point and then projects those back to the quantile level
    :param points: initial points to generate new points around, size [n, d]
    :param marg_cdf: cumulative distributions functions of marginals
    :param marg_ppf: quantile functions of marginals
    :param eps_ball: size of ball on the probability level
    :param eps_cut_off_quantile: cuts off before 0 and 1 at probability level to exclude -infty, infty values
    :param multiplier: determines how many points are generated for each initial point
    :return: array of new points of size [n*multiplier, d]
    """

    # read out input
    if len(points.shape) == 1:
        points.reshape(1, -1)
    (n, d) = points.shape

    # get probability levels
    q_levels = np.zeros([n, d])
    for i in range(d):
        q_levels[:, i] = marg_cdf[i](points[:, i])

    # generate new probability levels around given ones
    if multiplier > 1:
        q_levels = np.tile(q_levels, (multiplier, 1))
        n *= multiplier
    q_levels += eps_ball * (2 * np.random.random_sample([n, d]) - 1)
    q_levels = np.maximum(eps_cut_off_quantile, np.minimum(q_levels, 1 - eps_cut_off_quantile))

    # transform probability levels to quantiles and return
    new_points = np.zeros([n, d])
    for i in range(d):
        new_points[:, i] = marg_ppf[i](q_levels[:, i])
    return new_points


def get_new_const_points(marg_sampler, cost_fun_vec, marg_interv_co, hvals, constraint_funs_vec, coeff_vals,
                         constant_val,
                         size=10, searchsize=10 ** 4, two_step=0, eps_ball=0.1, two_step_n=10 ** 3, two_step_mult=1,
                         line_n=10 ** 2, line_mult=1, use_bounds=0, bounds=(), marginal_ball=0, marg_cdf=(),
                         marg_ppf=(), use_higher_order=0, h_higher_val=()):
    """

    :param marg_sampler: function to sample points from each marginal
    :param cost_fun_vec: cost function in vector form (input [n, d] shaped)
    :param marg_interv_co: intervals for marginals
    :param hvals: optimal values for h functions as output from dual_lp_new
    :param constraint_funs_vec: vectorized constraint functions (input [n, d] shaped)
    :param coeff_vals: values for constraint functions as output from dual_lp_new
    :param constant_val: some constant value
    :param size: number of points from global search to use for local search
    :param searchsize: number of overall global points considered
    :param two_step: 0/1 whether to use local search or not
    :param eps_ball: epsilon value of local ball
    :param two_step_n: how many local points to consider
    :param two_step_mult: how many local points to use for each global point
    :param line_n: how many points on a line segment to consider between global and local good points
    :param line_mult: how many points on the line segment to output
    :param use_bounds: 0/1 whether to cut off tails of points or not
    :param bounds: where to cut tails
    :param marginal_ball: 1, if use copula ball (cf gen_local_product_around), and 0, then simply uniform ball
    :param marg_cdf: cumulative distribution functions of marginals
    :param marg_ppf: quantile functions of marginals
    :param use_higher_order: whether to use higher order polynomials on intervals or not
    :param h_higher_val: if use_higher_order, these are the corresponding multiplier for each interval
    :return: new points of shape [size*two_step_mult*line_mult, d]
    """

    # read out input
    (n1, d) = hvals.shape
    n2 = searchsize
    k = len(coeff_vals)

    # generate global points from product measure of marginals
    new_pot_points = np.zeros([n2, d])
    for i in range(d):
        new_pot_points[:, i] = marg_sampler[i](n2)
    if use_bounds:
        new_pot_points = np.maximum(new_pot_points, bounds[0:1, :])
        new_pot_points = np.minimum(new_pot_points, bounds[1:2, :])

    # sort global points into intervals
    new_points_into = np.zeros([n2, d])
    for i in range(d):
        new_points_into[:, i] = np.searchsorted(marg_interv_co[:, i], new_pot_points[:, i])

    # evaluate constraint and choose best points accordingly
    new_points_into = new_points_into.astype(int)
    scores_new_points = eval_points(cost_fun_vec, hvals, constraint_funs_vec, coeff_vals, constant_val, new_pot_points,
                                    new_points_into, use_higher_order=use_higher_order, h_higher_val=h_higher_val)
    ind = np.argpartition(-scores_new_points, -size)[-size:]

    # start local search
    if two_step == 1:
        points_good = new_pot_points[ind, :]
        points_out = np.zeros([0, d])  # will be the points that are returned
        for i in range(size):  # iterate over good points from global search

            # generate local points
            if marginal_ball == 1:
                points_around = gen_local_product_around(points_good[i:i + 1, :], marg_cdf, marg_ppf, eps_ball,
                                                         multiplier=two_step_n)
            else:
                perturb = 2 * np.random.random_sample([two_step_n, d]) - 1
                points_around = points_good[i:i + 1, :] + eps_ball * perturb
            if use_bounds:
                points_around = np.maximum(points_around, bounds[0:1, :])
                points_around = np.minimum(points_around, bounds[1:2, :])

            # sort local points into intervals
            points_around_into = np.zeros([two_step_n, d])
            for j in range(d):
                points_around_into[:, j] = np.searchsorted(marg_interv_co[:, j], points_around[:, j])
            points_around_into = points_around_into.astype(int)

            # evaluate inequality constraint and choose best points accordingly
            scores_around = eval_points(cost_fun_vec, hvals, constraint_funs_vec, coeff_vals, constant_val,
                                        points_around,
                                        points_around_into, use_higher_order=use_higher_order,
                                        h_higher_val=h_higher_val)
            ind_around = np.argpartition(-scores_around, -two_step_mult)[-two_step_mult:]
            even_better_points = points_around[ind_around, :]

            # perform line search and append point_out with best points
            lin_part = np.linspace(0, 1, line_n).reshape(-1, 1)
            for j in range(two_step_mult):
                points_line = lin_part * points_good[i:i + 1, :] + (1 - lin_part) * even_better_points[j:j + 1, :]
                points_line_into = np.zeros([line_n, d])
                for k_ind in range(d):
                    points_line_into[:, k_ind] = np.searchsorted(marg_interv_co[:, k_ind], points_line[:, k_ind])
                points_line_into = points_line_into.astype(int)
                scores_line = eval_points(cost_fun_vec, hvals, constraint_funs_vec, coeff_vals, constant_val,
                                          points_line,
                                          points_line_into, use_higher_order=use_higher_order,
                                          h_higher_val=h_higher_val)
                ind_line = np.argpartition(-scores_line, -line_mult)[-line_mult:]
                points_out = np.concatenate([points_out, points_line[ind_line, :]], axis=0)
        return points_out
    else:
        return new_pot_points[ind, :]


def get_new_const_points_dist(marg_sampler, cost_fun_vec, marg_interv_co, hvals, constraint_funs_vec, coeff_vals,
                              constant_val, constr_points_so_far, searchsize=10 ** 4, out_size=100, use_bounds=0,
                              bounds=(), use_higher_order=0, h_higher_val=(), max_matrix_size=200000000,
                              max_pot_points=1000, max_so_far=100000):
    """
    this function should generate new points that satisfy both, 1) they violate the constraint, and 2) they are far
    away from previous points considered as constraint points.
    In hindsight this function is not super useful. No matter how far away a point is from the current points, if the
    violation of the inequality constraint is only very minor, then of course simply increasing the constant very
    slightly can make up for it...
    However, it may still make sense since this kind of constraint function at least does not just add points at the
    same spot locally which may be fixed by some local function adjustments that are very cheap...
    :param marg_sampler: functions to sample points from marginals
    :param cost_fun_vec: cost function vectorized (input [n, d] shaped)
    :param marg_interv_co: intervals for marginals
    :param hvals: values for h functions as output by dual_lp_new
    :param constraint_funs_vec: constraint functions vectorized (input [n, d] shaped)
    :param coeff_vals: multiplier values for constraint functions
    :param constant_val: some constant value
    :param constr_points_so_far: constraint points that are included in the optimization problem so far
    :param searchsize: number of global points that are initially generated
    :param out_size: number of total points returned
    :param use_bounds: 0/1 whether to cut off tails or not
    :param bounds: where to cut off tails
    :param use_higher_order: 0/1 whether to use higher order polynomials on intervals or not
    :param h_higher_val: correpsonding values for higher order polynomials
    :param max_matrix_size: maximum size of matrix for memory
    :param max_pot_points: max size of potential points for memory
    :param max_so_far: max size of good points to consider for memory
    :return: [out_size, d] array of new points
    """

    # generate global points from product measure
    (n1, d) = hvals.shape
    n2 = searchsize
    k = len(coeff_vals)
    new_pot_points = np.zeros([n2, d])
    for i in range(d):
        new_pot_points[:, i] = marg_sampler[i](n2)
    if use_bounds:
        new_pot_points = np.maximum(new_pot_points, bounds[0:1, :])
        new_pot_points = np.minimum(new_pot_points, bounds[1:2, :])

    # sort global points into intervals
    new_points_into = np.zeros([n2, d])
    for i in range(d):
        new_points_into[:, i] = np.searchsorted(marg_interv_co[:, i], new_pot_points[:, i])
    new_points_into = new_points_into.astype(int)
    scores_new_points = eval_points(cost_fun_vec, hvals, constraint_funs_vec, coeff_vals, constant_val, new_pot_points,
                                    new_points_into, use_higher_order=use_higher_order, h_higher_val=h_higher_val)
    min_val = np.min(scores_new_points)

    # take points where constraint is strongly violated
    inds_very_neg = np.where((scores_new_points < -0.01) + (scores_new_points < 0.8 * min_val))[0]
    if len(inds_very_neg) < 10:
        inds_very_neg = np.where(scores_new_points <= 0)[0]
    if len(inds_very_neg) < 10:
        inds_very_neg = np.where(scores_new_points <= 10 ** 9)[0]
    points_very_neg = new_pot_points[inds_very_neg, :]

    # cap the size of the points considered for memory reasons
    if len(points_very_neg) * len(constr_points_so_far) > max_matrix_size:
        if len(points_very_neg) > max_pot_points:
            points_very_neg = points_very_neg[
                              np.random.choice(np.arange(0, len(points_very_neg), dtype=int), size=max_pot_points,
                                               replace=False), :]
        if len(constr_points_so_far) > max_so_far:
            constr_points_so_far_2 = constr_points_so_far[
                                     np.random.choice(np.arange(0, len(constr_points_so_far), dtype=int),
                                                      size=max_so_far, replace=False), :]
        else:
            constr_points_so_far_2 = constr_points_so_far
    else:
        constr_points_so_far_2 = constr_points_so_far

    # calculate distance matrices between new potential points and the considered points so far
    dist_mat = distance_matrix(points_very_neg, constr_points_so_far_2, p=1)
    dme = np.min(dist_mat, axis=1)  # respective distance to the set of points

    # generate points to return; start with the one furthest away from the considered points, add it to points, repeat
    out_points = np.zeros([0, d])
    for i in range(out_size):
        ind1 = np.argmax(dme)
        out_points = np.concatenate([out_points, points_very_neg[ind1:ind1 + 1, :]], axis=0)
        dme = np.minimum(dme, np.sum(np.abs(points_very_neg - points_very_neg[ind1:ind1 + 1, :]), axis=1))

    return out_points


def get_new_const_points_from_old(curr_support, cost_fun_vec, marg_interv_co, hvals, constraint_funs_vec, coeff_vals,
                                  constant_val, size=10, eps_ball=0.1, two_step_n=10 ** 3, two_step_mult=1,
                                  line_n=10 ** 2, line_mult=1, use_bounds=0, bounds=(), marginal_ball=0, marg_cdf=(),
                                  marg_ppf=(), use_higher_order=0, h_higher_val=()):
    """
    Generates new points from a given set of points that are deemed very promising to look nearby, i.e., usually
    the support of the current primal optimizer.
    Particularly useful for finetuning of optimizers, helps lead to optimal arrangements on a fine scale
    :param curr_support: points to look nearby, i.e., current support
    :param cost_fun_vec: cost function vectorized ([n, d] shaped input)
    :param marg_interv_co: intervals for marginals
    :param hvals: values for piecewise constant functions on intervals
    :param constraint_funs_vec: constraint functions vectorized ([n, d] shaped inputs)
    :param coeff_vals: coefficients for constraint functions
    :param constant_val: constant vlaue
    :param size: integer, effectively used to bound curr support if it is too large (reduces local size then)
    :param eps_ball: ball of local search
    :param two_step_n: how many local points to consider for each given point
    :param two_step_mult: how many local points are chosen among the considered ones
    :param line_n: number of points on line segment between global and local points
    :param line_mult: number of points chosen from line segment
    :param use_bounds: 0/1 whether to cut off tails or not
    :param bounds: where to cut off tails
    :param marginal_ball: whether to use local product measure, c.f. gen_local_product_around
    :param marg_cdf: cumulative distribution function for marginals
    :param marg_ppf: quantile functions for marginals
    :param use_higher_order: 0/1 whether to use higher order polynomials on intervals or not
    :param h_higher_val: corresponding values if use_higher_order > 0
    :return: array of shape [?, d] (? depends on curr_support, size, two_step_mult and line_mult)
    """

    # read out input
    (n1, d) = hvals.shape
    (n2, d) = curr_support.shape
    k = len(coeff_vals)
    new_pot_points = curr_support
    if use_bounds:
        new_pot_points = np.maximum(new_pot_points, bounds[0:1, :])
        new_pot_points = np.minimum(new_pot_points, bounds[1:2, :])

    points_out = np.zeros([0, d])
    two_step_mult = min(two_step_mult, two_step_n)

    # if current points are too large, randomly choose a subsample
    if len(new_pot_points) > size:
        inds_here = np.random.choice(list(range(0, len(new_pot_points))), size=size, replace=False)
        points_good = new_pot_points[inds_here, :]
    else:
        points_good = new_pot_points

    # Below is the old way, but I don't quite remember why I did it that way and it seems quite stupid in hindsight
    # if len(new_pot_points) > size:
    #     two_step_n = int(max(1, np.round(two_step_n/min(100, len(points_good)/size))))

    # iterate over global points considered
    for i in range(len(points_good)):

        # generate local points around
        if marginal_ball == 1:
            points_around = gen_local_product_around(points_good[i:i + 1, :], marg_cdf, marg_ppf, eps_ball,
                                                     multiplier=two_step_n)
        else:
            perturb = 2 * np.random.random_sample([two_step_n, d]) - 1
            points_around = points_good[i:i + 1, :] + eps_ball * perturb
        if use_bounds:
            points_around = np.maximum(points_around, bounds[0:1, :])
            points_around = np.minimum(points_around, bounds[1:2, :])

        # sort points into interval
        points_around_into = np.zeros([two_step_n, d])
        for j in range(d):
            points_around_into[:, j] = np.searchsorted(marg_interv_co[:, j], points_around[:, j])
        points_around_into = points_around_into.astype(int)

        # evaluate inequality constraint term and pick the best points accordingly
        scores_around = eval_points(cost_fun_vec, hvals, constraint_funs_vec, coeff_vals, constant_val, points_around,
                                    points_around_into, use_higher_order=use_higher_order, h_higher_val=h_higher_val)
        ind_around = np.argpartition(-scores_around, -two_step_mult)[-two_step_mult:]
        even_better_points = points_around[ind_around, :]

        # search on line segment between good local and initial global point and add points to points_out
        lin_part = np.linspace(0, 1, line_n).reshape(-1, 1)
        for j in range(two_step_mult):
            points_line = lin_part * points_good[i:i + 1, :] + (1 - lin_part) * even_better_points[j:j + 1, :]
            points_line_into = np.zeros([line_n, d])
            for k_ind in range(d):
                points_line_into[:, k_ind] = np.searchsorted(marg_interv_co[:, k_ind], points_line[:, k_ind])
            points_line_into = points_line_into.astype(int)
            scores_line = eval_points(cost_fun_vec, hvals, constraint_funs_vec, coeff_vals, constant_val, points_line,
                                      points_line_into, use_higher_order=use_higher_order, h_higher_val=h_higher_val)
            ind_line = np.argpartition(-scores_line, -line_mult)[-line_mult:]
            points_out = np.concatenate([points_out, points_line[ind_line, :]], axis=0)

    # if, for some reason, points_out is larger than size*two_step_mult*line_mult, then we take again the best points
    if len(points_out) > size * two_step_mult * line_mult:
        points_out_into = np.zeros(points_out.shape)
        for i in range(d):
            points_out_into[:, i] = np.searchsorted(marg_interv_co[:, i], points_out[:, i])
        points_out_into = points_out_into.astype(int)
        scores_points_out = eval_points(cost_fun_vec, hvals, constraint_funs_vec, coeff_vals, constant_val, points_out,
                                        points_out_into, use_higher_order=use_higher_order, h_higher_val=h_higher_val)
        ind = np.argpartition(-scores_points_out, -min((len(scores_points_out)), size * two_step_mult * line_mult))[
              -min(len(scores_points_out), size * two_step_mult * line_mult):]
        points_good = points_out[ind, :]
    return points_good
