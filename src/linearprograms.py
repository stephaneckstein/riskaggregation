import numpy as np
from gurobipy import *

"""
This file includes code to run linear programs using Gurobi.
(relaxed dual formulation for multi-marginal OT with additional constraints)
"""


def dual_lp_new(margs_cdfs, cost_fun, constraints, marg_interv_co, const_points, const_points_into, outputflag=0,
                precompute_muik=0, muik_p=(), model_out=0, use_higher_order=0, use_precomp_higher=0, price_high=(),
                n_calc_higher=1000, marg_ppf=(), high_tol=0, mt=0, num_focus=0):
    """

    :param margs_cdfs: list with d elements, each being a cumulative distribution function of a marginal
    :param cost_fun: cost function, takes as input a single d-dimensional vector (i.e., not vectorized)
    :param constraints: list of 3 elements, each being list of same size, [f_1, ...], [c_1, ...], [b_1, ...]
    enables dual function trading alpha_i f_i for price alpha_i c_i,
    where alpha_i <= 0, if b_i == -1, alpha_i >= 0, if b_i == 0, and alpha_i arbitrary, else
    :param marg_interv_co: list with d array, each encodes boundaries of intervals for discretization of a marginal
    :param const_points: [n, d] array of points, which encode the points where the dual inequality constraint is tested
    :param const_points_into: array of integers of size [n, d], where [i, j]-th entry gives the position of
    const_points[i, j] in the intervals encoded by marg_interv_co
    :param outputflag: 0 or 1, whether to print gurobi output or not
    :param precompute_muik: 0/1, whether prob. masses of each interval corresponding to marg_interv_co are precomputed
    :param muik_p: if precompute_muik==1, muik_p must be the precomputed values
    :param model_out: 0/1, whether to output the solved gurobi model
    :param use_higher_order: some integer. If 0, then only piece-wise constant functions used.
    If 1, then all piece-wise linear functions used, if 2, then all piece-wise polynomial of order 2 are used, etc
    :param use_precomp_higher: 0/1, whether the integrals of the monomials up to use_higher_order are precomputed
    :param price_high: if use_precomp_higher==1, then this must encode the respective integral values
    :param n_calc_higher: if not use_precomp_higher, this specifies the number of sample points to compute price_high
    :param marg_ppf: as margs_cdfs, but quantile functions. Only used if use_precomp_higher=0 and use_higher_order > 0
    :param high_tol: whether to use high tolerance within gurobi
    :param mt: if nonzero, sets timelimit within gurobi
    :param num_focus: numerical focus variable for gurobi (higher values give more precision. Sometimes important for
    reinitializing the problem)
    :return: if not model_out, returns optimal value, optimal h values, and optimal constraint multipliers
    if model out, additionally returns gurobi variables
    """
    # read out parameters
    const_points_into = const_points_into.astype(int)
    ct_funs = constraints[0]
    ct_prices = np.array(constraints[1])
    ct_short = constraints[2]
    K = len(ct_funs)
    n1 = len(marg_interv_co)+1
    (n2, d) = const_points.shape

    # obtain probability mass at each interval
    if precompute_muik:
        muik = muik_p
    else:
        muik = np.zeros([n1, d])
        for i in range(d):
            probs = margs_cdfs[i](marg_interv_co[:, i])
            probs = np.insert(probs, [0, len(probs)], [0, 1])
            muik[:, i] = probs[1:] - probs[:-1]

    # initialize model
    m = Model('Primal')
    if num_focus == 1:
        m.setParam('NumericFocus', 3)
    if outputflag == 0:
        m.setParam('OutputFlag', 0)
    if high_tol == 1:
        m.setParam('OptimalityTol', 10**-3)
        m.setParam('FeasibilityTol', 10**-7)
        print('Using high tolerance')
    if mt:
        m.setParam('TimeLimit', mt)
    h_var = m.addVars(n1, d, name='h_var', lb=-float('inf'))
    coeff_var = m.addVars(K, name='coeff_var', lb=-float('inf'))
    constant = m.addVars(1, name='constant', lb=-float('inf'))

    # if higher order polynomials in dual are used, specify integral values
    if use_higher_order > 0:
        n_h = use_higher_order
        h_higher = m.addVars(n1, d, n_h, name='higher')
        if not use_precomp_higher:
            price_high = np.zeros([n1, d, n_h])
            for i in range(d):
                probs = margs_cdfs[i](marg_interv_co[:, i])
                probs = np.insert(probs, [0, len(probs)], [0, 1])
                for j in range(n1):
                    lhs_h = probs[j]
                    rhs_h = probs[j+1]
                    p_h = np.linspace(lhs_h, rhs_h, n_calc_higher+1)
                    e_h = marg_ppf[i]((p_h[1:] + p_h[:-1])/2)
                    # NOTE: Have to supply ppf to use higher order; Or precompute!
                    for k in range(n_h):
                        price_high[j, i, k] = np.mean(e_h**(k+1))

    # setting objective value for dual
    obj = quicksum([h_var[i, j] * muik[i, j] for i in range(n1) for j in range(d)]) + \
          quicksum([coeff_var[i] * ct_prices[i] for i in range(K)]) + constant[0]
    if use_higher_order > 0:
        obj += quicksum(([h_higher[i, j, k] * muik[i, j] * price_high[i, j, k] for i in range(n1) for j in range(d)
                          for k in range(n_h)]))

    # inequality constraint
    if not use_higher_order:
        for i in range(n2):
            m.addConstr(quicksum([h_var[const_points_into[i, j], j]
                                  for j in range(d)]) + quicksum([coeff_var[j] * ct_funs[j](const_points[i, :])
                         for j in range(K)]) + constant[0] >= cost_fun(const_points[i, :]), name='ineq_constr_'+str(i))
    else:
        for i in range(n2):
            m.addConstr(quicksum([h_var[const_points_into[i, j], j] for j in range(d)])
                        + quicksum([h_higher[const_points_into[i, j], j, k] * const_points[i, j]**(k+1)
                                    for j in range(d) for k in range(n_h)])
                        + quicksum([coeff_var[j] * ct_funs[j](const_points[i, :])
                         for j in range(K)]) + constant[0] >= cost_fun(const_points[i, :]), name='ineq_constr_'+str(i))

    # short sell constraints
    for i in range(K):
        if ct_short[i] == 0:
            m.addConstr(coeff_var[i] >= 0, name='lb_'+str(i))
        if ct_short[i] == -1:
            m.addConstr(coeff_var[i] <= 0, name='ub_'+str(i))

    # optimize
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    # return output
    try:
        objective_val = m.ObjVal
    except:
        m.setParam('DualReductions', 0)
        m.setParam('InfUnbdInfo', 1)
        m.optimize()
        objective_val = -10**9
        m.setParam('InfUnbdInfo', 0)

    if not model_out:
        return objective_val, np.array([[h_var[i, j].x for j in range(d)] for i in range(n1)]), \
               [coeff_var[i].x for i in range(K)], constant[0].x
    else:
        if not use_higher_order:
            return objective_val, np.array([[h_var[i, j].x for j in range(d)] for i in range(n1)]), \
                   [coeff_var[i].x for i in range(K)], constant[0].x, m, h_var, coeff_var, constant
        else:
            return objective_val, np.array([[h_var[i, j].x for j in range(d)] for i in range(n1)]), \
                   [coeff_var[i].x for i in range(K)], constant[0].x, m, h_var, coeff_var, constant, h_higher, \
                   np.array([[[h_higher[i, j, k].x for k in range(n_h)] for j in range(d)] for i in range(n1)])


def solve_given_model_new_constraint(cost_fun, constraints, m, h_var, coeff_var, constant, new_points, new_sort_into,
                                     n_old, n1, model_out=1, use_higher_order=0, h_higher=()):
    """
    takes a model as output from dual_lp_new (or itself), adds constraints, and solves the model again
    :param cost_fun: as in dual_lp_new
    :param constraints: as in dual_lp_new
    :param m: model as output from previous iteration
    :param h_var: model variable as output from previous iteration (piece wise constant functions)
    :param coeff_var: model variable as output from previous iteration (additional constraint multiplier)
    :param constant: constant model variable
    :param new_points: as const_points in dual_lp_new; additional points to add to test inequalit constraint
    :param new_sort_into: as const_points_into in dual_lp_new for new_points
    :param n_old: number of inequality constraint points so far (supplied for labeling of inequality constraints)
    :param n1: length of intervals (i.e., length of marg_interv_co form dual_lp_new)
    :param model_out: 0/1 whether to return model or not
    :param use_higher_order: whether to use higher order polynomials or not
    :param h_higher: model variable as output from previous iteration (piece wise polynomial functions)
    :return: same as dual_lp_new
    """

    # read out input
    (n_new, d) = new_points.shape
    K = len(constraints[0])

    # add new constraints
    if not use_higher_order:
        for i in range(n_new):
            m.addConstr(quicksum([h_var[new_sort_into[i, j], j] for j in range(d)]) + constant[0] +
                        quicksum([coeff_var[j] * constraints[0][j](new_points[i, :])
                                  for j in range(K)]) >= cost_fun(new_points[i, :]),
                            name='ineq_constr_' + str(n_old + i))
    else:
        for i in range(n_new):
            m.addConstr(quicksum([h_var[new_sort_into[i, j], j] for j in range(d)])
                        + quicksum([h_higher[new_sort_into[i, j], j, k] * new_points[i, j]**(k+1)
                                    for j in range(d) for k in range(use_higher_order)])
                        + quicksum([coeff_var[j] * constraints[0][j](new_points[i, :])
                         for j in range(K)]) + constant[0] >= cost_fun(new_points[i, :]),
                        name='ineq_constr_'+str(n_old + i))

    # optimize; automatically uses previous optimal solution as starting point. Drastically faster than reinitializing
    m.optimize()

    # return output
    try:
        objective_val = m.ObjVal
    except:
        m.setParam('DualReductions', 0)
        m.setParam('InfUnbdInfo', 1)
        m.optimize()
        objective_val = -10 ** 9
        m.setParam('InfUnbdInfo', 0)

    if not model_out:
        if not use_higher_order:
            return objective_val, np.array([[h_var[i, j].x for j in range(d)] for i in range(n1+1)]), \
                   [coeff_var[i].x for i in range(K)], constant[0].x
        else:
            return objective_val, np.array([[h_var[i, j].x for j in range(d)] for i in range(n1+1)]), \
                   [coeff_var[i].x for i in range(K)], constant[0].x, \
                   np.array([[[h_higher[i, j, k].x for k in range(use_higher_order)] for j in range(d)]
                             for i in range(n1+1)])

    else:
        if not use_higher_order:
            return objective_val, np.array([[h_var[i, j].x for j in range(d)] for i in range(n1+1)]), \
                   [coeff_var[i].x for i in range(K)], constant[0].x, m, h_var, coeff_var, constant
        else:
            return objective_val, np.array([[h_var[i, j].x for j in range(d)] for i in range(n1+1)]), \
                   [coeff_var[i].x for i in range(K)], constant[0].x, m, h_var, coeff_var, constant, \
                   h_higher, np.array([[[h_higher[i, j, k].x for k in range(use_higher_order)] for j in range(d)]
                                       for i in range(n1+1)])
