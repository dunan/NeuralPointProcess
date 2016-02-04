import click
import numpy as np
import cvxpy as CVX
import pickle as P
import multiprocessing as MP
from datetime import datetime

def k(param, t1, t2):
    """Kernel function."""
    return np.exp(-1.0 * param * (t2 - t1))

def K(param, t1, t2):
    """Integrated kernel function."""
    return -1.0 * param * k(param, t1, t2)

def kernel_worker(params):
    '''Returns one kernel value computed at params[1][params[2]].'''
    kernel_param, times, idx_range = params
    return (idx_range, [np.sum(k(kernel_param, times[:idx], times[idx]))
                        for idx in range(*idx_range)])

def get_optimal_seq_for(num_times, num_cpu):
    '''Finds the best split of quadratic amount of work amount processors.'''
    ks = [ int(np.round(np.sqrt(i * 1.0 / num_cpu) * num_times))
           for i in range(0, num_cpu + 1) ]
    return zip(ks[:-1], ks[1:])


@click.command()
@click.option('--time',
    'time_data',
    type=click.File('r'),
    prompt='Time file',
    help='Time data file.')
@click.option('--save',
    'save_file',
    prompt='Save file',
    type=click.File('wb'),
    help='File to save the pickled results in.')
@click.option('--T',
    'period',
    default=-1,
    type=float,
    help='The periodicity of data (in the same units as the data provided)')
@click.option('--kernel-param',
    'kernel_param',
    default=1.0,
    type=float,
    help='The value of the kernel.')
@click.option('--quiet',
    'quiet',
    is_flag=True,
    help='Provide verbose output.')
def run(time_data, save_file, period, kernel_param, quiet):
    verbose = not quiet
    max_t = -1
    all_times = []
    all_times_delta = []
    if verbose:
        click.echo('Starting ...')
    for row_idx, time_row in enumerate(time_data):
        user_times = np.array([float(x) for x in time_row.split()])
        max_t = max(max_t, np.max(user_times))
        if period <= 0:
            time_data_raw = user_times[1:] - user_times[:-1]
        else:
            time_data_raw = user_times - [(x // period) * period for x in user_times]
        all_times.append(user_times)
        all_times_delta.append(time_data_raw)

    if verbose:
        click.echo('Finished reading data.')

    num_users = len(all_times)

    # Only two scalar variables
    mu = CVX.Variable()
    alpha = CVX.Variable()

    alpha_const = 0.0
    for times in all_times:
        alpha_const += np.sum(K(kernel_param, times, max_t))

    if verbose:
        click.echo('Creating objective expression.')

    # This is the survival term.
    LL_exp = -1.0 * (max_t * num_users * mu + alpha_const * alpha)

    # These will be the log-expressions
    kernels = []
    num_processes = MP.cpu_count() - 1
    with MP.Pool(processes=num_processes) as pool:
        for user_idx, times in enumerate(all_times):
            user_kernels = [None] * len(times)
            seq = get_optimal_seq_for(len(times), num_processes)
            start_time = datetime.now()
            for idx_range, kernel_vals in pool.imap(kernel_worker,
                    ((kernel_param, times, idx_range) for idx_range in seq)):
                user_kernels[idx_range[0]:idx_range[1]] = kernel_vals
                # if verbose:
                #     click.echo('User {}: {}-{} / {}, time taken = {:.2f} seconds'
                #             .format(user_idx + 1,
                #                     idx_range[0], idx_range[1],
                #                     len(times),
                #                     (datetime.now() - start_time).total_seconds()))
        kernels.append(user_kernels)

    for user_kernels in kernels:
        LL_exp += CVX.sum_entries(CVX.log(mu + alpha * user_kernels))

    # for times in all_times:
    #     for idx, t2 in enumerate(times):
    #         LL_exp += CVX.log(mu + alpha * np.sum(k(kernel_param, times[:idx], t2)))

    if verbose:
        click.echo('Solving optimization problem.')

    constraints = [mu >= 0.0, alpha >= 0.0]
    prob = CVX.Problem(CVX.Maximize(LL_exp), constraints)
    soln = prob.solve(verbose=verbose, solver='CVXOPT', kktsolver='robust')

    state = {
        'all_times': all_times,
        'all_times_delta': all_times_delta,
        'mu': mu,
        'alpha': alpha,
        'soln': soln,
        'max_t': max_t
    }
    P.dump(state, save_file)
    if verbose:
        click.echo('Done.')



