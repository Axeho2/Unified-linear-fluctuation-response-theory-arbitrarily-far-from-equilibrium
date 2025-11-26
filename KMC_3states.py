#%%

import os
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

params = {
            'text.usetex' : True,
            'font.family' : 'serif',
            'font.size' : 11,
            'text.latex.preamble' : '\n'.join([
                    r'\usepackage{amsfonts}',
                ]),
}
plt.rcParams.update(params)

#%%

# generate the trajectory of a continuous time markov chain
@nb.jit(nopython=True)
def traj(R, p_ini):
    n = len(R[0, :])    # total number of states
    for i in range(n):
        R[i, i] = 0     # modify R with diagonal elements 0
    # choose a state based on the initial prob distribution
    state = np.searchsorted(np.cumsum(p_ini), np.random.rand(), side="right")
    # calculate waiting time with a random number
    waiting_time = -np.log(np.random.rand()) / np.sum(R[:, state])
    while True:
        yield state, waiting_time
        # generate the next state
        state = np.searchsorted(np.cumsum(R[:, state]/np.sum(R[:, state])), np.random.rand(), side="right")
        # calculate the waiting time on the next state
        waiting_time = -np.log(np.random.rand()) / np.sum(R[:, state])

#%%

# initial probability
prob_ini = np.array([0.01, 1, 0.01])
prob_ini = prob_ini / np.sum(prob_ini)

# rate matrix
R = np.array([[   0.0,    30,   70],
              [    90,     0,   50],
              [    40,    10,    0]])
# R_{ii} = \sum_j R_{ji}
n = len(R[0])
R[np.arange(n), np.arange(n)] = -R.sum(axis=0)

#%%

# number of ensembles
ensemble = 8000  # 1000000
# time length
time_length = 10

time_step_num = 20
time_axis = np.linspace(0, time_length, time_step_num)

#%%

@nb.jit(nopython=True)
def sample(R, prob_ini, ensemble, time_length, time_step_num=50):
    # number of states
    n = len(R[0])
    # store dwelling time on diagonal elements and number of jumps on off-diagonal elements
    counting_ensemble = np.zeros((ensemble, n, n))

    # store data at each time step
    step_counting_ensemble = np.zeros((time_step_num, ensemble, n, n))

    # calculate time axis
    time_axis = np.linspace(0, time_length, time_step_num)

    for i in range(ensemble):
        # label total simulation time
        t = 0
        # generate a stochastic trjectory generator
        traj_generator = traj(R, prob_ini)
        # generate initial state and the dwelling time on it
        state, waiting_time = next(traj_generator)
        # record dwelling time
        counting_ensemble[i][state][state] += waiting_time
        # record total simulation time
        t += waiting_time
        for j in range(1, time_step_num):
            while t < time_axis[j]:
                old_state = state
                # generate the next state and the dwelling time on it
                state, waiting_time = next(traj_generator)
                # record dwelling time
                counting_ensemble[i][state][state] += waiting_time
                # record number of jumps
                counting_ensemble[i][state][old_state] += 1
                # record total simulation time
                t += waiting_time
            for k in range(n):
                # copy dwelling time at each time step
                step_counting_ensemble[j][i][k][k] = counting_ensemble[i][k][k]
                for l in range(n):
                    # copy jump number at each time step
                    step_counting_ensemble[j][i][k][l] = counting_ensemble[i][k][l]
    return step_counting_ensemble

#%%

@nb.jit(nopython=True)
def cov_Nij_Nkl(step_counting_ensemble, i, j, k, l):
    covariance = np.zeros(time_step_num)
    for step in range(1, time_step_num):
        N_ij_avg = 0
        N_kl_avg = 0
        N_ij_N_kl_avg = 0
        for ens in range(ensemble):
            N_ij = step_counting_ensemble[step][ens][i][j]
            N_kl = step_counting_ensemble[step][ens][k][l]
            N_ij_avg += N_ij
            N_kl_avg += N_kl
            N_ij_N_kl_avg += N_ij * N_kl
        N_ij_avg /= ensemble
        N_kl_avg /= ensemble
        N_ij_N_kl_avg /= ensemble
        covariance[step] = N_ij_N_kl_avg - N_ij_avg * N_kl_avg
    return covariance

@nb.jit(nopython=True)
def cov_Ti_Tj(step_counting_ensemble, i, j):
    covariance = np.zeros(time_step_num)
    for step in range(1, time_step_num):
        T_i_avg = 0
        T_j_avg = 0
        T_i_T_j_avg = 0
        for ens in range(ensemble):
            T_i = step_counting_ensemble[step][ens][i][i]
            T_j = step_counting_ensemble[step][ens][j][j]
            T_i_avg += T_i
            T_j_avg += T_j
            T_i_T_j_avg += T_i * T_j
        T_i_avg /= ensemble
        T_j_avg /= ensemble
        T_i_T_j_avg /= ensemble
        covariance[step] = T_i_T_j_avg - T_i_avg * T_j_avg
    return covariance

@nb.jit(nopython=True)
def cov_Nij_Tk(step_counting_ensemble, i, j, k):
    covariance = np.zeros(time_step_num)
    for step in range(1, time_step_num):
        N_ij_avg = 0
        T_k_avg = 0
        N_ij_T_k_avg = 0
        for ens in range(ensemble):
            N_ij = step_counting_ensemble[step][ens][i][j]
            T_k = step_counting_ensemble[step][ens][k][k]
            N_ij_avg += N_ij
            T_k_avg += T_k
            N_ij_T_k_avg += N_ij * T_k
        N_ij_avg /= ensemble
        T_k_avg /= ensemble
        N_ij_T_k_avg /= ensemble
        covariance[step] = N_ij_T_k_avg - N_ij_avg * T_k_avg
    return covariance

@nb.jit(nopython=True)
def get_avg_Nij(step_counting_ensemble, i, j):
    avg_N_ij = np.zeros(time_step_num)
    for step in range(1, time_step_num):
        N_ij_avg = 0
        for ens in range(ensemble):
            N_ij = step_counting_ensemble[step][ens][i][j]
            N_ij_avg += N_ij
        N_ij_avg /= ensemble
        avg_N_ij[step] = N_ij_avg
    return avg_N_ij

#%%

# perturbed edge [1]->[2]
ptb_edge = np.array([1, 2])
# observed activity [1]->[0]
obs_edge = np.array([1, 0])

# perturbation
ptb = 1
# perturbed R
ptb_R = R.copy()
ptb_R[ptb_edge[1], ptb_edge[0]] += ptb

#%%

fig, ax = plt.subplots(figsize=(2.5,2), layout='tight')

for i in range(30):
    print(i)
    step_counting_ensemble = sample(R, prob_ini, ensemble, time_length, time_step_num)
    ptb_step_counting_ensemble = sample(ptb_R, prob_ini, ensemble, time_length, time_step_num)

    response_2 = (get_avg_Nij(ptb_step_counting_ensemble, obs_edge[1], obs_edge[0]) - get_avg_Nij(step_counting_ensemble, obs_edge[1], obs_edge[0])) / ptb

    if i == 0:
        response_1 = cov_Nij_Nkl(step_counting_ensemble, ptb_edge[1], ptb_edge[0], obs_edge[1], obs_edge[0]) / R[ptb_edge[1], ptb_edge[0]] - cov_Nij_Tk(step_counting_ensemble, ptb_edge[1], ptb_edge[0], obs_edge[0])
        ax.plot(time_axis, response_1, color='blue', alpha=0.7)
    ax.scatter(time_axis, response_2, color='orange', marker='o', s=3, alpha=0.6)#, linestyle='--', alpha=0.6)

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\partial_{R_{21}} \langle N_{01} \rangle$')
ax.set_xlim((0, time_length))
# ax.set_xticks([0, 2, 4, 6, 8])

plt.tight_layout()
plt.grid()
plt.show()
path = os.path.abspath(os.path.dirname(__file__))
# plt.savefig(path + '/prob_edge_perturb.pdf')
