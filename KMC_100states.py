import networkx as nx
import numpy as np
import random
from scipy.spatial import Delaunay

random.seed(0)

def generate_planar_graph_with_max_degree(n, max_degree):
    # 生成随机点集
    points = [(random.random(), random.random()) for _ in range(n)]

    # 使用 Delaunay 三角剖分生成平面图
    tri = Delaunay(points)

    # 创建图并添加边
    G = nx.Graph()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                G.add_edge(simplex[i], simplex[j])

    # 限制节点的最大度数
    while True:
        # 找到所有度数超过 max_degree 的节点
        high_degree_nodes = [node for node, degree in G.degree() if degree > max_degree]
        if not high_degree_nodes:
            break  # 如果没有度数超过 max_degree 的节点，退出循环

        # 随机选择一个高度数节点
        node = random.choice(high_degree_nodes)

        # 找到与该节点相连的边
        edges = list(G.edges(node))

        # 随机删除一条边
        if edges:
            edge_to_remove = random.choice(edges)
            G.remove_edge(*edge_to_remove)

    return G, points

def generate_rate_matrix(G, n):
    # 初始化速率矩阵
    Q = np.zeros((n, n))

    # 为每条边随机赋予 transition rate
    for u, v in G.edges():
        rate_uv = random.uniform(10, 100)  # 正向速率
        rate_vu = random.uniform(10, 100)  # 逆向速率
        Q[u, v] = rate_uv
        Q[v, u] = rate_vu

    # 计算对角线元素
    for i in range(n):
        Q[i, i] = -np.sum(Q[:, i])

    return Q

# 参数设置
n = 100  # 节点数
max_degree = 5  # 单个节点的最大度数

# 生成图
G, points = generate_planar_graph_with_max_degree(n, max_degree)

# 生成速率矩阵
Q = generate_rate_matrix(G, n)

# 输出速率矩阵
# print("Rate Matrix Q:")
# print(Q)

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

prob_ini = np.random.rand(n)
prob_ini = prob_ini / np.sum(prob_ini)

# rate matrix
R = np.copy(Q)

#%%

# number of ensembles
ensemble = 10000  # 1000000
# time length
time_length = 3

time_step_num = 10
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

nonzero_indices = np.where(R > 0)

# 将所有大于零的索引组合成 (行, 列) 对
nonzero_pairs = list(zip(nonzero_indices[0], nonzero_indices[1]))

ptb_edge = np.array([0, 0])
obs_edge = np.array([0, 0])
# 随机选择一个大于零的元素
selected_row, selected_col = random.choice(nonzero_pairs)
ptb_edge[0], ptb_edge[1] = int(selected_row), int(selected_col)
selected_row, selected_col = random.choice(nonzero_pairs)
obs_edge[0], obs_edge[1] = int(selected_row), int(selected_col)

print(f'Perturbed edge: {ptb_edge[1]} -> {ptb_edge[0]}')
print(f'Observed edge: {obs_edge[1]} -> {obs_edge[0]}')

# perturbation
ptb = 1
# perturbed R
ptb_R = R.copy()
ptb_R[ptb_edge[1], ptb_edge[0]] += ptb

#%%

fig, ax = plt.subplots(figsize=(2.5,2.5), layout='tight')

response_1 = np.zeros(time_step_num)
response_2 = np.zeros(time_step_num)

for i in range(300):
    print(i)

    step_counting_ensemble = sample(R, prob_ini, ensemble, time_length, time_step_num)
    ptb_step_counting_ensemble = sample(ptb_R, prob_ini, ensemble, time_length, time_step_num)

    response_1 += cov_Nij_Nkl(step_counting_ensemble, ptb_edge[1], ptb_edge[0], obs_edge[1], obs_edge[0]) / R[ptb_edge[1], ptb_edge[0]] - cov_Nij_Tk(step_counting_ensemble, ptb_edge[1], ptb_edge[0], obs_edge[0])

    response_2 += (get_avg_Nij(ptb_step_counting_ensemble, obs_edge[1], obs_edge[0]) - get_avg_Nij(step_counting_ensemble, obs_edge[1], obs_edge[0])) / ptb

    if (i + 1) % 20 == 0:
        ax.plot(time_axis, response_1/10, color='blue', alpha=0.8)
        ax.scatter(time_axis, response_2/10, color='orange', marker='o', s=3, alpha=0.6)

        response_1 = np.zeros(time_step_num)
        response_2 = np.zeros(time_step_num)

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\partial_{R_{27 \leftarrow 15}} \langle N_{72 \leftarrow 67} \rangle$')
ax.set_xlim((0, time_length))
# ax.set_xticks([0, 2, 4, 6, 8])

plt.tight_layout()
plt.grid()
plt.show()
path = os.path.abspath(os.path.dirname(__file__))
# plt.savefig(path + '/prob_edge_perturb.pdf')
