import networkx as nx
import matplotlib.pyplot as plt
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
        Q[i, i] = -np.sum(Q[i, :])

    return Q

# 参数设置
n = 100  # 节点数
max_degree = 5  # 单个节点的最大度数

# 生成图
G, points = generate_planar_graph_with_max_degree(n, max_degree)

# 生成速率矩阵
Q = generate_rate_matrix(G, n)

nonzero_indices = np.where(Q > 0)

# 将所有大于零的索引组合成 (行, 列) 对
nonzero_pairs = list(zip(nonzero_indices[0], nonzero_indices[1]))

ptb_edge = np.array([0, 0])
obs_edge = np.array([0, 0])
# 随机选择一个大于零的元素
selected_row, selected_col = random.choice(nonzero_pairs)
ptb_edge[0], ptb_edge[1] = int(selected_row), int(selected_col)
selected_row, selected_col = random.choice(nonzero_pairs)
obs_edge[0], obs_edge[1] = int(selected_row), int(selected_col)

print(ptb_edge, obs_edge)

pos = {i: points[i] for i in range(n)}  # 节点位置

# 使用 matplotlib 的 scatter 和 plot 绘制图
plt.figure(figsize=(2.5, 2.5))

# 绘制所有边（灰色）
for edge in G.edges():
    x1, y1 = pos[edge[0]]  # 边起点的坐标
    x2, y2 = pos[edge[1]]  # 边终点的坐标
    plt.plot([x1, x2], [y1, y2], color='gray', linewidth=2, alpha=0.8)

# 绘制高亮边（红色和蓝色）
x1, y1 = pos[ptb_edge[0]]  # 扰动边的起点坐标
x2, y2 = pos[ptb_edge[1]]  # 扰动边的终点坐标
plt.plot([x1, x2], [y1, y2], color='red', linewidth=3, label='Perturbation Edge')

x1, y1 = pos[obs_edge[0]]  # 观测边的起点坐标
x2, y2 = pos[obs_edge[1]]  # 观测边的终点坐标
plt.plot([x1, x2], [y1, y2], color='blue', linewidth=3, label='Observation Edge')

# 绘制节点
node_x = [pos[i][0] for i in range(n)]  # 节点的 x 坐标
node_y = [pos[i][1] for i in range(n)]  # 节点的 y 坐标
plt.scatter(node_x, node_y, s=7, zorder=5)

# 隐藏坐标轴上的数字和刻度
plt.xticks([])
plt.yticks([])

# 显示图形
plt.show()