from block_matrix import BlockMatrix
import poisson_problem
from matplotlib import pyplot as plt

sparse_data = []
full_data = []
ratio = []
x_values = [i for i in range(2, 100)]
for i in x_values:
    matrix = BlockMatrix(3, i)
    sparse_data += [matrix.eval_sparsity()[0] * 3]
    full_data += [(matrix.n-1)**(2*matrix.d)]
    ratio += [sparse_data[i-2] / full_data[i-2]]

_, ax1 = plt.subplots(figsize=(5, 5))
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("y", fontsize = 20, rotation = 0)
ax1.yaxis.set_label_coords(-0.01, 1)
plt.xlabel("x", fontsize = 20)
ax1.xaxis.set_label_coords(1.02, 0.025)
ax1.yaxis.get_offset_text().set_fontsize(20)
ax1.grid()

plt.plot(x_values, sparse_data, label = "sparse")
plt.plot(x_values, full_data, label = "full")
plt.plot(x_values, ratio, label = "ratio")

plt.legend(fontsize=20, loc="upper left")
plt.show()
