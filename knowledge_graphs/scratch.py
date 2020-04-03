import scipy.sparse as sp
import numpy as np

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def main():
	file = 'graphs/basic_minecraft_lca_with_syn.npy'
	adj_matrix = np.load(file)
	print(normalize_adj(adj_matrix))
if __name__ == '__main__':
	main()