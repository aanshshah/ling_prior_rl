import gym
import numpy as np
import torch

class GraphEnv(gym.ObservationWrapper):
    def __init__(self, env, kg_dict, dont_crop_adj, one_hot_edges):
        super(GraphEnv, self).__init__(env)

        self.kg_dict = kg_dict
        self.entities = kg_dict['entities']
        self.num_entities = len(kg_dict['entities'])
        self.ord_to_idx = {ord(c): i + 1 for i, c in enumerate(self.entities)}
        self.reindex_func = np.vectorize((lambda x: self.ord_to_idx.get(x, 0)))

        self.one_hot_edges = one_hot_edges
        self.num_node_feats = self.kg_dict['num_node_feats']
        self.num_edge_feats = self.kg_dict['num_edge_feats']
        self.dont_crop_adj = dont_crop_adj

        self.edges = {idx: [] for idx in range(self.num_entities + 1)}

        for edge_dict in kg_dict['edges']:
            src_idx = self.entities.index(edge_dict['src'])
            dst_idx = self.entities.index(edge_dict['dst'])
            feature_idx = edge_dict['feature_idx']

            src_idx += 1
            dst_idx += 1
            self.edges[src_idx].append((dst_idx, feature_idx))

        kg_node_feats = np.zeros((self.num_entities + 1, self.num_node_feats), dtype=np.float32)
        for node_dict in kg_dict['nodes']:
            node_feats_type = node_dict['feature_type']
            node_idx = self.entities.index(node_dict['node']) + 1
            if node_feats_type == 'one_hot':
                feature_idx = node_dict['feature_idx']
                kg_node_feats[node_idx, feature_idx] = 1
            elif node_feats_type == 'list':
                kg_node_feats[node_idx, :] = node_dict['feats']
            else:
                raise ValueError('Unknown node feature type {}.'.format(node_feats_type))
        self.kg_node_feats = kg_node_feats

    def observation(self, obs):
        # obs = self.reindex_func(obs)
        obs = obs[0].astype(np.int64)
        kg_node_feats = self.kg_node_feats.copy()

        if self.dont_crop_adj:
            obs_vals_set = set(list(range(self.num_entities + 1)))
            obs_vals_list = list(obs_vals_set)
            other_entities = []
        else:
            obs_vals_set = set(obs.flatten())
            obs_vals_list = list(obs_vals_set)
            other_entities = [i for i in range(self.num_entities + 1) if i not in obs_vals_set]

        kg_node_feats[other_entities] = 0

        edges = []
        rels = []

        for src_idx in obs_vals_list:
            for dst_idx, rel in self.edges[src_idx]:
                if dst_idx in obs_vals_set:
                    edges.append((src_idx, dst_idx))

                    if self.one_hot_edges:
                        new_rel = np.zeros((self.num_edge_feats,), dtype=np.float32)
                        new_rel[rel] = 1
                        rel = new_rel

                    rels.append(rel)

        edges = np.array(edges).T
        rels = np.array(rels)

        return {'kg_node_feats': kg_node_feats, 'kg_edges': edges, 'kg_rels': rels, 'obs': obs}

    @staticmethod
    def batch_observations(obs_batch):
        obs_batch = [[elem['kg_node_feats'], elem['kg_edges'], elem['kg_rels'], elem['obs']] for elem in obs_batch]
        obs_batch = [[torch.from_numpy(t) for t in tl] for tl in obs_batch]

        kg_node_feats_list, edges_list, rels_list, obs_list = zip(*obs_batch)
        num_kg_nodes = kg_node_feats_list[0].size(0)

        new_edges_list = []
        for i, edges in enumerate(edges_list):
            edges = edges + num_kg_nodes * i
            new_edges_list.append(edges)
        edges_list = new_edges_list

        kg_node_feats = torch.stack(kg_node_feats_list, 0)
        edges = torch.cat(edges_list, 1)
        rels = torch.cat(rels_list, 0)
        obs = torch.stack(obs_list, 0)

        return {'kg_node_feats': kg_node_feats, 'kg_edges': edges, 'kg_rels': rels, 'obs': obs}
