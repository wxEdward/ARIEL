import os.path as osp
import pickle as pkl

import torch
import random
import numpy as np
from torch_geometric.data import InMemoryDataset, Data

from copy import deepcopy
from itertools import repeat

class SPMotif(InMemoryDataset):
    splits = ['train', 'val', 'test']

    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):

        assert mode in self.splits
        self.mode = mode
        super(SPMotif, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy']

    @property
    def processed_file_names(self):
        return ['SPMotif_train.pt', 'SPMotif_val.pt', 'SPMotif_test.pt']

    def download(self):
        if not osp.exists(osp.join(self.raw_dir, 'raw', 'SPMotif_train.npy')):
            print("raw data of `SPMotif` doesn't exist, please redownload from our github.")
            raise FileNotFoundError

    def process(self):

        idx = self.raw_file_names.index('{}.npy'.format(self.mode))
        edge_index_list, label_list, ground_truth_list, role_id_list, pos = np.load(
            osp.join(self.raw_dir, self.raw_file_names[idx]), allow_pickle=True)
        data_list = []
        for idx, (edge_index, y, ground_truth, z, p) in enumerate(
                zip(edge_index_list, label_list, ground_truth_list, role_id_list, pos)):
            edge_index = torch.from_numpy(edge_index)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            node_idx = torch.unique(edge_index)
            assert node_idx.max() == node_idx.size(0) - 1
            x = torch.zeros(node_idx.size(0), 4)
            index = [i for i in range(node_idx.size(0))]
            x[index, z] = 1
            x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).unsqueeze(dim=0)
            data = Data(x=x, y=y, z=z,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        pos=p,
                        edge_gt_att=torch.LongTensor(ground_truth),
                        name=f'SPMotif-{self.mode}-{idx}', idx=idx)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        idx = self.processed_file_names.index('SPMotif_{}.pt'.format(self.mode))
        print(self.processed_paths[idx])
        print(len(data_list))
        torch.save(self.collate(data_list), self.processed_paths[idx])

def augment(data, augmentation = 'random5'):
    if augmentation == 'dnodes':
        data_aug = drop_nodes(deepcopy(data))
    elif augmentation == 'pedges':
        data_aug = permute_edges(deepcopy(data))
    elif augmentation == 'subgraph':
        data_aug = subgraph(deepcopy(data))
    elif augmentation == 'mask_nodes':
        data_aug = mask_nodes(deepcopy(data))
    elif augmentation == 'none':
        """
        if data.edge_index.max() > data.x.size()[0]:
            print(data.edge_index)
            print(data.x.size())
            assert False
        """
        data_aug = deepcopy(data)
        data_aug.x = torch.ones((data.edge_index.max()+1, 1))

    elif augmentation == 'random2':
        n = np.random.randint(2)
        if n == 0:
           data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
           data_aug = subgraph(deepcopy(data))
        else:
            print('sample error')
            assert False

    elif augmentation == 'random3':
        n = np.random.randint(3)
        if n == 0:
           data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
           data_aug = permute_edges(deepcopy(data))
        elif n == 2:
           data_aug = subgraph(deepcopy(data))
        else:
            print('sample error')
            assert False

    elif augmentation == 'random4':
        n = np.random.randint(4)
        if n == 0:
           data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
           data_aug = permute_edges(deepcopy(data))
        elif n == 2:
           data_aug = subgraph(deepcopy(data))
        elif n == 3:
           data_aug = mask_nodes(deepcopy(data))
        else:
            print('sample error')
            assert False
    elif augmentation == 'random5':
        n = np.random.randint(3)
        if n == 0:
           data_aug = drop_nodes(deepcopy(data))
        elif n == 1:
           data_aug = permute_edges(deepcopy(data))
        elif n == 2:
           data_aug = mask_nodes(deepcopy(data))
        else:
            print('sample error')
            assert False

    else:
        print('augmentation error')
        assert False

    # print(data, data_aug)
    # assert False

    return data, data_aug

def drop_nodes(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num / 10)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index

    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def permute_edges(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int(edge_num / 10)

    edge_index = data.edge_index.transpose(0, 1).numpy()

    idx_add = np.random.choice(node_num, (permute_num, 2))
    # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]

    # edge_index = np.concatenate((np.array([edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)]), idx_add), axis=0)
    # edge_index = np.concatenate((edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)], idx_add), axis=0)
    edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
    # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add
    data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data

def subgraph(data):

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * 0.2)

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        if sample_node in idx_sub:
            continue
        idx_sub.append(sample_node)
        idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    idx_nondrop = idx_sub
    idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}

    # data.x = data.x[idx_nondrop]
    edge_index = data.edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    data.edge_index = edge_index



    # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
    # edge_index = [[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
    # data.edge_index = torch.tensor(edge_index).transpose_(0, 1)

    return data


def mask_nodes(data):

    node_num, feat_dim = data.x.size()
    mask_num = int(node_num / 10)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data
