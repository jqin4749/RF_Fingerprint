import numpy as np
import pickle
import os
from definitions import *

def load_from_disk(dataset_index):
    root_dir = './'
    if dataset_index == WIFI_PREAM:
        dataset_name = 'grid_2019_12_25.pkl'
    elif dataset_index == WIFI2_PREAM:
        dataset_name = 'grid_2020_02_03.pkl'
    elif dataset_index == WIFI3_PREAM:
        dataset_name = 'grid_2020_02_04.pkl'
    elif dataset_index == WIFI4_PREAM:
        dataset_name = 'grid_2020_02_05.pkl'
    elif dataset_index == WIFI5_PREAM:
        dataset_name = 'grid_2020_02_06.pkl'
        
    else:
        raise ValueError('Wrong dataset index')
    dataset_path = root_dir + dataset_name
    with open(dataset_path,'rb') as f:
        dataset = pickle.load(f)
    return dataset

def fetch_mdatasets():
    dataset1 = load_from_disk(WIFI_PREAM)
    dataset1['Index'] = WIFI_PREAM
    dataset2 = load_from_disk(WIFI2_PREAM)
    dataset2['Index'] = WIFI2_PREAM
    dataset3 = load_from_disk(WIFI3_PREAM)
    dataset3['Index'] = WIFI3_PREAM
    dataset4 = load_from_disk(WIFI4_PREAM)
    dataset4['Index'] = WIFI4_PREAM
    dataset5 = load_from_disk(WIFI5_PREAM)
    dataset5['Index'] = WIFI5_PREAM
    dataset =  merge_datasets([dataset1,dataset2,dataset3,dataset4])
    test_dataset=dataset5
    return (dataset,test_dataset)


# Merge multiple datasets
def merge_datasets(datasets):
    # Merge node lists
    full_list = []
    for d in datasets:
        full_list = full_list + d['node_list']
    full_list = sorted(list(set(full_list)))
    
    # Initialize destination dataset 
    full_dataset = {}
    full_dataset['node_list'] = full_list
    full_dataset['data'] = [[] for i in range(len(full_list))]
    full_dataset['data'] 
    # Place data in lists for each node
    for d in datasets:
        for src_i,src_n in enumerate(d['node_list']):
            dest_i = full_dataset['node_list'].index(src_n)
            full_dataset['data'][dest_i].append(d['data'][src_i])
    
    # Merge lists for each node and shuffle
    for i,d in enumerate(full_dataset['data']):
        if len(d)==1:
            dd = d[0]
        else:
            dd = np.concatenate(d)
        full_dataset['data'][i] = dd
        np.random.shuffle(full_dataset['data'][i])
    return full_dataset

# normalize data
def norm(sig_u):
    if len(sig_u.shape)==3:
        pwr = np.sqrt(np.mean(np.sum(sig_u**2,axis = -1),axis = -1))
        sig_u = sig_u/pwr[:,None,None]
    if len(sig_u.shape)==2:
        sig_u = (sig_u - np.min(sig_u))/(np.max(sig_u) - np.min(sig_u)) * 2 - 1
    # print(sig_u.shape)
    return sig_u

class unit():
    def __init__(self, data, name, id, cat = 'X') -> None:
        self.name_ = name # str
        self.id_ = id # int
        self.cat_ = cat
        self.data_ = np.concatenate([np.expand_dims(norm(d),axis=0) for d in data],axis=0)
        return 

class dataset():
    data_units_ = []
    data_units_diff_day_ = []
    O_size_ = 63 # this is fixed according to the paper

    def __init__(self, A_size=10, K_size=10) -> None:
        (self.dataset_, self.test_dataset_) = fetch_mdatasets()
        self.update_AK(A_size, K_size)
        self.gen_data_units()
        self.gen_data_units_diff_day()
        return

    def update_AK(self, A_size, K_size):
        assert A_size+K_size+self.O_size_ <= len(self.dataset_['data']), '[update_AK] too large'
        self.A_size_ = A_size
        self.K_size_ = K_size
        return 

    def gen_data_units(self):
        data_units = [unit(d,n,-1,'X') for d,n in zip(self.dataset_['data'], self.dataset_['node_list'])]
        A_idx = list(np.random.choice(len(self.dataset_['data']), self.A_size_, replace=False))
        K_idx = list(np.random.choice([i for i in range(len(self.dataset_['data'])) if i not in A_idx], self.K_size_, replace=False))
        O_idx = list(np.random.choice([i for i in range(len(self.dataset_['data'])) if i not in A_idx+K_idx], self.O_size_, replace=False)) 
        for idx, i in enumerate(A_idx):
            data_units[i].id_ = idx
            data_units[i].cat_ = 'A'
        for i in K_idx:
            data_units[i].id_ = self.A_size_
            data_units[i].cat_ = 'K'
        for i in O_idx:
            data_units[i].id_ = self.A_size_
            data_units[i].cat_ = 'O'
        self.data_units_ = data_units
        return 

    def gen_data_units_diff_day(self):
        A_names = [d.name_ for d in self.get_set_A()]
        O_names = [d.name_ for d in self.get_set_O()]
        A_ids = [d.id_ for d in self.get_set_A()]
        data_units = []
        for d, n in zip(self.test_dataset_['data'], self.test_dataset_['node_list']):
            if n in A_names:
                data_units.append(unit(d,n,A_ids[A_names.index(n)],'A'))
            if n in O_names:
                data_units.append(unit(d,n,self.A_size_, 'O'))
        self.data_units_diff_day_ =  data_units       
        return 

    def shuffle(self):
        self.gen_data_units()
        self.gen_data_units_diff_day()

    def get_set_A(self):
        return [d for d in self.data_units_ if d.cat_ == 'A']
    def get_set_K(self):
        return [d for d in self.data_units_ if d.cat_ == 'K']
    def get_set_O(self):
        return [d for d in self.data_units_ if d.cat_ == 'O']    

    def get_train_test_set(self):
        A_set = self.get_set_A()
        K_set = self.get_set_K()
        O_set = self.get_set_O()

        train = []
        test = []
        for u in A_set:
            train_idx = np.random.choice(len(u.data_), int(len(u.data_)*0.7), replace=False)
            test_idx = [i for i in range(len(u.data_)) if i not in train_idx]
            train.append((u.data_[train_idx], u.id_))
            test.append((u.data_[test_idx], u.id_))
            
        train += [(u.data_, u.id_) for u in K_set]
        test += [(u.data_, u.id_) for u in O_set]
        
        train_x = []
        train_y = []
        for i in train:
            for x in i[0]:
                train_x.append(np.expand_dims(x,axis=0))
                train_y.append(np.expand_dims(i[1],axis=0))
        test_x = []
        test_y = []
        for i in test:
            for x in i[0]:
                test_x.append(np.expand_dims(x,axis=0))
                test_y.append(np.expand_dims(i[1],axis=0))
        
        train_x = np.concatenate(train_x,axis=0)
        train_y = np.concatenate(train_y,axis=0).reshape(-1,1)
        test_x = np.concatenate(test_x,axis=0)
        test_y = np.concatenate(test_y,axis=0).reshape(-1,1)
        return {'train_x': train_x, 'train_y': train_y, 
                'test_x': test_x, 'test_y': test_y}
    
    def get_diff_day_test_set(self):
        A_set = [d for d in self.data_units_diff_day_ if d.cat_ == 'A']
        O_set = [d for d in self.data_units_diff_day_ if d.cat_ == 'O']   

        test_x = []
        test_y = []
        for i in A_set + O_set:
            for x in i.data_:
                test_x.append(np.expand_dims(x,axis=0))
                test_y.append(np.expand_dims(i.id_,axis=0))
        test_x = np.concatenate(test_x,axis=0)
        test_y = np.concatenate(test_y,axis=0).reshape(-1,1)

        return {'test_x': test_x, 'test_y': test_y}