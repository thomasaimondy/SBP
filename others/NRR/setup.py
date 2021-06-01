# -*- coding: utf-8 -*-

"""
------------------------------------------------------------------------------

Copyright (C) 2019 Université catholique de Louvain (UCLouvain), Belgium.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

------------------------------------------------------------------------------

 "setup.py" - Setup configuration and dataset loading.

 Project: DRTP - Direct Random Target Projection

 Authors:  C. Frenkel and M. Lefebvre, Université catholique de Louvain (UCLouvain), 09/2019

 Cite/paper: C. Frenkel, M. Lefebvre and D. Bol, "Learning without feedback: Direct random target projection
    as a feedback-alignment algorithm with layerwise feedforward training," arXiv preprint arXiv:1909.01311, 2019.

------------------------------------------------------------------------------
"""


import torch
import torchvision
from torchvision import transforms,datasets
import sys
import subprocess
import json
#加载tidigits数据集
from python_speech_features import fbank
import numpy as np
import numpy.random as rd
import scipy.io.wavfile as wav
from sklearn.preprocessing import normalize
import os
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import scipy.io as sio

class SynthDataset(torch.utils.data.Dataset):

    def __init__(self, select, type):
        self.dataset, self.input_size, self.input_channels, self.label_features = torch.load( './DATASETS/'+select+'/'+type+'.pt')

    def __len__(self):
        return len(self.dataset[1])

    def __getitem__(self, index):
        return self.dataset[0][index], self.dataset[1][index]

def use_cuda(enabled, device_id=0):
    """Verifies if CUDA is available and sets default device to be device_id."""
    if not enabled:
        return None
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device_id)
    return device_id

def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        #memory_load = get_gpu_memory_usage()
        #cuda_device = np.argmin(memory_load).item()
        #torch.cuda.set_device(cuda_device)
        #device = torch.cuda.current_device()
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #use_cuda(True, device_id=0)
    else:
        device = torch.device('cpu')

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    if args.dataset == "regression_synth":
        print("=== Loading the synthetic regression dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_regression_synth(args, kwargs)
    elif args.dataset == "classification_synth":
        print("=== Loading the synthetic classification dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_classification_synth(args, kwargs)
    elif args.dataset == "MNIST":
        print("=== Loading the MNIST dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_mnist(args, kwargs)
    elif args.dataset == "CIFAR10":
        print("=== Loading the CIFAR-10 dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cifar10(args, kwargs)
    elif args.dataset == "CIFAR10aug":
        print("=== Loading and augmenting the CIFAR-10 dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cifar10_augmented(args, kwargs)
    elif args.dataset == "tidigits":
        print("=== Loading and augmenting the tidifits dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_tidigits(args, kwargs)
    elif args.dataset == "dvsgesture":
        print("=== Loading and augmenting the dvsgesture dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_gesture(args, kwargs)
    elif args.dataset == "timit":
        print("=== Loading and augmenting the TIMIT dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_timit(args, kwargs)
    elif args.dataset == "nettalk":
        print("=== Loading and augmenting the nettalk dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_nettalk(args, kwargs)
    else:
        print("=== ERROR - Unsupported dataset ===")
        sys.exit(1)
    args.regression = (args.dataset == "regression_synth")

    return (device, train_loader, traintest_loader, test_loader)

def get_gpu_memory_usage():
    if sys.platform == "win32":
        curr_dir = os.getcwd()
        nvsmi_dir = r"C:\Program Files\NVIDIA Corporation\NVSMI"
        os.chdir(nvsmi_dir)
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
        os.chdir(curr_dir)
    else:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return gpu_memory

def load_dataset_regression_synth(args, kwargs):

    trainset = SynthDataset("regression","train")
    testset  = SynthDataset("regression", "test")

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.input_size     = trainset.input_size
    args.input_channels = trainset.input_channels
    args.label_features = trainset.label_features

    return (train_loader, traintest_loader, test_loader)

def load_dataset_classification_synth(args, kwargs):

    trainset = SynthDataset("classification","train")
    testset  = SynthDataset("classification", "test")

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.input_size     = trainset.input_size
    args.input_channels = trainset.input_channels
    args.label_features = trainset.label_features

    return (train_loader, traintest_loader, test_loader)

def load_dataset_mnist(args, kwargs):
    rootfile = '.'
    train_loader     = torch.utils.data.DataLoader(datasets.MNIST(rootfile+'/DATASETS/MNIST', train=True,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(datasets.MNIST(rootfile+'/DATASETS/MNIST', train=True,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(datasets.MNIST(rootfile+'/DATASETS/MNIST', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    args.input_size     = 28
    args.input_channels = 1
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def load_dataset_cifar10(args, kwargs):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_cifar10 = transforms.Compose([transforms.ToTensor(),normalize,])

    train_loader     = torch.utils.data.DataLoader(datasets.CIFAR10('/home/zhangdz/jiashuncheng/SBP-test-0126/DATASETS/CIFAR10', train=True,  download=True, transform=transform_cifar10), batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(datasets.CIFAR10('/home/zhangdz/jiashuncheng/SBP-test-0126/DATASETS/CIFAR10', train=True,  download=True, transform=transform_cifar10), batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(datasets.CIFAR10('/home/zhangdz/jiashuncheng/SBP-test-0126/DATASETS/CIFAR10', train=False, download=True, transform=transform_cifar10), batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.input_size     = 32
    args.input_channels = 3
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def load_dataset_cifar10_augmented(args, kwargs):
    #Source: https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                             std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]]),])

    trainset = torchvision.datasets.CIFAR10('./DATASETS/CIFAR10AUG', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    traintestset = torchvision.datasets.CIFAR10('./DATASETS/CIFAR10AUG', train=True, download=True, transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=args.test_batch_size, shuffle=False)

    testset = torchvision.datasets.CIFAR10('./DATASETS/CIFAR10AUG', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

    args.input_size     = 32
    args.input_channels = 3
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def read_data(path, n_bands, n_frames):
    overlap = 0.5

    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.waV') and file[0] != 'O':
                filelist.append(os.path.join(root, file))
    # filelist = filelist[:1002]

    n_samples = len(filelist)

    def keyfunc(x):
        s = x.split('/')
        return (s[-1][0], s[-2], s[-1][1]) # BH/1A_endpt.wav: sort by '1', 'BH', 'A'
    filelist.sort(key=keyfunc)

    feats = np.empty((n_samples, 1, n_bands, n_frames))
    labels = np.empty((n_samples,), dtype=np.long)
    with tqdm(total=len(filelist)) as pbar:
        for i, file in enumerate(filelist):
            pbar.update(1)
            label = file.split('/')[-1][0]  # if using windows, change / into \\
            if label == 'Z':
                labels[i] = np.long(0)
            else:
                labels[i] = np.long(label)
            rate, sig = wav.read(file)
            duration = sig.size / rate
            winlen = duration / (n_frames * (1 - overlap) + overlap)
            winstep = winlen * (1 - overlap)
            feat, energy = fbank(sig, rate, winlen, winstep, nfilt=n_bands, nfft=4096, winfunc=np.hamming)
            # feat = np.log(feat)
            final_feat = feat[:n_frames]
            final_feat = normalize(final_feat, norm='l1', axis=0)
            feats[i] = np.expand_dims(np.array(final_feat),axis=0)
        # feats[i] = feat[:n_frames].flatten() # feat may have 41 or 42 frames
        # feats[i] = feat.flatten() # feat may have 41 or 42 frames

    # feats = normalize(feats, norm='l2', axis=1)
    # normalization
    # feats = preprocessing.scale(feats)

    np.random.seed(42)
    p = np.random.permutation(n_samples)
    feats, labels = feats[p], labels[p]

    n_train_samples = int(n_samples * 0.7)
    print('n_train_samples:',n_train_samples)

    train_set = (feats[:n_train_samples], labels[:n_train_samples])
    test_set = (feats[n_train_samples:], labels[n_train_samples:])

    return train_set, train_set, test_set

def datatobatch(args,train_loader):
    temp, temp2 = [], []
    label, label2 = [], []
    for i, data in enumerate(train_loader[0]):
        if i % args.batch_size == 0 and i != 0:
            temp2.append(temp)
            label2.append(label)
            temp, label = [], []
            temp.append(data)
            label.append(train_loader[1][i])
        else:
            temp.append(data)
            label.append(train_loader[1][i])
    temp2 = torch.tensor(temp2)
    label2 = torch.tensor(label2)
    a = (temp2, label2)
    return a

class Tidigits(Dataset):
    def __init__(self,train_or_test,input_channel,n_bands,n_frames,transform=None, target_transform = None):
        super(Tidigits, self).__init__()
        self.n_bands = n_bands
        self.n_frames = n_frames
        rootfile = '.'
        dataname = rootfile + '/DATASETS/tidigits/packed_tidigits_nbands_'+str(n_bands)+'_nframes_' + str(n_frames)+'.pkl'
        if os.path.exists(dataname):
            with open(dataname,'rb') as fr:
                [train_set, val_set, test_set] = pickle.load(fr)
        else:
            print('Tidigits Dataset Has not been Processed, now do it.')
            train_set, val_set, test_set = read_data(path=rootfile+'/DATASETS/tidigits/isolated_digits_tidigits', n_bands=n_bands, n_frames=n_frames)#(2900, 1640) (2900,)
            with open(dataname,'wb') as fw:
                pickle.dump([train_set, val_set, test_set],fw)
        if train_or_test == 'train':
            self.x_values = train_set[0]
            self.y_values = train_set[1]

        elif train_or_test == 'test':
            self.x_values = test_set[0]
            self.y_values = test_set[1]
        elif train_or_test == 'valid':
            self.x_values = val_set[0]
            self.y_values = val_set[1]
        self.transform =transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        sample = self.x_values[index]
        label = self.y_values[index]
        return sample, label
    
    def __len__(self):
        return len(self.x_values)

class Gesture(Dataset):
    def __init__(self, train_or_test, transform = None, target_transform = None):
        super(Gesture, self).__init__()
        rootfile = '/mnt/lustre/xushuang4/jiashuncheng/code/sbp-snn-master/compare_with_others/BRP'
        # rootfile = '.'
        mat_fname = rootfile+'/DATASETS/DVS_gesture_100.mat'
        mat_contents = sio.loadmat(mat_fname)
        if train_or_test == 'train':
            self.x_values = mat_contents['train_x_100']
            self.y_values = mat_contents['train_y']
        elif train_or_test == 'test':
            self.x_values = mat_contents['test_x_100']
            self.y_values = mat_contents['test_y']
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index, :, :]
        sample = torch.reshape(torch.tensor(sample), (sample.shape[0], 32, 32)).unsqueeze(0)
        label = self.y_values[index].astype(np.float32)
        label = torch.topk(torch.tensor(label), 1)[1].squeeze(0)
        return sample, label

    def __len__(self):
        return len(self.x_values)

class Nettalk(Dataset):
    def __init__(self, train_or_test, transform = None, target_transform = None):
        super(Nettalk, self).__init__()
        mat_fname = 'DATASETS/nettalk_small.mat'
        mat_contents = sio.loadmat(mat_fname)
        if train_or_test == 'train':
            self.x_values = mat_contents['train_x']
            self.y_values = mat_contents['train_y']
        elif train_or_test == 'test':
            self.x_values = mat_contents['test_x']
            self.y_values = mat_contents['test_y']
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index]
        label = self.y_values[index]
        return sample, label

    def __len__(self):
        return len(self.x_values)

#以下是TIMIT数据集的读取程序
def pad_vector(v, n_time, pad_value=0.):
    if len(v.shape) == 2:
        shp = v.shape
        return np.concatenate([v, pad_value * np.ones((n_time - shp[0], shp[1]))], axis=0)
    elif len(v.shape) == 1:
        shp = v.shape

    return np.concatenate([v, pad_value * np.zeros((n_time - shp[0],))], axis=0)

def sparsity_dense_vector(vector, blank_symbol):
    indices = []
    values = []
    d_vector = np.diff(vector)
    change_indices = np.where(d_vector != 0)[0]
    last_value = blank_symbol
    for ind in change_indices:
        value = vector[ind]
        indices.append(ind)
        values.append(value)
        # last_value=value

    # last_v = blank_symbol
    # for v in values:
    #     assert v != blank_symbol, 'v: {}'.format(blank_symbol)
    #     assert v != last_v, 'v {} last_v {} '.format(v,last_v)
    #     last_v = v

    return np.array(indices, dtype=np.int), np.array(values, dtype=np.int)

def label_stack_to_sparse_tensor(label_stack, blank_symbol):
    sparse_tensor = {'indices': [], 'values': []}

    for i_batch, phns in enumerate(label_stack):
        indices, values = sparsity_dense_vector(phns, blank_symbol)

        sparse_tensor['indices'].append([[i_batch, i_time] for i_time in indices])
        sparse_tensor['values'].append(values)

    sparse_tensor['indices'] = np.concatenate(sparse_tensor['indices'])
    sparse_tensor['values'] = np.concatenate(sparse_tensor['values'])

    return sparse_tensor

class Timit(Dataset):
    def __init__(self, phase, data_path = './DATASETS/timit_processed', preproc = None, use_reduced_phonem_set = True, return_sparse_phonem_tensor = False, epsilon = 1e-10):
        super(Timit, self).__init__()
        assert preproc is not None
        self.phase = phase
        self.data_path = data_path
        self.preproc = preproc
        self.use_reduced_phonem_set = use_reduced_phonem_set
        self.return_sparse_phn_tensor = return_sparse_phonem_tensor

        self.n_time = 780

        self.epsilon = epsilon

        self.n_feats = {'fbank': 41 * 3, 'mfccs': 13 * 3, 'htk': 13 * 3 if 'htk_mfcc' in data_path else 41 * 3, 'cochspec': 86 * 3, 'cochspike': 86} # 'mfccspike': 13 * 31, 'melspec': 16 * 3, 'melspike': 496
        self.n_features = self.n_feats[preproc]
        self.n_phns = 39 if use_reduced_phonem_set else 61

        # Load features from files
        self.feature_stack_train, self.phonem_stack_train, self.meta_data_train, _, _ = self.load_data_stack('train')
        self.feature_stack_test, self.phonem_stack_test, self.meta_data_test, self.vocabulary, self.wav_test = self.load_data_stack('test')
        self.feature_stack_develop, self.phonem_stack_develop, self.meta_data_develop, self.vocabulary, self.wav_val = self.load_data_stack('develop')

        def add_derivatives(features):
            n_features = features[0].shape[1]

            # add derivatives:
            get_delta = lambda v : np.concatenate([np.zeros((1, v.shape[1])), v[2:] - v[:-2], np.zeros((1, v.shape[1]))],axis = 0)
            d_features = [get_delta(f) for f in features]
            d2_features = [get_delta(f) for f in d_features]

            features = [np.concatenate([f, d_f, d2_f], axis=1) for f,d_f,d2_f in zip(features,d_features,d2_features)]
            assert (features[0].shape[1] == self.n_features)
            return features

        if self.preproc not in ['cochspike', 'htk']:
            self.feature_stack_train = add_derivatives(self.feature_stack_train)
            self.feature_stack_test = add_derivatives(self.feature_stack_test)
            self.feature_stack_develop = add_derivatives(self.feature_stack_develop)

        # normalize the features
        concatenated_training_features = np.concatenate(self.feature_stack_train, axis = 0)
        means = np.mean(concatenated_training_features, axis = 0)
        stds = np.std(concatenated_training_features, axis = 0)

        if self.preproc != 'cochspike':
            self.feature_stack_train = [(f - means) / np.maximum(stds,self.epsilon) for f in self.feature_stack_train]
            self.feature_stack_test = [(f - means) / np.maximum(stds,self.epsilon) for f in self.feature_stack_test]
            self.feature_stack_develop = [(f - means) / np.maximum(stds,self.epsilon) for f in self.feature_stack_develop]

        self.feature_stack_train = np.array(self.feature_stack_train,dtype=object)
        self.feature_stack_test = np.array(self.feature_stack_test,dtype=object)
        self.feature_stack_develop = np.array(self.feature_stack_develop,dtype=object)

        assert (len(self.vocabulary) == self.n_phns)

        self.n_train = len(self.feature_stack_train)
        self.n_test = len(self.feature_stack_test)
        self.n_develop = len(self.feature_stack_develop)

        # print('Dataset sizes: test {} \t train {} \t validation {}'.format(self.n_test,self.n_train,self.n_develop))

    def __len__(self):
        if self.phase == 'train':
            return self.n_train
        if self.phase == 'develop':
            return self.n_develop
        if self.phase == 'test':
            return self.n_test

    def reduce_phonem_list(self, phn_list):
        return [self.phonem_reduction_map[k] for k in phn_list]

    def load_data_stack(self, dataset):
        path = os.path.join(self.data_path, dataset)

        # Define the link to the pickle objects
        if self.preproc == 'fbank':
            feature_path = os.path.join(path, 'filter_banks.pickle')
        elif self.preproc == 'mfccs':
            feature_path = os.path.join(path, 'mfccs.pickle')
        elif self.preproc == 'htk':
            feature_path = os.path.join(path, 'htk.pickle')
        # elif self.preproc == 'mfccspike':
        #     feature_path = os.path.join(path, 'mfcc_spike_stack.pickle')
        # elif self.preproc == 'melspec':
        #     feature_path = os.path.join(path, 'specgram.pickle')
        # elif self.preproc == 'melspike':
        #     feature_path = os.path.join(path, 'spike.pickle')
        elif self.preproc == 'cochspec':
            feature_path = os.path.join(path, 'coch_raw.pickle')
        elif self.preproc == 'cochspike':
            feature_path = os.path.join(path, 'coch_spike.pickle')
        else:
            raise NotImplementedError('Preprocessing %s not available' % self.preproc)

        if self.use_reduced_phonem_set:
            phonem_path = os.path.join(path, 'reduced_phonems.pickle')
            vocabulary_path = os.path.join(path, 'reduced_phonem_list.json')
        else:
            phonem_path = os.path.join(path, 'phonems.pickle')
            vocabulary_path = os.path.join(path, 'phonem_list.json')

        # Load the data
        with open(feature_path, 'rb') as f:
            data_stack = pickle.load(f)

        with open(phonem_path, 'rb') as f:
            phonem_stack = pickle.load(f)

            for phns in phonem_stack:
                assert ((np.array(phns) < self.n_phns).all()), 'Found phonems up to {} should be maximum {}'.format(
                    np.max(phns), self.n_phns)

        # Load the vocabulay
        with open(vocabulary_path, 'r') as f:
            vocabulary = json.load(f)

        assert vocabulary[0] == ('sil' if self.use_reduced_phonem_set else 'h#')
        self.silence_symbol_id = 0

        # Load meta data
        with open(os.path.join(path, 'metadata.pickle'), 'rb') as f:
            metadata = pickle.load(f)

        assert vocabulary[0] == ('sil' if self.use_reduced_phonem_set else 'h#')
        self.silence_symbol_id = 0

        with open(os.path.join(path, 'reduced_phn_index_mapping.json'), 'r') as f:
            self.phonem_reduction_map = json.load(f)

        # Load raw audio
        wav_path = os.path.join(path, 'wav.pickle')
        with open(wav_path, 'rb') as f:
            wav_stack = pickle.load(f)

        return data_stack, phonem_stack, metadata, vocabulary, wav_stack

    def __getitem__(self, idx):
        if self.phase == 'train':
            feature_stack = self.feature_stack_train[idx]
            phonem_stack = self.phonem_stack_train[idx]
            wavs = None
        elif self.phase == 'test':
            feature_stack = self.feature_stack_test[idx]
            phonem_stack = self.phonem_stack_test[idx]
            wavs = self.wav_test[idx]
        elif self.phase == 'develop':
            feature_stack = self.feature_stack_develop[idx]
            phonem_stack = self.phonem_stack_develop[idx]
            wavs = self.wav_val[idx]

        seq_len = feature_stack.shape[0]
        #feature = feature_stack
        feature = pad_vector(feature_stack, self.n_time)

        if self.return_sparse_phn_tensor:
            phns = label_stack_to_sparse_tensor(phonem_stack, self.silence_symbol_id)
        else:
            phns = pad_vector(phonem_stack, self.n_time, self.silence_symbol_id)
            #phns = phonem_stack
            
        return torch.FloatTensor(feature), torch.LongTensor(phns), seq_len
#以上是TIMIT数据集的读取程序

def load_dataset_tidigits(args, kwargs):
    
    n_bands = 39
    n_frames = 39
    args.input_size = n_bands
    args.input_channels = 1
    args.label_features = 10

    train_dataset = Tidigits('train',args.input_channels, n_bands,n_frames,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))
    traintest_dataset = Tidigits('valid',args.input_channels,n_bands,n_frames,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))
    test_dataset = Tidigits('test',args.input_channels,n_bands,n_frames,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,shuffle = True, drop_last = True)
    traintest_loader = torch.utils.data.DataLoader(dataset=traintest_dataset, batch_size=args.test_batch_size,shuffle = False,drop_last = True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size,shuffle = False,drop_last = True)

    

    return (train_loader, traintest_loader, test_loader)


def load_dataset_gesture(args, kwargs):

    args.input_size     = 32*32
    args.input_channels = 1
    args.label_features = 11

    train_dataset = Gesture('train', transform=transforms.ToTensor())
    test_dataset = Gesture('test', transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = args.batch_size,shuffle = True,drop_last=True)
    traintest_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = args.batch_size,shuffle = False,drop_last=True)

    return (train_loader, traintest_loader, test_loader)


def load_dataset_timit(args, kwargs):
    args.input_size = 39
    args.input_channels = 1
    args.label_features = 39

    train_dataset = Timit('train', preproc='mfccs')
    test_dataset = Timit('test', preproc='mfccs')

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    traintest_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)


    return (train_loader, traintest_loader, test_loader)

def load_dataset_nettalk(args, kwargs):
    args.input_size = 189
    args.input_channels = 1
    args.label_features = 26

    train_dataset = Nettalk('train', transform=transforms.ToTensor())
    test_dataset = Nettalk('test', transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True,drop_last=True)
    traintest_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False,drop_last=True)

    return (train_loader, traintest_loader, test_loader)

