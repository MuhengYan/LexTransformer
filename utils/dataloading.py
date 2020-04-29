import os
import errno
from config import EMB_DIR

import pickle
import numpy as np
from torch.utils.data import Dataset

#----functions for loading embeddings

def emb_from_cache(emb_name):
    file_path = os.path.join(EMB_DIR, '{}_cached.pkl'.format(emb_name))
    return pickle.load(open(file_path, 'rb'))


def save_emb_cache(emb_name, data):
    file_path = os.path.join(EMB_DIR, '{}_cached.pkl'.format(emb_name))
    pickle.dump(data, open(file_path, 'wb'), protocol=2)
    return

    
def load_embeddings(emb_name):
    try:
        cache = emb_from_cache(emb_name)
        print('Loaded {} from cached file!'.format(emb_name))
        return cache
    except OSError:
        pass
    
    file_path = os.path.join(EMB_DIR, '{}.pkl'.format(emb_name))
    print('Loading {} from raw file!'.format(emb_name))
    
    if os.path.exists(file_path):
        emb_data = pickle.load(open(file_path, 'rb'))
        
        print('Indexing {}...'.format(emb_name))
        word2idx = {}
        idx2word = {}
        embeddings = []
              
        #take idx=0 as zero padding
        dim = len(next(iter(emb_data.values())))
        embeddings.append(np.zeros(dim))
        
        #indexing the word vectors
        for _idx, _word in enumerate(emb_data, 1):
            _vector = emb_data[_word]
            word2idx[_word] = _idx
            idx2word[_idx] = _word
            embeddings.append(_vector)
        
        #add UNK token for out of vocab words
        if '<UNK>' not in word2idx:
            word2idx['<UNK>'] = len(word2idx) + 1
            idx2word[len(word2idx) + 1] = '<UNK>'
            embeddings.append(np.random.uniform(low=-0.05, high=0.05, size=dim))
       
#         assert len(set([len(x) for x in embeddings])) == 1
        print('Indexed {} word vectors.'.format(len(word2idx)))
        embeddings = np.array(embeddings, dtype='float32')
        
        #save cache for the indexing
        save_emb_cache(emb_name, (word2idx, idx2word, embeddings))
        return word2idx, idx2word, embeddings
    else:
        print('{} not found'.format(file_path))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), file)
        
        
#--dataset class
class BaseDataset(Dataset):
    def __init__(self, 
                 X, y, z=None, 
                 length=None, name=None, 
                 verbose=True):
        self.X = X
        self.y = y
        self.z = z
        self.name = name
        
        self.set_length(length)
        
        if verbose:
            self.stats()
    
    
    def set_length(self, length):
        if length is None:
            self.length = max([len(item) for item in self.X])
        else:
            self.length = length
            
    
    def stats(self):
        raise NotImplementedError
            
            
    @staticmethod
    def _load_from_cache():
        return
    
    def _save_cache():
        return
    
    
    
class LocalDataset(BaseDataset):
    
    def __init__(self, 
                 X, y, z, 
                 word2idx,
                 length=None, name=None, 
                 verbose=True):
        self.word2idx = word2idx
        
        BaseDataset.__init__(self, X, y, z, length, name, verbose)
        
    def stats(self):
        vocab = []
        unk = []
        
        for seq in self.X:
            for token in seq:
                vocab.append(token)
                if token not in self.word2idx:
                    unk.append(token)
        
        labels = {}
        for lab in self.y:
            if np.argmax(lab) in labels:
                labels[np.argmax(lab)] += 1
            else:
                labels[np.argmax(lab)] = 1        
        
        print()
        print('Dataset:', self.name)
        print('Total #token:', len(vocab), 'Total #UNK:', len(unk), '{:.2f}%'.format(100 * len(unk)/len(vocab)))
        print('Unique #token:', len(set(vocab)), 'Unique #UNK:', len(set(unk)), '{:.2f}%'.format(100 * len(set(unk))/len(set(vocab))))
        print('Label Distribution:')
        for key in labels:
            print(key, ':' ,labels[key])
        print()
        
    
    
    def __len__(self):
        
        assert len(self.X) == len(self.y) and len(self.X) == len(self.z)
        
        return len(self.X)
    
    
    def __getitem__(self, index):
        X, y, z = self.X[index], self.y[index], self.z[index]
        
        X = self._to_vector(X, self.word2idx, self.length, unk_pol='random')
        z = self._to_vector(z, self.word2idx, self.length, unk_pol='zero')
        
        if isinstance(y, (list, tuple)):
            y = np.array(y)

        return X, y, z, len(self.X[index])
    
    
    @staticmethod
    def _to_vector(sequence, word2idx, length, unk_pol='random'):
        
        seq_vec = np.zeros(length).astype(int)
        
        for i, token in enumerate(sequence[:length]):
            if token in word2idx:
                seq_vec[i] = word2idx[token]
            elif token.lower() in word2idx:
                seq_vec[i] = word2idx[token.lower()]
            else:
                if unk_pol == 'random':
                    seq_vec[i] = word2idx['<UNK>']
                elif unk_pol == 'zero':
                    seq_vec[i] = 0
                else:
                    raise ValueError('UNK policy not recognized!')
        return seq_vec


class GlobalDataset(BaseDataset):

    def __init__(self,
                 X, y, z,
                 word2idx,
                 length=None, length_z=None, name=None,
                 verbose=True):
        self.word2idx = word2idx
        self.length_z = length_z
        BaseDataset.__init__(self, X, y, z, length, name, verbose)

    def stats(self):
        vocab = []
        unk = []

        for seq in self.X:
            for token in seq:
                vocab.append(token)
                if token not in self.word2idx:
                    unk.append(token)

        labels = {}
        for lab in self.y:
            if np.argmax(lab) in labels:
                labels[np.argmax(lab)] += 1
            else:
                labels[np.argmax(lab)] = 1

        print()
        print('Dataset:', self.name)
        print('Total #token:', len(vocab), 'Total #UNK:', len(unk), '{:.2f}%'.format(100 * len(unk) / len(vocab)))
        print('Unique #token:', len(set(vocab)), 'Unique #UNK:', len(set(unk)),
              '{:.2f}%'.format(100 * len(set(unk)) / len(set(vocab))))
        print('Label Distribution:')
        for key in labels:
            print(key, ':', labels[key])
        print()

    def __len__(self):

        assert len(self.X) == len(self.y) and len(self.X) == len(self.z)

        return len(self.X)

    def __getitem__(self, index):
        X, y, z = self.X[index], self.y[index], self.z[index]

        X = self._to_vector(X, self.word2idx, self.length, unk_pol='random')
        z = self._to_vector(z, self.word2idx, self.length_z, unk_pol='zero')

        if isinstance(y, (list, tuple)):
            y = np.array(y)

        return X, y, z, len(self.X[index])

    @staticmethod
    def _to_vector(sequence, word2idx, length, unk_pol='random'):

        seq_vec = np.zeros(length).astype(int)

        for i, token in enumerate(sequence[:length]):
            if token in word2idx:
                seq_vec[i] = word2idx[token]
            elif token.lower() in word2idx:
                seq_vec[i] = word2idx[token.lower()]
            else:
                if unk_pol == 'random':
                    seq_vec[i] = word2idx['<UNK>']
                elif unk_pol == 'zero':
                    seq_vec[i] = 0
                else:
                    raise ValueError('UNK policy not recognized!')
        return seq_vec


class PlainDataset(BaseDataset):
    
    def __init__(self, 
                 X, y,
                 word2idx,
                 length=None, name=None, 
                 verbose=True):
        self.word2idx = word2idx
        
        BaseDataset.__init__(self, X, y, None, length, name, verbose)
        
    def stats(self):
        vocab = []
        unk = []
        
        for seq in self.X:
            for token in seq:
                vocab.append(token)
                if token not in self.word2idx:
                    unk.append(token)
        
        labels = {}
        for lab in self.y:
            if np.argmax(lab) in labels:
                labels[np.argmax(lab)] += 1
            else:
                labels[np.argmax(lab)] = 1        
        
        print()
        print('Dataset:', self.name)
        print('Total #token:', len(vocab), 'Total #UNK:', len(unk), '{:.2f}%'.format(100 * len(unk)/len(vocab)))
        print('Unique #token:', len(set(vocab)), 'Unique #UNK:', len(set(unk)), '{:.2f}%'.format(100 * len(set(unk))/len(set(vocab))))
        print('Label Distribution:')
        for key in labels:
            print(key, ':' ,labels[key])
        print()
        
    
    def __len__(self):
        
        assert len(self.X) == len(self.y)
        
        return len(self.X)
    
    
    def __getitem__(self, index):
        X, y = self.X[index], self.y[index]
        
        X = self._to_vector(X, self.word2idx, self.length, unk_pol='random')
       
        if isinstance(y, (list, tuple)):
            y = np.array(y)
        return X, y, len(self.X[index])
    
    
    @staticmethod
    def _to_vector(sequence, word2idx, length, unk_pol='random'):
        
        seq_vec = np.zeros(length).astype(int)
        
        for i, token in enumerate(sequence[:length]):
            if token in word2idx:
                seq_vec[i] = word2idx[token]
            elif token.lower() in word2idx:
                seq_vec[i] = word2idx[token.lower()]
            else:
                if unk_pol == 'random':
                    seq_vec[i] = word2idx['<UNK>']
                elif unk_pol == 'zero':
                    seq_vec[i] = 0
                else:
                    raise ValueError('UNK policy not recognized!')
        return seq_vec
        