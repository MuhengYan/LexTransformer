import pickle
import torch
import torch.optim


from config import PAD
from hyper_params import args

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.dataloading import load_embeddings, LocalDataset, PlainDataset
from utils.train import TransformerOptimizer, TrainingManager
from utils.train import train_epoch, eval_epoch

# from LexTransformer.Modules import Embed, PosEmbed
# from LexTransformer.Encoders import LexiconTransformerEncoder
from LexTransformer.Models import LexiconTransformerClassifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        
        


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Model is running on:', device, '!')
print()

set_seed(123)

word2idx, idx2word, embeddings = load_embeddings(args['embedding_loc'])
batch_size = args['batch_size']

#----------subject to change------------
dev_dataset = pickle.load(open('datasets/sst-2/train.pkl', 'rb'))
X = dev_dataset['X']
y = dev_dataset['y']
# z = pickle.load(open('datasets/sst_dev/z.pkl', 'rb'))
z = None

# dataset = LocalDataset(X, y, z, word2idx, name='dev_test', length=args['seq_length'])
dataset = PlainDataset(X, y, word2idx, name='dev_train', length=args['seq_length'])

loader_train = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)

dev_dataset = pickle.load(open('datasets/sst-2/dev.pkl', 'rb'))
X = dev_dataset['X']
y = dev_dataset['y']
# z = pickle.load(open('datasets/sst_dev/z.pkl', 'rb'))
z = None

dataset = PlainDataset(X, y, word2idx, name='dev_dev', length=args['seq_length'])
loader_dev = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)

dev_dataset = pickle.load(open('datasets/sst-2/test.pkl', 'rb'))
X = dev_dataset['X']
y = dev_dataset['y']
# z = pickle.load(open('datasets/sst_dev/z.pkl', 'rb'))
z = None

dataset = PlainDataset(X, y, word2idx, name='dev_test', length=args['seq_length'])
loader_test = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)



#--------------------------------------


model = LexiconTransformerClassifier(length=args['seq_length'], 
                                     emb_dim=args['embedding_dim'], 
                                     embeddings=embeddings, 
                                     n_transformer=args['transformer_layers'], 
                                     num_head=args['num_heads'], 
                                     d_k=args['sequence_attn_dim'],
                                     d_linear=args['feed_forward_dim'], 
                                     d_kl=args['lex_attn_dim'], 
                                     alpha=args['alpha'], 
                                     n_logits=args['output_dim'], 
                                     dropout=args['dropout'])

params = filter(lambda p: p.requires_grad, model.parameters())

base_optimizer = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9) #copying transformer settings

optimizer = TransformerOptimizer(optimizer=base_optimizer, 
                                 d_model=args['embedding_dim'],
                                 warmup_steps=args['warmup_steps'])

manager = TrainingManager(tolerance=args['tolerance'])

model = model.to(device)
print(model)
print()

for ep in range(1, args['max_epoch']):
    train_loss, train_acc = train_epoch(model=model,
                                        optimizer=optimizer,
                                        loader=loader_train,
                                        epoch=ep, 
                                        device=device)
    
    dev_loss, dev_acc, _, _ = eval_epoch(model=model, 
                                         loader=loader_dev, 
                                         epoch=ep, 
                                         device=device)
    
    test_loss, test_acc, _, _ = eval_epoch(model=model, 
                                           loader=loader_test, 
                                           epoch=ep, 
                                           device=device)
    print()
    print('----Epoch', ep, 'Summary-----')
    print('Train', 'Loss:', train_loss, 'Acc:', train_acc)
    print('Dev', 'Loss:', dev_loss, 'Acc:', dev_acc)
    print('Test', 'Loss:', test_loss, 'Acc:', test_acc)
    print('------------------------')
    
    passed = manager.checkpoint(model=model, args=args, score=dev_acc)
    print()
    
    if not passed:
        print('early stopping...')
        break
        
