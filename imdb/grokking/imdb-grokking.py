import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter,OrderedDict
import string
import re
import os
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from tunnel_methods import EarlyExit, Rank
import wandb
import warnings
warnings.filterwarnings("ignore")

def rank_analysis(
    model, 
    dataset, 
    layers: list, 
    name=None,
    rpath=None, 
    device="cuda:0",
    eval_hessian=False,
):
    rank = Rank(
        model=model, data=dataset, layers=layers, rpath=rpath, device=device,plotter=lambda x: x + 1
    )
    rank.analysis(eval_hessian=eval_hessian)
    if rpath is not None:
        rank.export(name)
        # rank.plot(name)
    rank.clean_up()
    return rank.result

def early_exits(
    model,
    train_data,
    test_data,
    layers: list,
    name=None,
    rpath=None,
    verbose=False
):
    early_exit = EarlyExit(
        model=model,
        train_data=train_data,  # type: ignore
        test_data=test_data,
        layers=layers,
        rpath=rpath,
        plotter=lambda x: x + 1,
    )

    early_exit.analysis(verbose=verbose)
    # early_exit.analysis()
    if rpath is not None:
        early_exit.export(name=name)
        # early_exit.plot(name=name)
    early_exit.clean_up()
    return early_exit.result

def set_seed(seed=0, dtype=torch.float64):
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def train(train_size, init_scale, wd, lr=0.0003,
          batch_size = 50, seed=0, epochs=5000, test_size=200, num_layers=6, use_wandb=True, rank=True, sharpness=False, data_dir="/scratch/homes/sfan/models/Omnigrok/imdb/grokking/IMDB_Dataset.csv",
          device="cuda:0"):
    set_seed(seed=seed)
    alpha = init_scale

    is_cuda = torch.cuda.is_available()
    if use_wandb:
        wandb.init()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    # if is_cuda:
    #     device = torch.device("cuda")
    #     print("GPU is available")
    # else:
    #     device = torch.device("cpu")
    #     print("GPU not available, CPU used")
    df = pd.read_csv(data_dir)
    
    X,y = df[:train_size+test_size]['review'].values,df[:train_size+test_size]['sentiment'].values
    x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=train_size,stratify=y)
    print(f'#train|test data: {x_train.shape}|{x_test.shape}')

    def preprocess_string(s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        # Replace all runs of whitespaces with no space
        s = re.sub(r"\s+", '', s)
        # replace digits with no space
        s = re.sub(r"\d", '', s)

        return s

    def tockenize(x_train,y_train,x_val,y_val):
        word_list = []

        stop_words = {'a',
                     'about',
                     'above',
                     'after',
                     'again',
                     'against',
                     'ain',
                     'all',
                     'am',
                     'an',
                     'and',
                     'any',
                     'are',
                     'aren',
                     "aren't",
                     'as',
                     'at',
                     'be',
                     'because',
                     'been',
                     'before',
                     'being',
                     'below',
                     'between',
                     'both',
                     'but',
                     'by',
                     'can',
                     'couldn',
                     "couldn't",
                     'd',
                     'did',
                     'didn',
                     "didn't",
                     'do',
                     'does',
                     'doesn',
                     "doesn't",
                     'doing',
                     'don',
                     "don't",
                     'down',
                     'during',
                     'each',
                     'few',
                     'for',
                     'from',
                     'further',
                     'had',
                     'hadn',
                     "hadn't",
                     'has',
                     'hasn',
                     "hasn't",
                     'have',
                     'haven',
                     "haven't",
                     'having',
                     'he',
                     'her',
                     'here',
                     'hers',
                     'herself',
                     'him',
                     'himself',
                     'his',
                     'how',
                     'i',
                     'if',
                     'in',
                     'into',
                     'is',
                     'isn',
                     "isn't",
                     'it',
                     "it's",
                     'its',
                     'itself',
                     'just',
                     'll',
                     'm',
                     'ma',
                     'me',
                     'mightn',
                     "mightn't",
                     'more',
                     'most',
                     'mustn',
                     "mustn't",
                     'my',
                     'myself',
                     'needn',
                     "needn't",
                     'no',
                     'nor',
                     'not',
                     'now',
                     'o',
                     'of',
                     'off',
                     'on',
                     'once',
                     'only',
                     'or',
                     'other',
                     'our',
                     'ours',
                     'ourselves',
                     'out',
                     'over',
                     'own',
                     're',
                     's',
                     'same',
                     'shan',
                     "shan't",
                     'she',
                     "she's",
                     'should',
                     "should've",
                     'shouldn',
                     "shouldn't",
                     'so',
                     'some',
                     'such',
                     't',
                     'than',
                     'that',
                     "that'll",
                     'the',
                     'their',
                     'theirs',
                     'them',
                     'themselves',
                     'then',
                     'there',
                     'these',
                     'they',
                     'this',
                     'those',
                     'through',
                     'to',
                     'too',
                     'under',
                     'until',
                     'up',
                     've',
                     'very',
                     'was',
                     'wasn',
                     "wasn't",
                     'we',
                     'were',
                     'weren',
                     "weren't",
                     'what',
                     'when',
                     'where',
                     'which',
                     'while',
                     'who',
                     'whom',
                     'why',
                     'will',
                     'with',
                     'won',
                     "won't",
                     'wouldn',
                     "wouldn't",
                     'y',
                     'you',
                     "you'd",
                     "you'll",
                     "you're",
                     "you've",
                     'your',
                     'yours',
                     'yourself',
                     'yourselves'}

        for sent in x_train:
            for word in sent.lower().split():
                word = preprocess_string(word)
                if word not in stop_words and word != '':
                    word_list.append(word)

        corpus = Counter(word_list)
        # sorting on the basis of most common words
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
        # creating a dict
        onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

        # tockenize
        final_list_train,final_list_test = [],[]
        for sent in x_train:
                final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                         if preprocess_string(word) in onehot_dict.keys()])
        for sent in x_val:
                final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                        if preprocess_string(word) in onehot_dict.keys()])

        encoded_train = [1 if label =='positive' else 0 for label in y_train]  
        encoded_test = [1 if label =='positive' else 0 for label in y_val] 
        return final_list_train, np.array(encoded_train),final_list_test,np.array(encoded_test),onehot_dict  
        # return np.array(final_list_train), np.array(encoded_train),np.array(final_list_test),np.array(encoded_test),onehot_dict

    x_train,y_train,x_test,y_test,vocab = tockenize(x_train,y_train,x_test,y_test)


    def padding_(sentences, seq_len):
        features = np.zeros((len(sentences), seq_len),dtype=int)
        for ii, review in enumerate(sentences):
            if len(review) != 0:
                features[ii, -len(review):] = np.array(review)[:seq_len]
        return features

    x_train_pad = padding_(x_train,500)
    x_test_pad = padding_(x_test,500)

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
    valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

    # dataloaders
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    # dataiter = iter(train_loader)
    # import pdb
    # pdb.set_trace()
    # sample_x, sample_y = dataiter.__next__()
    
    def L2(model):
        L2_ = 0.
        for p in model.parameters():
            L2_ += torch.sum(p**2)
        return L2_

    def rescale(model, alpha):
        for p in model.parameters():
            p.data = alpha * p.data

    class SentimentRNN(nn.Module):
        def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.0):
            super(SentimentRNN,self).__init__()

            self.output_dim = output_dim
            self.hidden_dim = hidden_dim

            self.no_layers = no_layers
            self.vocab_size = vocab_size

            # embedding and LSTM layers
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

            #lstm
            self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                               num_layers=no_layers, batch_first=True)
            # dropout layer
            self.dropout = nn.Dropout(drop_prob)

            # linear and sigmoid layer
            self.fc = nn.Linear(self.hidden_dim, output_dim)
            self.sig = nn.Sigmoid()

        def forward(self,x,hidden):
            batch_size = x.size(0)
            # embeddings and lstm_out
            embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
            #print(embeds.shape)  #[50, 500, 1000]
            lstm_out, hidden = self.lstm(embeds, hidden)

            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 

            # dropout and fully connected layer
            out = self.dropout(lstm_out)
            out = self.fc(out)

            # sigmoid function
            sig_out = self.sig(out)

            # reshape to be batch_size first
            sig_out = sig_out.view(batch_size, -1)

            sig_out = sig_out[:, -1] # get last batch of labels

            # return last sigmoid output and hidden state
            return sig_out, hidden

        def init_hidden(self, batch_size):
            ''' Initializes hidden state '''
            # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
            # initialized to zero, for hidden state and cell state of LSTM
            h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
            c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
            hidden = (h0,c0)
            return hidden


    no_layers = num_layers
    vocab_size = len(vocab) + 1 #extra 1 for padding
    embedding_dim = 64
    output_dim = 1
    hidden_dim = 256
    model = SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.0)

    #moving to gpu
    model.to(device)
    
    rescale(model, alpha)
    L2_ = L2(model)

    #print(model)

    # loss and optimization functions
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = wd)

    # function to predict accuracy
    def acc(pred,label):
        pred = torch.round(pred.squeeze())
        return torch.sum(pred == label.squeeze()).item()

    clip = 2
    # train for some number of epochs
    train_accs = []
    test_accs = []
    test_losses = []
    train_losses = []
    norms = []
    # last_layer_norms = []
    step = 0
    total_steps = int(epochs*train_size/batch_size)
    with tqdm(total=total_steps) as pbar:
        for epoch in range(epochs):
            train_losses = []
            train_acc = 0.0
            model.train()
            # initialize hidden state 
            h = model.init_hidden(batch_size)
            for inputs, labels in train_loader:
                model.train()
                model.to(device)
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.shape[0] < batch_size:
                    h = model.init_hidden(inputs.shape[0])

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                model.zero_grad()
                output,h = model(inputs,h)

                # calculate the loss and perform backprop
                loss = criterion(output.squeeze().float(), labels.float())
                loss.backward()
                train_losses.append(loss.item())
                # calculating accuracy
                accuracy = acc(output,labels)
                train_acc = accuracy/inputs.shape[0]
                #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                
                val_h = model.init_hidden(batch_size)
                model.eval()
                inputs, labels = next(iter(valid_loader))
                val_h = tuple([each.data for each in val_h])

                inputs, labels = inputs.to(device), labels.to(device)

                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze().float(), labels.float())

                test_losses.append(val_loss.item())

                accuracy = acc(output,labels)
                val_acc = accuracy/batch_size
                test_accs.append(val_acc)
                train_accs.append(train_acc)
                if step % 10 == 0:
                    print(f'step : {step} train_loss : {loss.item()} val_loss : {val_loss.item()}')
                    print(f'train_accuracy : {train_acc} val_accuracy : {val_acc}')
                    pbar.set_description("L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                        train_losses[-1],
                        test_losses[-1],
                        train_accs[-1] * 100, 
                        test_accs[-1] * 100))
                    # wandb logging
                    if use_wandb:
                        wandb_dict = {
                            'train/loss': train_losses[-1],
                            'train/acc': train_accs[-1],
                            'test/loss': test_losses[-1],
                            'test/acc': test_accs[-1],
                            'log_step': step,
                        }
                    with torch.no_grad():
                        total = sum(torch.pow(p, 2).sum() for p in model.parameters())
                        norms.append(float(np.sqrt(total.item())))
                        # last_layer = sum(torch.pow(p, 2).sum() for p in model[-1].parameters())
                        # last_layer_norms.append(float(np.sqrt(last_layer.item())))
                        wandb_dict["norm"] = norms[-1]
                        # wandb_dict["last_layer_norm"] = last_layer_norms[-1]
                    # rank analysis
                    if rank:
                        rank_result = rank_analysis(model=model,
                                                    dataset=valid_data,
                                                    layers=['lstm'],
                                                    name="rank",
                                                    rpath=None,
                                                    device=device,
                                                    eval_hessian=sharpness,)
                        rank_result = rank_result["rank"]
                        if sharpness:
                            top_eigenvalue = rank_result["top_eigenvalues"]
                            trace = rank_result["trace"]
                            import pdb
                            pdb.set_trace()
                        if use_wandb:
                            for l,v in rank_result.items():
                                wandb_dict[f"rank/layer_{l}"] = v
                            if sharpness:
                                wandb_dict[f"sharpness/top_eigenvalue"] = top_eigenvalue
                                wandb_dict[f"sharpness/trace"] = trace
                                
                    if use_wandb:
                        wandb.log(wandb_dict, commit=True) 
                step += 1
                pbar.update(1)

# Add cli params
import argparse
args_parser = argparse.ArgumentParser()
# DomainConfigArguments
args_parser.add_argument('--data_dir', default='/scratch/homes/sfan/models/Omnigrok/imdb/grokking/IMDB_Dataset.csv', type=str)
args_parser.add_argument('--wandb_proj', default='imdb-grok', type=str)
args_parser.add_argument('--wandb_run', default='imdb_rnn_test', type=str)
args_parser.add_argument('--train_points', default=1000, type=int)
args_parser.add_argument('--batch_size', default=100, type=int)
args_parser.add_argument('--total_steps', default=50000, type=int)
args_parser.add_argument('--weight_decay', default=1.0, type=float)
args_parser.add_argument('--lr', default=3e-4, type=float)
args_parser.add_argument('--initialization_scale', default=8.0, type=float)
args_parser.add_argument('--depth', default=4, type=int)
args_parser.add_argument('--seed', default=0, type=int)
args_parser.add_argument('--device', default="cuda", type=str)
args_parser.add_argument('--rank', action='store_true')
args_parser.add_argument('--probe', action='store_true')
args_parser.add_argument('--sharpness', action='store_true')


def run():
    args = args_parser.parse_args()
    os.environ["WANDB_PROJECT"] = args.wandb_proj # name your W&B project 
    wandb.init(project=args.wandb_proj, name=args.wandb_run, config=args)
    epochs = (args.total_steps*args.batch_size//args.train_points)
    train(train_size=args.train_points, init_scale=args.initialization_scale, wd=args.weight_decay, lr=args.lr,
          batch_size=args.batch_size, seed=args.seed, epochs=epochs, test_size=200, num_layers=args.depth, use_wandb=True, rank=True, sharpness=args.sharpness, data_dir=args.data_dir, device=args.device)

if __name__ == "__main__":
    run()
