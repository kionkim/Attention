import collections, os, re, sys, time
import pandas as pd
import mxnet as mx
import numpy as np

from mxnet import gluon, autograd, nd
from mxnet.gluon import nn, rnn
from mxnet.ndarray.linalg import gemm2
from tqdm import tqdm
import spacy
from utils import *

os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'


class Sentence_Representation(nn.Block):
    def __init__(self, **kwargs):
        super(Sentence_Representation, self).__init__()
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        
        with self.name_scope():
            self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
            self.g_fc1 = nn.Dense(self.hidden_dim,activation='relu')
            self.g_fc2 = nn.Dense(self.hidden_dim,activation='relu')
            self.attn = nn.Dense(1, activation = 'tanh')
            
    def forward(self, x):
        embeds = self.embed(x) # batch * time step * embedding
        x_i = embeds.expand_dims(1)
        x_i = nd.repeat(x_i,repeats= self.sentence_length, axis=1) # batch * time step * time step * embedding
        x_j = embeds.expand_dims(2)
        x_j = nd.repeat(x_j,repeats= self.sentence_length, axis=2) # batch * time step * time step * embedding
        x_full = nd.concat(x_i,x_j,dim=3) # batch * time step * time step * (2 * embedding)
        # New input data
        _x = x_full.reshape((-1, 2 * self.emb_dim))
        
        # Network for attention
        _attn = self.attn(_x)
        _att = _attn.reshape((-1, self.sentence_length, self.sentence_length))
        _att = nd.sigmoid(_att)
        att = nd.softmax(_att, axis = 1)
        
        _x = self.g_fc1(_x) # (batch * time step * time step) * hidden_dim
        _x = self.g_fc2(_x) # (batch * time step * time step) * hidden_dim
        # sentence_length*sentence_length개의 결과값을 모두 합해서 sentence representation으로 나타냄

        x_g = _x.reshape((-1, self.sentence_length, self.sentence_length, self.hidden_dim))
    
        _inflated_att = _att.expand_dims(axis = -1)
        _inflated_att = nd.repeat(_inflated_att, repeats = self.hidden_dim, axis = 3)

        x_q = nd.multiply(_inflated_att, x_g)

        sentence_rep = nd.mean(x_q.reshape(shape = (-1, self.sentence_length **2, self.hidden_dim)), axis= 1)
        return sentence_rep, att
    

class SA_Classifier(nn.Block):
    def __init__(self, sen_rep, classifier, context, **kwargs):
        super(SA_Classifier, self).__init__(**kwargs)
        self.context = context
        with self.name_scope():
            self.sen_rep = sen_rep
            self.classifier = classifier
            
    def forward(self, x):
        # Initial hidden state
        # sentence representation할 때 hidden의 context가 cpu여서 오류 발생. context를 gpu로 전환
        x, att = self.sen_rep(x)
        x = nd.flatten(x)
        res = self.classifier(x)
        return res, att       
    
    

def train(n_epoch, train_data,valid_data, model, trainer, loss, log_interval):
    for epoch in tqdm(range(n_epoch), desc = 'epoch'):
        ## Training
        train_data.reset()
        # Epoch training stats
        start_epoch_time = time.time()
        epoch_L = 0.0
        epoch_sent_num = 0
        epoch_wc = 0
        # Log interval training stats
        start_log_interval_time = time.time()
        log_interval_wc = 0
        log_interval_sent_num = 0
        log_interval_L = 0.0

        for i, batch in enumerate(train_data):
            _data = batch.data[0].as_in_context(context)
            _label = batch.data[1].as_in_context(context)
            L = 0
            wc = len(_data)
            log_interval_wc += wc
            epoch_wc += wc
            log_interval_sent_num += _data.shape[1]
            epoch_sent_num += _data.shape[1]
            with autograd.record():
                _out, _ = sa(_data)
                L = L + loss(_out, _label).mean().as_in_context(context)
            L.backward()
            trainer.step(_data.shape[0])
            log_interval_L += L.asscalar()
            epoch_L += L.asscalar()
            if (i + 1) % log_interval == 0:
                tqdm.write('[Epoch {} Batch {}/{}] elapsed {:.2f} s, \
                        avg loss {:.6f}, throughput {:.2f}K wps'.format(
                        epoch, i + 1, train_data.num_data//train_data.batch_size,
                        time.time() - start_log_interval_time,
                        log_interval_L / log_interval_sent_num,
                        log_interval_wc / 1000 / (time.time() - start_log_interval_time)))
                # Clear log interval training stats
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_L = 0
        end_epoch_time = time.time()
        test_avg_L, test_acc = evaluate(sa, loss, valid_data, log_interval, context)
        tqdm.write('[Epoch {}] train avg loss {:.6f}, valid acc {:.2f}, \
            valid avg loss {:.6f}, throughput {:.2f}K wps'.format(
            epoch, epoch_L / epoch_sent_num,
            test_acc, test_avg_L, epoch_wc / 1000 /
            (end_epoch_time - start_epoch_time)))


def plot_attention(net, n_samples = 10, mean = False):
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    idx = np.random.choice(np.arange(len(va_x)), size = n_samples, replace = False)
    _dat = [va_x[i] for i in idx]
    
    w_idx = []
    word = [[idx2word[x] for x in y] for y in _dat]
    original_txt = [va_origin[i] for i in idx]
    out, att = net(nd.array(_dat, ctx = context)) 

    _a = []
    _w = []
    for x, y, z in zip(word, att, original_txt):
        _idx = [i for i, _x in enumerate(x) if _x is not 'PAD']
        _w.append(np.array([x[i] for i in _idx]))
        _y = y[:, _idx]
        _a.append(np.array([_y[i].asnumpy() for i in _idx]))
        
    _label = [va_y[i] for i in idx]
    _pred = (nd.sigmoid(out) > .5).asnumpy()
    
    fig, axes = plt.subplots(np.int(np.ceil(n_samples / 3)), 3, sharex = False, sharey = False)
    plt.subplots_adjust(hspace=1)

    fig.set_size_inches(20, 20)
    plt.subplots_adjust(hspace=1)
    
    
if __name__ == '__main__':
    
    max_sen_len = 20
    max_vocab = 10000
    batch_size = 16
    learning_rate = .0002
    log_interval = 100
    emb_dim = 10 # Emb dim
    hidden_dim = 30 # Hidden dim for LSTM
    
    x, y, origin_txt, idx2word, word2idx = prepare_data('../data/umich-sentiment-train.txt', max_sen_len, max_vocab)
    vocab_size = len(idx2word)

    ## Data process - tr/va split and define iterator

    tr_idx = np.random.choice(range(len(x)), int(len(x) * .8))
    va_idx = [x for x in range(len(x)) if x not in tr_idx]

    tr_x = [x[i] for i in tr_idx]
    tr_y = [y[i] for i in tr_idx]
    tr_origin = [origin_txt[i] for i in tr_idx]

    va_x = [x[i] for i in va_idx]
    va_y = [y[i] for i in va_idx]
    va_origin = [origin_txt[i] for i in va_idx]

    train_data = mx.io.NDArrayIter(data=[tr_x, tr_y], batch_size=batch_size, shuffle = False, last_batch_handle = 'discard')
    valid_data = mx.io.NDArrayIter(data=[va_x, va_y], batch_size=batch_size, shuffle = False, last_batch_handle = 'discard')


    context = mx.cpu()

    #### Sentence Representation
    param = {'emb_dim': emb_dim, 'hidden_dim': hidden_dim, 'vocab_size': vocab_size, 'sentence_length': max_sen_len, 'dropout': .2}
    sen_rep = Sentence_Representation(**param)
    sen_rep.collect_params().initialize(mx.init.Xavier(), ctx = context)


    #### Classifier
    classifier = nn.Sequential()
    classifier.add(nn.Dense(16, activation = 'relu'))
    classifier.add(nn.Dense(8, activation = 'relu'))
    classifier.add(nn.Dense(1))
    classifier.collect_params().initialize(mx.init.Xavier(), ctx = context)

    #### Sentiment analysis classifier
    sa = SA_Classifier(sen_rep, classifier, context)
    loss = gluon.loss.SigmoidBCELoss()
    trainer = gluon.Trainer(sa.collect_params(), 'adam', {'learning_rate': 1e-3})
    sa.hybridize()

    ### Train
    train(2, train_data,valid_data, sa, trainer, loss, log_interval)

    ### Prediction
    result = get_pred(sa, loss, idx2word, valid_data, context)
    
    ## Classification results
    print('Number of wrong classification = {} out of {}'.format(result[result.pred_sa != result.label].shape[0], result.shape[0]))
    print('Wrongly classified exmples = {}'.format(result[result.pred_sa != result.label].head(10)))