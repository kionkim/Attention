import os
import pandas as pd
import numpy as np
import collections
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, auc
from mxnet import gluon
from tqdm import tqdm


import re, sys, time
import mxnet as mx
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
            self.drop = nn.Dropout(.2)
            self.bi_rnn = rnn.BidirectionalCell(
                 rnn.LSTMCell(hidden_size = self.hidden_dim // 2),  #mx.rnn.LSTMCell doesnot work
                 rnn.LSTMCell(hidden_size = self.hidden_dim // 2)
            )
            self.w_1 = nn.Dense(self.d, use_bias = False)
            self.w_2 = nn.Dense(self.r, use_bias = False)

    def forward(self, x, hidden):
        embeds = self.embed(x) # batch * time step * embedding
        h, _ = self.bi_rnn.unroll(length = embeds.shape[1] \
                                       , inputs = embeds \
                                       , layout = 'NTC' \
                                       , merge_outputs = True)
        # For understanding
        batch_size, time_step, _ = h.shape
        # get self-attention
        _h = h.reshape((-1, self.hidden_dim))
        _w = nd.tanh(self.w_1(_h))
        w = self.w_2(_w)
        _att = w.reshape((-1, time_step, self.r)) # Batch * Timestep * r
        att = nd.softmax(_att, axis = 1)
        x = gemm2(att, h, transpose_a = True)  # h = Batch * Timestep * (2 * hidden_dim), a = Batch * Timestep * r
        return x, att
    
    
class SA_SA_Classifier(nn.Block):
    def __init__(self, sen_rep, classifier, context, **kwargs):
        super(SA_SA_Classifier, self).__init__(**kwargs)
        self.context = context
        with self.name_scope():
            self.sen_rep = sen_rep
            self.classifier = classifier
            
    def forward(self, x):
        # Initial hidden state
        hidden = self.sen_rep.bi_rnn.begin_state()
        lstm_out, att = self.sen_rep(x, hidden)
        x = nd.flatten(lstm_out)
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
                _out, att = model(_data)
                pen = gemm2(att, att, transpose_b = True)
                # Penalty
                tmp = nd.dot(att[0], att[0].T) -nd.array(np.identity(att[0].shape[0]), ctx = context)
                pen = nd.sum(nd.multiply(nd.abs(tmp), nd.abs(tmp)))
                L = L + loss(_out, _label).mean().as_in_context(context) + .5 * pen
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
        test_avg_L, test_acc = evaluate(sa, loss,valid_data, log_interval,  context)
        tqdm.write('[Epoch {}] train avg loss {:.6f}, valid acc {:.2f}, \
            valid avg loss {:.6f}, throughput {:.2f}K wps'.format(
            epoch, epoch_L / epoch_sent_num,
            test_acc, test_avg_L, epoch_wc / 1000 /
            (end_epoch_time - start_epoch_time)))

if __name__ == '__main__':
    from mxnet import gluon, autograd, nd
    from mxnet.gluon import nn, rnn
    from mxnet.ndarray.linalg import gemm2
    import mxnet as mx
    
    max_sen_len = 20
    max_vocab = 10000

    x, y, origin_txt, idx2word, word2idx = prepare_data('../data/umich-sentiment-train.txt', max_sen_len, max_vocab)
    vocab_size = len(idx2word)
    pd.DataFrame(y, columns = ['yn']).reset_index().groupby('yn').count().reset_index()

    ## Data process - tr/va split and define iterator
    tr_idx = np.random.choice(range(len(x)), int(len(x) * .8))
    va_idx = [x for x in range(len(x)) if x not in tr_idx]

    tr_x = [x[i] for i in tr_idx]
    tr_y = [y[i] for i in tr_idx]
    tr_origin = [origin_txt[i] for i in tr_idx]

    va_x = [x[i] for i in va_idx]
    va_y = [y[i] for i in va_idx]
    va_origin = [origin_txt[i] for i in va_idx]

    batch_size = 16
    learning_rate = .0002
    log_interval = 100
    emb_dim = 50 # Emb dim
    hidden_dim = 30 # Hidden dim for LSTM

    train_data = mx.io.NDArrayIter(data=[tr_x, tr_y], batch_size=batch_size, shuffle = False)
    valid_data = mx.io.NDArrayIter(data=[va_x, va_y], batch_size=batch_size, shuffle = False)
    context = mx.cpu()

    #### Sentence Representation
    param = {'emb_dim': emb_dim, 'hidden_dim': hidden_dim, 'vocab_size': vocab_size, 'd': 10, 'r': 5, 'dropout': .2}
    sen_rep = Sentence_Representation(**param)
    sen_rep.collect_params().initialize(mx.init.Xavier(), ctx = context)

    #### Classifier
    classifier = nn.Sequential()
    classifier.add(nn.Dense(16, activation = 'relu'))
    classifier.add(nn.Dense(8, activation = 'relu'))
    classifier.add(nn.Dense(1))
    classifier.collect_params().initialize(mx.init.Xavier(), ctx = context)

    #### Sentiment analysis classifier
    sa = SA_SA_Classifier(sen_rep, classifier, context)
    loss = gluon.loss.SigmoidBCELoss()
    trainer = gluon.Trainer(sa.collect_params(), 'adam', {'learning_rate': 1e-3})

    #### Train
    train(2, train_data,valid_data, sa, trainer, loss, log_interval)
        
    ### Prediction
    result = get_pred(sa, loss, idx2word, valid_data, context)
    ## Classification results
    result[result.pred_sa != result.label].shape

    print('Number of wrong cases = {}'.format(result.shape[0]))
    print('Wrong answers')
    print(result[result.pred_sa != result.label].head(10))