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
            
    def forward(self, x):
        embeds = self.embed(x) # batch * time step * embedding
        x_i = embeds.expand_dims(1) # 64 * 1* 40 * 2000* : 0.02GB
        x_i = nd.repeat(x_i,repeats= self.sentence_length, axis=1) # 64 * 40 * 40 * 2000: 1.52GB
        x_j = embeds.expand_dims(2) # 64 * 40 * 1 * 2000
        x_j = nd.repeat(x_j,repeats= self.sentence_length, axis=2) # 64 * 40 * 40 * 2000: 1.52GB
        x_full = nd.concat(x_i,x_j,dim=3) # 64 * 40 * 40 * 4000: 3.04GB
        # batch*sentence_length*sentence_length개의 batch를 가진 2*VOCABULARY input을 network에 feed
        _x = x_full.reshape((-1, 2 * self.emb_dim))
        
        _x = self.g_fc1(_x) # (64 * 40 * 40) * 256: .1GB 추가메모리는 안먹나?
        _x = self.g_fc2(_x) # (64 * 40 * 40) * 256: .1GB (reuse)
        # sentence_length*sentence_length개의 결과값을 모두 합해서 sentence representation으로 나타냄
        x_g = _x.reshape((-1, self.sentence_length * self.sentence_length, self.hidden_dim)) # (64, 40*40, 256) : .1GB
        sentence_rep = x_g.sum(1) # (64, 256): ignorable
        return sentence_rep
    
    
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
        x = self.sen_rep(x)
        x = nd.flatten(x)
        res = self.classifier(x)
        return res       
    
    
def evaluate(net, dataIterator, context):
    dataIterator.reset()
    loss = gluon.loss.SigmoidBCELoss()
    total_L = 0.0
    total_sample_num = 0
    total_correct_num = 0
    start_log_interval_time = time.time()
    for i, batch in enumerate(dataIterator):
        data =  batch.data[0].as_in_context(context)
        label = batch.data[1].as_in_context(context)
        output = net(data)
        L = loss(output, label)
        pred = (output > 0.5).reshape((-1,))
        total_L += L.sum().asscalar()
        total_sample_num += len(label)
        total_correct_num += (pred == label).sum().asscalar()
        if (i + 1) % log_interval == 0:
            print('[Batch {}/{}] elapsed {:.2f} s'.format(
                i + 1, dataIterator.num_data//dataIterator.batch_size,
                time.time() - start_log_interval_time))
            start_log_interval_time = time.time()
    avg_L = total_L / float(total_sample_num)
    acc = total_correct_num / float(total_sample_num)
    return avg_L, acc
    
    
def get_pred(net, iterator):
    pred_sa = []
    label_sa = []
    va_text = []
    iterator.reset()
    for i, batch in enumerate(iterator):
        if i % 100 == 0:
            print('i = {}'.format(i))
        data =  batch.data[0].as_in_context(context)
        label = batch.data[1].as_in_context(context)
        output = net(data)
        L = loss(output, label)
        pred = (nd.sigmoid(output) > 0.5).reshape((-1,))
        pred_sa.extend(pred.asnumpy())
        label_sa.extend(label.asnumpy())
        va_text.extend([' '.join([idx2word[np.int(x)] for x in y.asnumpy() if idx2word[np.int(x)] is not 'PAD']) for y in data])
    pred_sa_pd = pd.DataFrame(pred_sa, columns  = ['pred_sa'])
    label_pd = pd.DataFrame(label_sa, columns = ['label'])
    text_pd = pd.DataFrame(va_text, columns = ['text'])
    res = pd.concat([text_pd, pred_sa_pd, label_pd], axis = 1)
    return res

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
                _out = sa(_data)
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
        test_avg_L, test_acc = evaluate(sa, valid_data, context)
        tqdm.write('[Epoch {}] train avg loss {:.6f}, valid acc {:.2f}, \
            valid avg loss {:.6f}, throughput {:.2f}K wps'.format(
            epoch, epoch_L / epoch_sent_num,
            test_acc, test_avg_L, epoch_wc / 1000 /
            (end_epoch_time - start_epoch_time)))


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
    
    ### Preprocessing using Spacy
    nlp = spacy.load("en")
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
    
    train_data = mx.io.NDArrayIter(data=[tr_x, tr_y], batch_size=batch_size, shuffle = False, last_batch_handle = 'discard')
    valid_data = mx.io.NDArrayIter(data=[va_x, va_y], batch_size=batch_size, shuffle = False, last_batch_handle = 'discard')

    context = mx.gpu()

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
    result = get_pred(sa, valid_data)
    ## Classification results
    print('Number of wrong classification = {} out of {}'.format(result[result.pred_sa != result.label].shape[0], result.shape[0]))
    print('Wrongly classified exmples = {}'.format(result[result.pred_sa != result.label].head(10)))