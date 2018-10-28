import logging
import mxnet as mx
import numpy as np
from utils import *

class alignment(gluon.HybridBlock):
    def __init__(self, n_hidden, **args):
        super(alignment, self).__init__(**args)
        with self.name_scope():
            self.weight = self.params.get('weight', shape = (n_hidden, n_hidden), allow_deferred_init = True)
        
    def hybrid_forward(self, F, inputs, output, weight):
        _s = F.dot(inputs, weight)
        return gemm2(_s, output)
    
    
class format_translator(gluon.Block):
    def __init__(self, n_hidden, in_seq_len, out_seq_len, vocab_size, ctx, **kwargs):
        super(format_translator, self).__init__(**kwargs)
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.ctx = ctx
        
        with self.name_scope():
            self.encoder = rnn.LSTMCell(hidden_size = n_hidden)
            self.decoder = rnn.LSTMCell(hidden_size = n_hidden)
            self.alignment = alignment(n_hidden)
            self.attn_weight = nn.Dense(self.in_seq_len, in_units = self.in_seq_len)            
            self.batchnorm = nn.BatchNorm(axis = 2)
            self.dense = nn.Dense(self.vocab_size, flatten = False)
            
    def forward(self, inputs, outputs):
        self.batch_size = inputs.shape[0]
        enout, (next_h, next_c) = self.encoder.unroll(inputs = inputs, length = self.in_seq_len, merge_outputs = True)
        
        for i in range(self.out_seq_len):
            # For each time step, caclculate context for attention
            # Use enout(batch_size * in_seq_len * n_hidden)
            _n_h = next_h.expand_dims(axis = 2)       
            score_i = self.alignment(enout, _n_h)
            # Create attention weight: alpha_1, ... alpha_(in_seq_len)
            alpha_i = nd.softmax(self.attn_weight(score_i))
            # alpha:(n_batch * in_seq_len) -> Expand alpha to (n_batch * in_seq_len * n_hidden)
            alpha_i = nd.softmax(self.attn_weight(score_i))
            # alpha:(n_batch * in_seq_len) -> Expand alpha to (n_batch * in_seq_len * n_hidden)
            alpha_expand = alpha_i.expand_dims(2)
            alpha_expand = nd.repeat(alpha_expand,repeats= self.n_hidden, axis=2) # n_batch * time step * n_hidden
            context = nd.multiply(alpha_expand, enout)
            context = nd.sum(context, axis = 1) # n_batch * n_hidden
            _in = nd.concat(outputs[:, i, :], context)
            #print('in shape = {}'.format(_in.shape))
            deout, (next_h, next_c) = self.decoder(_in, [next_h, next_c],)
            #print('deout shape = {}'.format(deout.shape))
            if i == 0:
                deouts = deout
            else:
                deouts = nd.concat(deouts, deout, dim = 1)   
        deouts = nd.reshape(deouts, (-1, self.out_seq_len, self.n_hidden))
        deouts = self.batchnorm(deouts)
        deouts_fc = self.dense(deouts)
        return deouts_fc
    
    def predict(self, input_str, char_indices, indices_char, input_digits = 9, lchars = 14, ctx = mx.cpu()):
        # No label when evaluating new example. So try to put the result of the previous time step
        alpha = []
        input_str = '[' + input_str + ']'
        X = nd.zeros((1, input_digits, lchars), ctx = ctx)
        for t, char in enumerate(input_str):
            X[0, t, char_indices[char]] = 1
        Y_init = nd.zeros((1, lchars), ctx = ctx)
        Y_init[0, char_indices['[']] = 1
        enout, (next_h, next_c) = self.encoder.unroll(inputs = X, length = self.in_seq_len, merge_outputs = True)
        deout = Y_init

        for i in range(self.out_seq_len):
            _n_h = next_h.expand_dims(axis = 2)
            ####### Attention part: To get context vector at jth point of output sequence
            score_i = self.alignment(enout, _n_h)
            alpha_i = nd.softmax(self.attn_weight(score_i))
            # alpha:(n_batch * in_seq_len) -> Expand alpha to (n_batch * in_seq_len * n_hidden)
            alpha_expand = alpha_i.expand_dims(2)
            alpha_expand = nd.repeat(alpha_expand,repeats= self.n_hidden, axis=2) # n_batch * time step * n_hidden
            context = nd.multiply(alpha_expand, enout)
            context = nd.sum(context, axis = 1) # n_batch * n_hidden
            
            _in = nd.concat(deout, context)
            deout, (next_h, next_c) = self.decoder(_in, [next_h, next_c],)
            deout = nd.expand_dims(deout, axis = 1)
            deout = self.batchnorm(deout)
            deout = deout[:, 0, :]
            deout_sm = self.dense(deout)
            deout = nd.one_hot(nd.argmax(nd.softmax(deout_sm, axis = 1), axis = 1), depth = self.vocab_size)
            if i == 0:
                ret_seq = indices_char[nd.argmax(deout_sm, axis = 1).asnumpy()[0].astype('int')]
            else:
                ret_seq += indices_char[nd.argmax(deout_sm, axis = 1).asnumpy()[0].astype('int')]
                
            if ret_seq[-1] == ']':
                break
            alpha.append(alpha_i.asnumpy())
        return ret_seq.strip(']').strip(), np.squeeze(np.array(alpha), axis = 1)
    
    
if __name__ == '__main__':
    from datetime import datetime
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    N = 1000
    N_train = int(N * .9)
    N_validation = N - N_train

    in_seq_len = 32
    out_seq_len = 32

    X, Y, Z, chars, char_indices, indices_char = generate_date_data(N)

    # train test split
    X_train, X_validation, Y_train, Y_validation, Z_train, Z_validation = \
        train_test_split(X, Y, Z, train_size=N_train)

    # Create dataloader
    tr_set = gluon.data.ArrayDataset(X_train, Y_train, Z_train)
    tr_data_iterator = gluon.data.DataLoader(tr_set, batch_size=256, shuffle=True)

    te_set =gluon.data.ArrayDataset(X_validation, Y_validation, Z_validation)
    te_data_iterator = gluon.data.DataLoader(te_set, batch_size=256, shuffle=True)

    # Define model, trainer, and loss
    ctx = mx.cpu()
    model = format_translator(300, in_seq_len, out_seq_len, len(chars), ctx)
    model.collect_params().initialize(mx.init.Xavier(), ctx = ctx)

    trainer = gluon.Trainer(model.collect_params(), 'rmsprop')
    loss = gluon.loss.SoftmaxCrossEntropyLoss(axis = 2, sparse_label = False)

    # Train model
    res = train(model, tr_data_iterator, te_data_iterator \
              , trainer, loss, char_indices, indices_char
              , epochs= 300, ctx = ctx, output_file_name = '../models/multiplicative_attention_{}'.format(datetime.now().strftime('%Y-%m-%d')))
