import collections, pickle, spacy, time
import mxnet as mx
import numpy as np
import pandas as pd
from datetime import datetime
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
from mxnet.ndarray.linalg import gemm2


def one_hot(x, vocab_size):
    res = np.zeros(shape = (vocab_size))
    res[x] = 1
    return res

def prepare_data(file_name, max_sen_len, max_vocab):
    nlp = spacy.load("en")
    word_freq = collections.Counter()
    max_len = 0
    num_rec = 0
    print('Count words and build vocab...')
    with open(file_name, 'rb') as f:
        for line in f:
            _lab, _sen = line.decode('utf8').strip().split('\t')
            words = [token.lemma_ for token in nlp(_sen) if token.is_alpha] # Stop word제거 안한 상태 
            # 제거를 위해 [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
            if len(words) > max_len:
                max_len = len(words)
            for word in words:
                word_freq[word] += 1
            num_rec += 1

    # most_common output -> list
    word2idx = {x[0]: i+2 for i, x in enumerate(word_freq.most_common(max_vocab - 2))}
    word2idx ['PAD'] = 0
    word2idx['UNK'] = 1

    idx2word= {i:v for v, i in word2idx.items()}
    vocab_size = len(word2idx)

    print('Prepare data...')
    y = []
    x = []
    origin_txt = []
    with open(file_name, 'rb') as f:
        for line in f:
            _label, _sen = line.decode('utf8').strip().split('\t')
            origin_txt.append(_sen)
            y.append(int(_label))
            words = [token.lemma_ for token in nlp(_sen) if token.is_alpha] # Stop word제거 안한 상태
            words = [x for x in words if x != '-PRON-'] # '-PRON-' 제거
            _seq = []
            for word in words:
                if word in word2idx.keys():
                    _seq.append(word2idx[word])
                else:
                    _seq.append(word2idx['UNK'])
            if len(_seq) < max_sen_len:
                _seq.extend([0] * ((max_sen_len) - len(_seq)))
            else:
                _seq = _seq[:max_sen_len]
            x.append(_seq)

    pd.DataFrame(y, columns = ['yn']).reset_index().groupby('yn').count().reset_index()
    return x, y ,origin_txt, idx2word, word2idx


class Sentence_Representation_RNN(nn.Block):
    def __init__(self, emb_dim, hidden_dim, vocab_size, dropout = .2, **kwargs):
        super(Sentence_Representation_RNN, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        with self.name_scope():
            self.hidden = []
            self.embed = nn.Embedding(vocab_size, emb_dim)
            self.lstm = rnn.LSTM(hidden_dim // 2, num_layers= 2, dropout = dropout, input_size = emb_dim, bidirectional=True)
            self.drop = nn.Dropout(.2)

    def forward(self, x, hidden):
        #print('x = {}'.format(x))
        embeds = self.embed(x) # batch * time step * embedding: NTC
        lstm_out, self.hidden = self.lstm(nd.transpose(embeds, (1, 0, 2)), hidden) #TNC로 변환
        _hid = [nd.transpose(x, (1, 0, 2)) for x in self.hidden]
        print('_hid len = {}'.format(len(_hid)))
        # Concatenate depreciated. use concat. input list of tensors
        _hidden = nd.concat(*_hid)
        return lstm_out, self.hidden

    def begin_state(self, *args, **kwargs):
        return self.lstm.begin_state(*args, **kwargs)
    
    
class SA_Classifier(nn.Block):
    def __init__(self, sen_rep, classifier, batch_size, context, **kwargs):
        super(SA_Classifier, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.context = context
        with self.name_scope():
            self.sen_rep = sen_rep
            self.classifier = classifier
            
    def forward(self, x):
        hidden = self.sen_rep.begin_state(func = mx.nd.zeros
                                        , batch_size = self.batch_size
                                        , ctx = self.context)
        print('hidden shape = {}'.format([x.shape for x in hidden]))
        #_x, _ = self.sen_rep(x, hidden)
        _, _x = self.sen_rep(x, hidden) # Use the last hidden step
        print('x shape = {}'.format(_x[0].shape))
        x = nd.reshape(x, (-1,))
        print('xaa = {}'.format(_x[1].shape))
        x = self.classifier(x)
        return x
    
    
def get_pred(net, loss, idx2word, iterator, context):
    pred_sa = []
    label_sa = []
    va_text = []
    iterator.reset()
    for i, batch in enumerate(iterator):
        if i % 100 == 0:
            print('i = {}'.format(i))
        data =  batch.data[0].as_in_context(context)
        label = batch.data[1].as_in_context(context)
        output, _ = net(data)
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

def evaluate(net, loss, dataIterator, log_interval, context):
    dataIterator.reset()
    loss = gluon.loss.SigmoidBCELoss()
    total_L = 0.0
    total_sample_num = 0
    total_correct_num = 0
    start_log_interval_time = time.time()
    for i, batch in enumerate(dataIterator):
        data =  batch.data[0].as_in_context(context)
        label = batch.data[1].as_in_context(context)
        output, _ = net(data)
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
    print('attention shape = {}'.format(att.shape))
    _a = []
    _w = []
    for x, y, z in zip(word, att, original_txt):
        _idx = [i for i, _x in enumerate(x) if _x is not 'PAD']
        _w.append(np.array([x[i] for i in _idx]))
        _a.append(np.array([y[i].asnumpy() for i in _idx]))
        
    _label = [va_y[i] for i in idx]
    _pred = (nd.sigmoid(out) > .5).asnumpy()
    
    fig, axes = plt.subplots(np.int(np.ceil(n_samples / 4)), 4, sharex = False, sharey = True)
    plt.subplots_adjust(hspace=1)
    if mean == True:
        fig.set_size_inches(20, 4)
        plt.subplots_adjust(hspace=5)
    else:
        fig.set_size_inches(20, 20)
        plt.subplots_adjust(hspace=1)
    cbar_ax = fig.add_axes([.91, .3, .04, .4])
    
# https://stackoverflow.com/questions/49899823/changing-width-of-heatmap-in-seaborn-to-compensate-for-font-size-reduction
def plot_neuron_heatmap(text, values, title, n_limit=80, savename='fig1.png',
                        cell_height=0.325, cell_width=0.15, dpi=100):
    from matplotlib import pyplot as plt
    import seaborn as sns
    text = text.replace('\n', '\\n')
    text = np.array(list(text + ' ' * (-len(text) % n_limit)))
    if len(values) > text.size:
        values = np.array(values[:text.size])
    else:
        t = values
        values = np.zeros(text.shape, dtype=np.float32)
        values[:len(t)] = t
    text = text.reshape(-1, n_limit)
    values = values.reshape(-1, n_limit)
    plt.figure(figsize=(cell_width * n_limit, cell_height * len(text)))
    hmap = sns.heatmap(values, annot=text, fmt='', cmap='RdYlGn', xticklabels=False, yticklabels=False, cbar=False)
    plt.subplots_adjust()
    plt.title(title)
    plt.savefig(savename, dpi=dpi)

def draw_sentence(_idx):
    # Get data from valid set for _idx
    _dat = [va_x[i] for i in _idx]

    w_idx = []
    word = [[idx2word[x] for x in y] for y in _dat]
    original_txt = [va_origin[i] for i in _idx]
    out, att = sa(nd.array(_dat, ctx = context)) 
    _a = []
    _w = []
    for x, y, z in zip(word, att, original_txt):
        _ix = [i for i, _x in enumerate(x) if _x is not 'PAD']
        _w.append(np.array([x[i] for i in _ix]))
        _a.append(np.array([y[i].asnumpy() for i in _ix]))

    _label = [va_y[i] for i in _idx]
    _pred = (nd.sigmoid(out) > .5).asnumpy()

    for i, _ix in enumerate(_idx):
        att_score = [] 
        _b = nd.softmax(nd.array(np.mean(_a[i], axis = 1))).asnumpy()
        for x in original_txt[i].split(' '):
            _x_lem = [token.lemma_ for token in nlp(x) if token.is_alpha]
            if len(_x_lem) > 0:
                x_lemma = [token.lemma_ for token in nlp(x) if token.is_alpha][0]
            else:
                x_lemma = ''
            if x_lemma in _w[i]:
                idx = np.argmax(x_lemma == _w[i])
                tmp = [_b[idx]] * len(x)
            else:
                idx = -1
                tmp = [1/len(_w[i])] * len(x)
            tmp.extend([1/len(_w[i])])
            att_score.extend(tmp)
        plot_neuron_heatmap(original_txt[i], att_score[:-1] \
                          , 'Label: {}, Pred: {}'.format(_label[i], np.int(_pred[i])) \
                          , n_limit= len(att_score[:-1]))


def generate_date_data(N, in_seq_len = 32, out_seq_len = 32):
    N_train = int(N * .9)
    N_validation = N - N_train
    
    added = set()
    questions = []
    answers = []
    answers_y = []
    
    while len(questions) < N:
        a = gen_date()
        if a in added:
            continue
        question = '[{}]'.format(a)
        answer = '[' + str(format_date(a)) + ']'
        answer = padding(answer, out_seq_len)
        answer_y = str(format_date(a)) + ']'
        answer_y = padding(answer_y, out_seq_len)

        added.add(a)
        questions.append(question)
        answers.append(answer)
        answers_y.append(answer_y)

    # Check the first 20000 characters to build vocab
    chars = list(set(''.join(questions[:20000])))
    chars.extend(['[', ']']) # Start and End of Expression
    chars.extend(list(set(''.join(answers[:20000]))))
    chars = np.sort(list(set(chars)))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    X = np.zeros((len(questions), in_seq_len, len(chars)), dtype=np.integer)
    Y = np.zeros((len(questions), out_seq_len, len(chars)), dtype=np.integer)
    Z = np.zeros((len(questions), out_seq_len, len(chars)), dtype=np.integer)

    for i in range(N):
        for t, char in enumerate(questions[i]):
            X[i, t, char_indices[char]] = 1
        for t, char in enumerate(answers[i]):
            Y[i, t, char_indices[char]] = 1
        for t, char in enumerate(answers_y[i]):
            Z[i, t, char_indices[char]] = 1
    return X, Y, Z, chars, char_indices, indices_char

def gen_test(N):
    q = []
    y = []
    
    for i in range(N):
        question = gen_date()
        answer_y = format_date(question)
        q.append(question)
        y.append(answer_y)
    return(q,y)

def gen_date():
    rnd = int(np.random.uniform(low = 1000000000, high = 1350000000))
    timestamp = datetime.fromtimestamp(rnd)
    return str(timestamp.strftime('%Y-%B-%d %a')) # '%Y-%B-%d %H:%M:%S'

def format_date(x):
    return str(datetime.strptime(x, '%Y-%B-%d %a').strftime('%m/%d/%Y, %A')).lower() #'%H%M%S:%Y%m%d'


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

    
def padding(chars, maxlen):
    if len(chars) < maxlen:
        return chars + ' ' * (maxlen - len(chars))
    else:
        return chars[:maxlen]
    
    
def calculate_loss(model, data_iter, loss_obj, ctx = mx.cpu()):
    test_loss = []
    for i, (x_data, y_data, z_data) in enumerate(data_iter):
        x_data = x_data.as_in_context(ctx).astype('float32')
        y_data = y_data.as_in_context(ctx).astype('float32')
        z_data = z_data.as_in_context(ctx).astype('float32')
        with autograd.predict_mode():
            z_output = model(x_data, y_data)
            loss_te = loss_obj(z_output, z_data)
        curr_loss = nd.mean(loss_te).asscalar()
        test_loss.append(curr_loss)
    return np.mean(test_loss)


def train(model, tr_data_iterator, te_data_iterator, trainer, loss, char_indices, indices_char, epochs  = 10, ctx = mx.cpu(), output_file_name = '../python/result'):
    tot_test_loss = []
    tot_train_loss = []
    for e in range(epochs):
        train_loss = []
        for i, (x_data, y_data, z_data) in enumerate(tr_data_iterator):
            x_data = x_data.as_in_context(ctx).astype('float32')
            y_data = y_data.as_in_context(ctx).astype('float32')
            z_data = z_data.as_in_context(ctx).astype('float32')

            with autograd.record():
                z_output = model(x_data, y_data)
                loss_ = loss(z_output, z_data)
            loss_.backward()
            trainer.step(x_data.shape[0])
            curr_loss = nd.mean(loss_).asscalar()
            train_loss.append(curr_loss)

        if e % 10 == 0:
            q, y = gen_test(10)
            n_correct = 0
            for i in range(10):
                with autograd.predict_mode():
                    p, attn = model.predict(q[i], char_indices, indices_char, input_digits = x_data.shape[1], lchars = len(indices_char))
                    p = p.strip()
                    iscorr = 1 if p == y[i] else 0
                    if iscorr == 1:
                        print(colors.ok + '☑' + colors.close, end=' ')
                        n_correct += 1
                    else:
                        print(colors.fail + '☒' + colors.close, end=' ')
                    print("{} = {}({}) {}".format(q[i], p, y[i], str(iscorr)))
                if n_correct == 10:
                    #file_name = "format_translator.params"
                    #model.save_parameters(file_name)
                    with open('{}_{}.pkl'.format(output_file_name, e), 'wb') as picklefile:
                        pickle.dump(model, picklefile)
                        
        #caculate test loss
        test_loss = calculate_loss(model, te_data_iterator, loss_obj = loss, ctx=ctx) 
        print("Epoch %s. Train Loss: %s, Test Loss : %s" % (e, np.mean(train_loss), test_loss))    
        tot_test_loss.append(test_loss)
        tot_train_loss.append(np.mean(train_loss))
    return tot_test_loss, tot_train_loss