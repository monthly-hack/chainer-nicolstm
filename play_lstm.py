import chainer.links as L
import chainer.serializers as S
import pickle
import nico_lstm
import unicodedata
import numpy as np
import six


with open('./sample_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    vocab += ['<soc>', '<eoc>']  # start of comment, end of comment
    index2vocab = {i: word for i, word in enumerate(vocab)}
    vocab2index = {word: i for i, word in enumerate(vocab)}

n_vocab = len(index2vocab)
rnn = nico_lstm.RNNForLM(n_vocab, 650)
model = L.Classifier(rnn)
S.load_npz('./result/lstm_model.npz', model)


def make_comment(word):
    try:
        com = unicodedata.normalize('NFKC', word)
    except:
        return '{} is not in vocab'.format(word)

    rnn.reset_state()
    init = rnn(np.asarray([vocab2index['<soc>']], dtype=np.int32), train=False)
    comment = ''
    for i in list(com):
        a = rnn(np.asarray([vocab2index[i]], dtype=np.int32), train=False)
        comment += i

    while True:
        now = index2vocab[int(np.argmax(a.data))]
        if now == '<eoc>':
            break
        elif len(comment) > 20:
            break
        comment += now
        a = rnn(np.asarray([vocab2index[now]], dtype=np.int32), train=False)

    return comment

try:
    while True:
        q = six.moves.input('>> ')
        print(make_comment(q))

except EOFError:
    pass
