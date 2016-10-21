# coding: utf-8

# NStepLSTM Sample Code
# ----
# Auther: ron zacapa @ Keio Univ.
# Date: 2016/10/19

import argparse
import logging
import pickle
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, link, reporter
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy


class RNN(chainer.Chain):

    def __init__(self, n_layer, n_vocab, n_units, dropout, use_cudnn):
        super(RNN, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.NStepLSTM(n_layer, n_units, n_units,
                           dropout, use_cudnn=use_cudnn),
            l2=L.Linear(n_units, n_vocab),
        )

    def __call__(self, hx, cx, xs, train=True):
        xs = [self.embed(item) for item in xs]
        hy, cy, ys = self.l1(hx, cx, xs, train=train)
        y = [self.l2(item) for item in ys]
        return y


class Classifier(link.Chain):
    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, *args, train=True):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x, train)
        for yi, ti in zip(self.y, t):
            if self.loss is not None:
                self.loss += self.lossfun(yi, ti)
            else:
                self.loss = self.lossfun(yi, ti)

        reporter.report({'loss': self.loss}, self)
        count = 0
        if self.compute_accuracy:
            for yi, ti in zip(self.y, t):
                if self.accuracy is not None:
                    self.accuracy += self.accfun(yi, ti) * len(ti)
                    count += len(ti)
                else:
                    self.accuracy = self.accfun(yi, ti) * len(ti)
                    count += len(ti)
            self.accuracy

            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss, self.accuracy, count

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', '-b', type=int, default=500,
                    help='Number of examples in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=2,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--gradclip', '-c', type=float, default=5,
                    help='Gradient norm threshold to clip')
parser.add_argument('--out', '-o', default='result',
                    help='Directory to output the result')
parser.add_argument('--unit', '-u', type=int, default=500,
                    help='Number of LSTM units in each layer')
parser.add_argument('--layer', '-l', type=int, default=2,
                    help='Number of LSTM layer')
parser.add_argument('--dropout', '-d', type=float, default=0.5,
                    help='Dropout ratio')
parser.add_argument('--log', '-r', type=str, default='log',
                    help='Name of logfile')
parser.add_argument('--cudnn', dest='use_cudnn', action='store_true')
parser.add_argument('--no-cudnn', dest='use_cudnn', action='store_false')
parser.set_defaults(use_cudnn=True)
args = parser.parse_args()

log_fn = '{}/{}.txt'.format(args.out, args.log)
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=log_fn, level=logging.DEBUG)

with open('./sample_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    vocab.append('<EOF>')
    vocab2index = {word: i for i, word in enumerate(vocab)}

with open('./sample_texts.pkl', 'rb') as f:
    texts = pickle.load(f)
    train = [[vocab2index[word] for word in text]
             for text in texts if text != '']

n_vocab = len(vocab)
logging.info('#vocab = {}'.format(n_vocab))
n_train = len(train)
logging.info('#train = {}'.format(n_train))
n_words = sum([len(text) for text in train])
logging.info('#words = {}'.format(n_words))

model = RNN(args.layer, n_vocab, args.unit,
            args.dropout, args.use_cudnn)

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()
    xp = cuda.cupy
else:
    xp = np

classify = Classifier(model)

optimizer = chainer.optimizers.SGD(lr=1.0)
optimizer.setup(classify)
optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

train_now = np.asarray(train)
train_next = np.asarray([item[1:] + [vocab2index['<EOF>']] for item in train])

for e in range(args.epoch):
    sum_loss = 0
    sum_acc = 0
    sum_count = 0
    perm = np.random.permutation(n_train)
    for i in range(0, n_train, args.batchsize):
        logging.info('# minibatch = {}/{}'.format(i, n_train))
        xs = [xp.asarray(item, dtype=np.int32)
              for item in train_now[perm[i:i + args.batchsize]]]
        hx = chainer.Variable(
            xp.zeros((args.layer, len(xs), args.unit), dtype=xp.float32))
        cx = chainer.Variable(
            xp.zeros((args.layer, len(xs), args.unit), dtype=xp.float32))
        t = [xp.asarray(item, dtype=np.int32)
             for item in train_next[perm[i:i + args.batchsize]]]

        loss, acc, count = classify(hx, cx, xs, t)
        sum_loss += loss.data
        sum_acc += acc.data
        sum_count += count
        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

    logging.info('# epoch = {}, loss = {}, acc = {}'.format(
        e, sum_loss, sum_acc / count))
    chainer.serializers.save_npz(
        '{}/{}'.format(args.out, 'nslstm_model.npz'), model)
