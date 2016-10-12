#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import argparse

import numpy as np
import pickle

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions



class RNNForLM(chainer.Chain):

    def __init__(self, n_vocab, n_units):
        super(RNNForLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x, train=True):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=train))
        h2 = self.l2(F.dropout(h1, train=train))
        y = self.l3(F.dropout(h2, train=train))
        return y



class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        self.iteration = 0

    def __next__(self):
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            batch = train_iter.__next__()
            x = np.asarray([example[0] for example in batch], dtype=np.int32)
            t = np.asarray([example[1] for example in batch], dtype=np.int32)
            if self.device >= 0:
                x = chainer.cuda.to_gpu(x, device=self.device)
                t = chainer.cuda.to_gpu(t, device=self.device)
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=1000,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=10,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=2,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    args = parser.parse_args()

    # Load nico-douga comments dataset
    with open('sample_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        vocab += ['<soc>', '<eoc>']  # start of comment, end of comment
        index2vocab = {i: word for i, word in enumerate(vocab)}
        vocab2index = {word: i for i, word in enumerate(vocab)}
    with open('sample_texts.pkl', 'rb') as f:
        texts = pickle.load(f)
        train = []
        for text in texts:
            train.append(vocab2index['<soc>'])
            for i in text:
                train.append(vocab2index[i])
            train.append(vocab2index['<eoc>'])

    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)
    n_train = len(train)  # train is just an array of integers
    print('#train =', n_train)
    train_iter = ParallelSequentialIterator(train, args.batchsize)

    # Prepare an RNNLM model
    rnn = RNNForLM(n_vocab, args.unit)
    model = L.Classifier(rnn)
    model.compute_accuracy = True  # we only want the perplexity
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    interval = 10 if args.test else 500
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'main/accuracy']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(
        update_interval=1 if args.test else 10))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, model)

    @training.make_extension(trigger=(500, 'iteration'))
    def save_model(trainer):
        chainer.serializers.save_npz('{}/{}'.format(
                                     args.out, 'lstm_model.npz'), model)

    trainer.extend(save_model)

    trainer.run()

if __name__ == '__main__':
    main()
