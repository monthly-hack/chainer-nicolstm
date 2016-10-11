# coding: utf-8
"""
予め maesyori1.py を実行して、last10comments.pklを生成してください。
$ python maeshori2.py
"""
import pickle
import random
import unicodedata
import multiprocessing
import logging

log_fn = 'log2.txt'
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=log_fn, level=logging.DEBUG)

logging.info('Open comments file')
with open('last10comments.pkl', 'rb') as f:
    comments = pickle.load(f)
    comments = [c for tf in comments for c in tf]
    random.shuffle(comments)
    sample_comments = comments[:5000000]
logging.info('Done shuffle and sampling')

n_data = len(sample_comments)
vocab = {}
texts = []
logging.info('Sampling {} comments'.format(n_data))

logging.info('Start NFKC')
for text in sample_comments:
    before = ''
    before_count = -2
    try:
        text = unicodedata.normalize('NFKC', text)
    except:
        text = ''

    for i in text:
        if i not in vocab:
            vocab[i] = 1
        else:
            vocab[i] += 1
        if i == before:
            before_count += 1
        else:
            before_count = -2
        before = i

    if before_count >= 1:
        texts.append(text[:-before_count])
    else:
        texts.append(text)

logging.info('End NFKC')
logging.info('n_texts {}'.format(len(texts)))
logging.info('n_vocab {}'.format(len(vocab)))

tmp = sorted(vocab.items(), key=lambda x: x[1])
tmp.reverse()
new_vocab = [i for i, j in tmp[:5000]]


def huga(text):
    flag = True
    for i in text:
        if i not in new_vocab:
            flag = False
    if flag:
        return(text)
    else:
        return('')

logging.info('Start exclude')
processes = max(1, multiprocessing.cpu_count() - 1)
p = multiprocessing.Pool(processes)
new_texts = p.map(huga, texts)
new_texts = [i for i in new_texts if i is not None or '']
logging.info('End exclude')
logging.info('new_texts {}'.format(len(new_texts)))
logging.info('new_vocab {}'.format(len(new_vocab)))

with open('sample_vocab.pkl', 'wb') as f:
    pickle.dump(new_vocab, f)

with open('sample_texts.pkl', 'wb') as f:
    pickle.dump(new_texts, f)
