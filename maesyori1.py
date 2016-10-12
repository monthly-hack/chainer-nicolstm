# coding: utf-8
"""
$ python maesyori1.py /path/to/nicocomm/data/thread/
"""
import tarfile
import glob
import sys
import pickle
import logging
import multiprocessing


folder_path = sys.argv[1].rstrip('/')
log_fn = 'log1.txt'
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename=log_fn, level=logging.DEBUG)
targz_list = glob.glob('{}/*.tar.gz'.format(folder_path))
logging.info(len(targz_list))


def get_comment(targz):
    result = []
    tf = tarfile.open(targz, 'r')
    item_list = tf.getnames()[1:]
    for ti in item_list:
        f = tf.extractfile(ti).read()
        comments = f.decode('utf-8').split('\n')
        # 各動画の最後の10コメントを抽出
        comments = comments[-11:-1]
        for comment in comments:
            try:
                comment = eval(comment)['comment']
                result.append(comment)
            except:
                print('null')
    logging.info('end of {}'.format(targz))
    return result

processes = max(1, multiprocessing.cpu_count() - 1)
p = multiprocessing.Pool(processes)
results = p.map(get_comment, targz_list)
logging.info('n_results {}'.format(len(results)))

logging.info('end')
with open('last10comments.pkl', 'wb') as f:
    pickle.dump(results, f)
