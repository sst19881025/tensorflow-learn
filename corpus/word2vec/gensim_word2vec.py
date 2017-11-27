#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import logging
import os
import sys
import multiprocessing
import pickle
import pdb

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from collections import Counter



def text_wd_cnt(text, outp):
    """ 计算文本的word count, 并存pickle
    """
    text = text.decode('utf-8') if isinstance(text, str) else text
    c = Counter()
    for wd in text.split(' '):
        c[wd] += 1
    w = open(outp, 'w')
    pickle.dump(c, w)
    w.close()
    return c


def replace_text_unknown(text, outp, wdcnt, cutoff=2):
    text = text.decode('utf-8') if isinstance(text, str) else text
    wdcnt_cutoff = filter(lambda (x,y):y <= cutoff, wdcnt.items())
    cutoff_len = len(wdcnt_cutoff)
    for i, (wd, cnt) in enumerate(wdcnt_cutoff):
        #pdb.set_trace()
        if wd != u' ':
            text = text.replace(wd, u'unknown')
        print '{} / {} finished!'.format(i, cutoff_len)
    w = open(outp, 'w')
    w.write(text.encode('utf-8'))
    w.close()
    return


if __name__ == '__main__':
    """
    # 执行
    python gensim_word2vec.py /home/sst/Documents/socialcredits/data/tenserflow/word2vec/news.text \
            news.text.model news.text.vector news.text.wdcnt.pkl
    # 加载
    model = gensim.models.Word2Vec.load("news.text.model")
    """
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 5:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp1, outp2, wdcnt_outp = sys.argv[1:5]
    text = open(inp, 'r').read().decode('utf-8')
    wdcnt = text_wd_cnt(text, wdcnt_outp)
    # 用unknown替换低频字符的数据集地址
    outp_unknown = '%s.unknown'%inp
    replace_text_unknown(text, outp_unknown, wdcnt, cutoff=2)
    
    model = Word2Vec(LineSentence(outp_unknown), size=64, window=5, min_count=1,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
