#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import logging
import os
import sys
import multiprocessing
import pickle
import numpy as np
import pymongo
import re
import pdb

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from collections import Counter


def form_text(data_path, usage='train'):
    """ 生成text文本
    """
    conn = pymongo.MongoClient(['127.0.0.1'])
    db = conn['corpus']
    count = db['news_sentence'].count()
    if usage == 'train':
        total = int(count*0.8)
        data = db['news_sentence'].find({}, {'word':1, 'tag':1, '_id':1})[:int(count*0.8)]
    elif usage == 'test':
        total = count - int(count*0.8)
        data = db['news_sentence'].find({}, {'word':1, 'tag':1, '_id':1})[int(count*0.8):]
    x_input, y_input = [], []
    text = ''
    for i, doc in enumerate(data):
        cur_sentence = doc['word']
        #cur_labels = doc['tag']
        x_row = fill_symbols(cur_sentence, size=32)
        #y_row = fill_symbols(cur_labels, size=32)
        x_input.append(x_row)
        #y_input.append(y_row)
        if i % 10000 == 0:
            print '%s / %s finished!'%(i, total)
    X = np.array(x_input)
    #Y = np.array(y_input)
    text = ' \n'.join([' '.join(map(lambda wd:strQ2B(wd), row)) for row in X])
    print 'start word analysis!'
    wdcnt, wd2id, id2wd = word_analysis(text)
    print 'start save'
    w = open(os.path.join(data_path, 'news_wdinfo.pkl'), 'w')
    pickle.dump({'wdcnt':wdcnt, 'wd2id':wd2id, 'id2wd':id2wd}, w)
    w.close()
    w = open(os.path.join(data_path, 'news.txt'), 'w')
    w.write(text.encode('utf-8'))
    w.close()
    return text, wdcnt

def fill_symbols(wd_list, size=32):
    """ 不足size的补0，超出size的截断
    """
    wd_list = wd_list[:size] + ['PAD'] * (size-min(size, len(wd_list)))
    return wd_list

def word_analysis(text):
    """ 计算文本的word count, 并存pickle
    """
    text = text.decode('utf-8') if isinstance(text, str) else text
    c = Counter()
    for wd in text.split(' '):
        c[wd] += 1
    wd2id = dict(zip(c.keys(), xrange(len(c))))
    wd2id.update({'UNKNOWN':-1, 'PAD':-2})
    id2wd = {_id:wd for (wd,_id) in wd2id.items()}
    return c, wd2id, id2wd


def replace_text_unknown(text, outp, wdcnt, cutoff=2):
    """ form x_input & y_input
    """
    text = text.decode('utf-8') if isinstance(text, str) else text
    wdcnt_cutoff = filter(lambda (x,y):y <= cutoff, wdcnt.items())
    cutoff_len = len(wdcnt_cutoff)
    for i, (wd, cnt) in enumerate(wdcnt_cutoff):
        if re.search(u'[^a-zA-Z0-9\s]', wd):
            text = text.replace(wd, u'UNKNOWN')
        if i % 100 == 0:
            print '{} / {} rep UNKNOWN finished!'.format(i, cutoff_len)
    w = open(outp, 'w')
    w.write(text.encode('utf-8'))
    w.close()
    return


def strQ2B(ustring):
    """ 把字符串全角转半角
    """
    ustring = ustring.decode('utf-8') if isinstance(ustring, str) else ustring
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code==0x3000:
            inside_code=0x0020
        else:
            inside_code-=0xfee0
        # 转完之后不是半角字符则返回原来的字符  
        if inside_code<0x0020 or inside_code>0x7e:
           rstring+=uchar
        else:
            rstring+=(unichr(inside_code))
    return rstring


if __name__ == '__main__':
    """
    # 执行
    python gensim_word2vec.py news.text.model news.text.vector
    # 加载
    model = gensim.models.Word2Vec.load("news.text.model")
    """
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    #data_path = '/home/sst/Documents/socialcredits/data/tenserflow/word2vec'
    data_path = '/usr/app/data/word2vec'
    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    outp1, outp2 = sys.argv[1:3]
    text, wdcnt = form_text(data_path, usage='train')
    # 用unknown替换低频字符的数据集地址
    text_unknown = os.path.join(data_path, 'news.txt.unknown')
    replace_text_unknown(text, text_unknown, wdcnt, cutoff=2)
    
    model = Word2Vec(LineSentence(text_unknown), size=64, window=5, min_count=1,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
