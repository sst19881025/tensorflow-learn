# -*- encoding: utf-8 -*-

from __future__ import print_function, unicode_literals
from bosonnlp import BosonNLP

import os
import re
import pdb
import pymongo

# 注意：在测试时请更换为您的API token。
nlp = BosonNLP('Ofzs3sqP.19198.Ax9F9vqfSTYG')



def nlp_seg(text):
    """ boson 分词标注
    # nlp.tag 完整的参数调用格式如下：
    # result = nlp.tag(s, space_mode=0, oov_level=3, t2s=0, special_char_conv=0)
    # 修改space_mode选项为1，如下：
    # result = nlp.tag(s, space_mode=1, oov_level=3, t2s=0, special_char_conv=0)
    # 修改oov_level选项为1，如下：
    # result = nlp.tag(s, space_mode=0, oov_level=1, t2s=0, special_char_conv=0)
    # 修改t2s选项为1，如下：
    # result = nlp.tag(s, space_mode=0, oov_level=3, t2s=1, special_char_conv=0)
    # 修改特殊字符转换选项为1,如下：
    # result = nlp.tag(s, space_mode=0, oov_level=3, t2s=0, special_char_conv=1)
    """
    token_str = u'，,；;、‘’“”！!？?：:。.'
    text = text.decode('utf-8') if isinstance(text, str) else text
    text = re.sub(u'(?<=[\u2E80-\u9FFF{}])[\s]+(?=[\u2E80-\u9FFF{}])'.format(\
                        token_str, token_str), '',  text).strip()
    sentence_list = [text] # map(lambda x:x.strip(), re.split(u'[,，;；.。?？!！]', text))
    result = nlp.tag(sentence_list, space_mode=1, oov_level=3, t2s=1, special_char_conv=0)
    return result


def save_data(_id, word, tag):
    conn = pymongo.MongoClient(['127.0.0.1:27017'])
    db = conn['corpus']
    #pos_tag_text = map(lambda x:' '.join(['%s/%s' % it for it in zip(x['word'], x['tag'])]), pos_tag)
    #db.news_pos_tag.save({'_id':_id, 'pos_tag':pos_tag, 'text':pos_tag_text})
    db['news_train'].save({'_id':_id, 'word':word, 'tag':tag})


def main(data_path):
    """ 存储词性标注结果到mongo
    """
    name_list = os.listdir(data_path)
    for i, name in enumerate(name_list):
        if name.endswith('~'):
            continue
        news = open(os.path.join(data_path, name)).read()
        wd_tag_data = nlp_seg(news)[0]
        wd_data = wd_tag_data['word']
        tag_data = wd_tag_data['tag']
        save_data(_id=name, word=wd_data, tag=tag_data)
        if i % 100 == 0:
            print('{} finished'.format(i))
        #pdb.set_trace()

def res_text(result):
    return map(lambda x:' '.join(['%s/%s' % it for it in zip(x['word'], x['tag'])]), result)


if __name__ == '__main__':
    data_path = '/usr/app/tensorflow-learn/corpus/data/news/'
    main(data_path)

    #text = '拉卡拉集团装入A股西藏旅游流产后，分拆支付业务单独在创业板上市，融资约20亿元，证监会今日披露了其招股书。 根据招股书，拉卡拉支付股份有限公司（以下简称拉卡拉支付）计划公开发行不超过4001万股，融资规模在20亿元左右，资金计划全部用于第三方支付产业升级项目。中信建设证券股份有限公司是此次公开发行的保荐人。 2016年5月份，拉卡拉集团曾筹划作价110亿元装入A股西藏旅游，但该交易方案出炉后被市场质疑涉嫌刻意规避借壳，上交所一度发出问询函要求重组双方说明此事，最终双方以政策变化等原因终止了交易。 此次计划在创业办上市的拉卡拉支付业务，只是原拉卡拉集团业务的一部分。 2016年10月份，拉卡拉集团曾对外公布业务重组方案：成立拉卡拉控股作为最高层的母公司，下设拉卡拉支付集团和考拉金服集团。 拉卡拉控股董事长孙陶然曾解释重组原因为，一方面是业务发展的需要，另外一方面也是根据监管的要求：拉卡拉支付集团旗下业务是由“一行三会”发牌照的业务，考拉金服集团旗下业务金融办、金融局来监管的业务。 招股书显示，2016年第四季度，拉卡拉支付剥离了主营增值金融业务的北京拉卡拉小贷、广州拉卡拉小贷等10家控股及参股子公司。 目前，拉卡拉支付经营范围为银行卡收单、互联网支付、数字电视支付、预付卡受理、移动电话支付。 拉卡拉支付股权结构图 股权结构方面，拉卡拉支付第一大股东为联想控股，直接持有公司31.38%股份，孙陶然和孙浩然合计直接持股比例为13.06%。 不过，招股书称拉卡拉支付目前无实际控制人。原因为，第一，根据联想控股出具《关于未对拉卡拉支付股份有限公司实施控制的声 明函》，联想控股对拉卡拉支付仅为财务性投资入股，以获取投资收益为目的，不单独或联合谋求对公司的控制；第二，拉卡拉支付的股权结构相对分散，且主要股东之间不存在一致行动安排，故无关联股东能实际支配公司经营。 经营业绩方面，招股书显示，拉卡拉支付2013、2014年度承受着亏损，净利润分别为 -12681.03万元、－19694.7万元，2015年起扭亏为盈，净利润为12356.34万元，2016年1月9日净利润21171.59万元。 2013年至2015年，拉卡拉支付营业收入分别为61707.93万元、91522.84万元、158838.94万元、199398.09万元。 [责编：张帆]'
    #result = nlp_pos_tag(text)
    #pdb.set_trace()
