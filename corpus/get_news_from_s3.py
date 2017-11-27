# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author  jakey
# Date 2017-11-09 11:17

import boto3
client = boto3.client('s3')
S3 = boto3.resource('s3')


def get_news_content(bucket_name, key_path):
    """
    根据公司名称获取新闻列表
    """
    news_bucket = S3.Bucket(bucket_name)
    try:
        obj = news_bucket.Object(key_path)
        content = obj.get()["Body"].read()
    except Exception as e:
        return ""
    return content


def list_key_objects(bucket_name, prefix_key):
    a = client.list_objects(
        Bucket=bucket_name,
        Prefix=prefix_key
    )
    return [item["Key"] for item in a["Contents"]]


def save_text(content, fold, fname):
    w = open('/'.join([fold, fname]), 'w')
    w.write(content)
    w.close()


def get_s3_text(bucket_name, prefix_key, dst):
    for i, key_path in enumerate(list_key_objects(bucket_name, prefix_key)):
        content = get_news_content(bucket_name, key_path)
        fname = key_path.rsplit('/', 1)[-1]
        save_text(content, dst, fname)
        if i % 100 == 0:
            print '{} finished'.format(i)
        #import pdb
        #pdb.set_trace()

if __name__ == "__main__":
    dst = '/usr/app/tensorflow-learn/corpus/data/news'
    get_s3_text("search-key-news", "text/2017/11/10", dst)
