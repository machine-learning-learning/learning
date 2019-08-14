#gensim用に記事を形態素解析して名詞だけを抽出したリストを作るやつ

import os
import time
import pickle
import MeCab
from tqdm import tqdm

#ニュース記事はlivedoorニュースコーパスを使用
os.chdir('./articles')
ls = os.listdir()

word_list = []

#当然のことながら辞書はmecab-ipadic-neologdを使用
mecab = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

for file in tqdm(ls):
    keywords = []
    f = open(file)
    node = mecab.parseToNode(f.read())
    while node:
        #「〜さん」などの接尾を除外して名詞をリストに挿入していく
        if node.feature.split(',')[0] == u'名詞' and node.feature.split(',')[1] != '接尾':
            keywords.append(node.surface)
        node = node.next
    word_list.append(keywords)
    f.close()

os.chdir('..')
print(word_list)
f = open('word_lists_for_gensim', 'wb')
pickle.dump(word_list, f)
f.close()
