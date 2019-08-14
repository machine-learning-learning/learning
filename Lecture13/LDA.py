import pickle
import gensim
from pprint import pprint

#livedoor ニュースの記事3400件を形態素解析したリスト
f = open('word_lists_for_gensim', 'rb')
word_lists = pickle.load(f)
f.close()

#最初の3記事のみ表示
pprint(word_lists[:3], compact = True)

#単語辞書を定義する
dictionary = gensim.corpora.Dictionary(word_lists)
dictionary.filter_extremes(no_below = 1, no_above = 0.6)

for key in dictionary.keys()[30:60]:
    print(key, dictionary[key])
print()

corpus = [dictionary.doc2bow(word_list) for word_list in word_lists]

for l in corpus[:3]:
    for t in l:
        print(dictionary[t[0]], t[1])
    print()

lda = gensim.models.LdaModel(corpus = corpus, id2word = dictionary, num_topics = 10)

#トピック分類した結果を表示
for topic in lda.show_topics():
    print(topic)
