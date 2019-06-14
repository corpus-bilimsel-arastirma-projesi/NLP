# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:28:22 2019

@author: asuer
"""
import pandas as pd
import os
import json
import pickle
from datetime import datetime
import numpy as np
%matplotlib inline

import plotly.plotly as py
import plotly.graph_objs as go

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from IPython.display import display
import base64
import string
import re
from collections import Counter
from time import time


# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.metrics import log_loss

import spacy

import gensim
from gensim.corpora import Dictionary  
import pyLDAvis.gensim
from gensim.models import LdaModel, LsiModel,  HdpModel


''' set english NLP'''
nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
my_stop_words = {u'say', u'\'s', u'mr', u'be', u'said', u'says', u'saying', u'today', u'\n ', u'\n'}
sw=  union = set(spacy_stopwords).union(set(my_stop_words))

'''set wd'''
print(os.getcwd())
print(os.listdir(os.getcwd()))
os.chdir("C:/Users/asuer/Documents/BAP/text_analysis")

'''read files and cobine them in a dataframe'''
uk =pd.read_json("UK_afterJaccard.json")
uk['country'] = 'uk'
cn=pd.read_json("CN_afterJaccard.json")
cn['country'] = 'cn'
us=pd.read_json("US_afterJaccard.json")
us['country'] = 'us'

us['source']=us['source'].str.lower()
us['source']=us['source'].replace('copyright .+: abstracts', '', regex=True)
us['source']=us['source'].replace(' blogs', '', regex=True)
us['source']=us['source'].replace('  ', '', regex=True)
us['source']=us['source'].replace('abstracts', '', regex=True)
us['source']=us['source'].replace('copyright', '', regex=True)
us['source']=us['source'].replace('washington postdc sports bog', 'washington post', regex=True)
us['source']=us['source'].replace('washingtonpost', 'washington post', regex=True)
us['source']=us['source'].str.strip()


df=pd.concat([us, uk, cn])
df = df.drop('docid', 1)
cols=['title', 'date', 'content', 'source','country']
df=df[cols]
df['date']=pd.to_datetime(df['date'])
de=pd.read_csv("dwCleaned.csv",  encoding='utf-8')
de['date'] = de['date'].replace('\n','', regex=True)
de.rename(columns={'Source': 'source'}, inplace=True)
de['country'] = 'de'
de['date'] = pd.to_datetime(de['date'])
df=pd.concat([df, de])

'''make a new year variable'''
yr = []
for date in df['date']:
    xx=date.year
    yr.append(str(xx))
df.loc[:, 'year'] = yr


'''barplot years and source,  clean US source'''

cat = 'source'
fil =df[cat].value_counts()
fil =pd.DataFrame(fil)
fil = fil[(fil.source > 10)]
'''do this for numerical like year'''
fil.plot(title='# of news')
plt.show()
# Barplot of occurances of each category in the  dataset
fil.plot.bar()



import string
df['content'] = df['content'].replace({r'[^\x00-\x7F]+':''}, regex=True)
pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
df['content']=df['content'].str.replace(pattern, ' ')
pattern2 = re.compile('.jpg://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
df['content']=df['content'].str.replace(pattern2, ' ')
df['content'] = df['content'].str.replace('[{}]'.format(string.digits), '')


df['content'] = df['content'].str.replace('[{}]'.format(string.punctuation), '')
df['content'] = df['content'] .str.lower()
df['content'] = df['content'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))


freq = pd.Series(' '.join(df['content']).split()).value_counts()
freq=pd.DataFrame(freq)
freq = freq[(freq ==1)]
freq=freq.dropna(how='all')

freq = list(freq.index)


import re
from collections import Counter

def viterbi_segment(text):
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    return words, probs[-1]

def word_prob(word): return dictionary[word] / total
def words(text): return re.findall('[a-z]+', text.lower())
with open('words_dictionary.json') as json_file:  
    wdic = json.load(json_file) 
dictionary =  Counter(wdic)
max_word_length = max(map(len, dictionary))
total = float(sum(dictionary.values()))

    
vit=[]
for i in freq:
   xx=viterbi_segment(i)
   vit.append(xx)

cc =pd.DataFrame(vit)
vv =list(cc[0])
for i in vv:
        if len(i) >2: # Only if lists is longer than 4 items
            print(i)

    
df['content'] = df['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df['content'].head()


df['word_count'] = df['content'].apply(lambda x: len(str(x).split(" ")))
df['word_count'].mean()
df['scaled']=(df['word_count']-df['word_count'].mean())/df['word_count'].std()
df['scaled'].plot.hist()
df =df[(df['scaled'] < 3) & (df['scaled'] >-3)]###filte 2017
df['scaled'].plot.hist()


def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))
df['avg_word'] = df['content'].apply(lambda x: avg_word(x))
df[['content', 'source','avg_word']].head()


results = set()
df['content'].str.lower().str.split().apply(results.update)
print(results)


df.content.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)



fil=df.groupby('source')['word_count'].mean()
fil.plot.bar()



#processed
# Combine all  text into one large string for wordcloud
all_text = ' '.join([text for text in df['content']])
print('Number of words in all_text:', len(all_text))
# Word cloud for entire  dataset
wordcloud = WordCloud( width=800, height=500, max_words =300, stopwords = sw,
                      random_state=21, max_font_size=110).generate(all_text)
plt.figure(figsize=(15, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#wordcloud for different categories: year and source
val='de'
cat='country'
# Grab all text from cat (year , source)
fil = df[df[cat] == val]###filte 2017
fil_text = ' '.join(text for text in fil['content'])
print('Number of words in eap_text:', len(fil_text))

wordcloud = WordCloud( width=800, height=500,
                      random_state=21, max_font_size=110).generate(fil_text)
plt.figure(figsize=(15, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()






new_stop_words = {u'time', u'number', u'year', u'imgs',u'jpg', u'syndigate', u'http', u'including',
                  u'Copyright', u' \n '}
swo=  union = set(sw).union(set(new_stop_words))
'''clean further sw'''
import re
pattern = re.compile(r'\b(' + r'|'.join(swo) + r')\b\s*')#regex to clean stopwords from texts
txt = []
for document in df["content"]:
    document=re.sub(r'[^\x00-\x7f]',r'', document)#non-ascii
    doc =  pattern.sub('', document)
    txt.append(doc)

df['content'] =txt
'''go back to line 95 until satisfied'''


do = df[df['country'] == 'de']###filte 2017

do=do['content']


start = time()
nlp = spacy.load("en_core_web_sm", disable = ['textcat']) #, disable = ['parser', 'textcat'])
for stopword in swo:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True
texts = []
for i in do:
    i=re.sub(r'[^\x00-\x7f]',r'', i)
    doc =  nlp(i)
    texts.append(doc)
end = time()
print(end - start)

'''
with open("mycorpus.txt", "wb") as fp:   #Pickling
   pickle.dump(texts, fp)
   '''
   
with open("mycorpus.txt", "rb") as fp:   # Unpickling
   texts = pickle.load(fp)   


import sys
sys.getsizeof(texts)


start = time()
txts=[]   
for i in texts:
    ion=[]
    for w in i:
         if  not w.is_stop and not w.is_punct and not w.like_num and w.lemma_ !='-PRON-':
            ion.append((w.lemma_))
    txts.append(ion)   

bigram = gensim.models.Phrases(txts)
txts = [bigram[line] for line in txts]
dictionary = Dictionary(txts)


txts = [bigram[line] for line in txts]
[bigram[line] for line in txts]
dictionary = Dictionary(txts)

corpus = [dictionary.doc2bow(text) for text in txts]
#tf=idf traNnsformation
from gensim import models
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
end=time()
print(end - start)


Counter(txts[3]).most_common(20) 


######
#Alpha: the document-topic density. Higher:  documents are composed of more topics 
# Beta:  topic-word density. higHER: topics are composed of a large number of words 

ldamodel = gensim.models.LdaModel(corpus=corpus_tfidf, num_topics=25, id2word=dictionary)
ldamodel.show_topics()
ldamodel.show_topics(25)
# document-topic proportions


lsimodel = gensim.models.LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)
lsimodel.show_topics(num_topics=10)  # Showing only the top 5 topics


from gensim.models import HdpModel

hdpmodel = gensim.models.HdpModel(corpus=corpus, id2word=dictionary)
hdpmodel.show_topics()  # Showing only the top 5 topics




from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF


no_features = 1000

start=time()
# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(do)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(do)
tf_feature_names = tf_vectorizer.get_feature_names()
end=time()

no_top_words = 10

feature_names=tfidf_feature_names
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
            
    
# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)            












