import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize,RegexpTokenizer
import string
import nltk
from nltk.corpus import stopwords
import en_core_web_sm
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import operator
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from wordcloud import WordCloud, STOPWORDS 
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster


#Function to run word2vec model
def runword2vec(final_txt):
    model = gensim.models.Word2Vec(final_txt,size=150,window=10,min_count=10,workers=10,iter=10)
    return model

#function to get the tfidf matrix
def tfidfwords(final_txt):
    cv=CountVectorizer()
    word_count_vector=cv.fit_transform([ ' '.join(i) for i in final_txt ])
    vect = TfidfVectorizer()
    tfidf_matrix = vect.fit_transform([ ' '.join(i) for i in final_txt ])
    vocab = [v[0] for v in sorted(vect.vocabulary_.items(), key=operator.itemgetter(1))]
    return tfidf_matrix,vocab
    
#function to get list of postive and negative words list from reviews
#reviews with star rating>4 is positive and <2 is negative
def getPosNegWords(rev1,vocab,tfidf_matrix):
    rev1['score']=rev1.stars-rev1.stars.mean()
    rev1=rev1.reset_index()
    pos_lis=list(rev1[rev1.stars>4].index)
    neg_lis=list(rev1[rev1.stars<2].index)
    pos=np.ravel(tfidf_matrix.toarray()[pos_lis].sum(axis=0))
    neg=np.ravel(tfidf_matrix.toarray()[neg_lis].sum(axis=0))
    posd=dict(zip(vocab, pos))
    negd=dict(zip(vocab, neg))
    poswords=sorted(posd.items(),key=operator.itemgetter(1),reverse=True)
    negwords=sorted(negd.items(),key=operator.itemgetter(1),reverse=True)
    return poswords,negwords

#function to visulize the word cloud
def sentimentcloud(poswords,negwords):
    wordcloud = WordCloud(width=1600, height=800, random_state=1, max_words=500, background_color='white',)

    wordcloud.generate_from_frequencies(dict(poswords[:50]))
    plt.figure(figsize=(20,10))
    plt.title("Positive words", fontsize=40,color='Red')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=10)
    plt.show()
    plt.savefig('pos.png')


    wordcloud.generate_from_frequencies(dict(negwords[:50]))
    plt.figure(figsize=(20,10))
    plt.title("Negative Sentiment", fontsize=40,color='Red')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=10)
    plt.savefig('neg.png')

#function to create dendogram to view the word topics
def getDendrogramTopics(tfidf_matrix,vocab,model):
    freq = np.ravel(tfidf_matrix.sum(axis=0))
    fdist = dict(zip(vocab, freq)) 
    topwords=[i[0] for i in sorted(fdist.items(),key=operator.itemgetter(1),reverse=True) if i[0] in model.wv.index2word][:50]
    corpus = topwords 
    word_vector = []
    for i in corpus:
        word_vector.append(model[i]) # Gets the word_vectors of each word
        word_vector1 = np.array(word_vector)#converting it into Numpy array
    HC = linkage(word_vector, 'ward')
    plt.figure(figsize=(15, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Words')
    plt.ylabel('distance')
    dendrogram(
        HC,labels=topwords,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.savefig('topics.png')
    
def main(argv):
    rev=pd.read_csv(sys.argv[1],encoding='UTF-8')#read the reviews file
    filt_condition=sys.argv[2]#Filter condition can be on Businees,user,category, or city
    filt_variable=sys.argv[3]#filter value
    rev1=rev[rev[filt_condition]==filt_variable]
    print len(rev),len(rev1)
    rev1['tokenized_text'] = rev1['text'].apply(lambda x:x.lower()).apply(word_tokenize) #word tokenization
    punctuation = list(string.punctuation)
    stopWords = set(stopwords.words('english'))
    docs=list(rev1['tokenized_text'])
#stop word removal
    filtered_docs=[]
    for i in docs:
        filt = []
        for w in i:
            if w.lower()not in stopWords and w not in punctuation:
                filt.append(w)
        filtered_docs.append(filt)
#stemming
    nlp = en_core_web_sm.load()

    final_txt=[[t.lemma_ for t in nlp(' '.join(i))] for i in filtered_docs]
    print len(final_txt)


    model=runword2vec(final_txt)

    tfidf_matrix,vocab=tfidfwords(final_txt)
    poswords,negwords=getPosNegWords(rev1,vocab,tfidf_matrix)
    sentimentcloud(poswords,negwords)
    getDendrogramTopics(tfidf_matrix,vocab,model)


if __name__ == "__main__":
    main(sys.argv)



