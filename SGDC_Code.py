
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[13]:


# read in the training dataset
doc = pd.read_csv(filepath_or_buffer='train.dat',header=None, sep='\t')
doc.columns=["class","text"]
list_text = doc['text'].values.tolist()


# In[14]:


#Cleaning the text by removing the stop words
stop = stopwords.words('english')
for i in stop :
    doc = doc.replace(to_replace=r'\b%s\b'%i, value="",regex=True)


# In[15]:


#removing extra characters
array = []
array = doc['text'].str.replace("\)","") 
array1 = array.str.replace("\(","")


# In[17]:


y=doc.iloc[:,0]


# In[18]:


#Convert a collection of text documents to a matrix of token counts
#produces a sparse representation of the counts 
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(max_features= 3000)
X_train = count.fit_transform(array1)


# In[19]:


#to Transform a count matrix to a normalized tf or tf-idf representation
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)


# In[22]:


test_data = pd.read_csv(filepath_or_buffer='test.dat',header=None,sep='\t')

#Cleaning the text 
stop = stopwords.words('english')
for i in stop :
    test_data = test_data.replace(to_replace=r'\b%s\b'%i, value="",regex=True)

test_data.columns=["text"]

#removing extra characters
array2 = []
array2 = test_data['text'].str.replace("\)","") 
array3 = array2.str.replace("\(","")


# In[24]:


from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
text_sgdc = Pipeline([('vect', CountVectorizer(ngram_range=(1, 6))),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, max_iter=5, random_state=42)),
 ])
text_sgdc.fit(array1, y)
predicted_sgdc = text_sgdc.predict(array3)


# In[25]:


output=pd.DataFrame(data=predicted_sgdc)
output.to_csv("Result_SGDC1.dat",index=False,quoting=3,header=None)

