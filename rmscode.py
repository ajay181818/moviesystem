#!/usr/bin/env python
# coding: utf-8

# # 🎥 Movies Recommender System�

# Importing the basic libraries 

# In[1]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# In[2]:


movies = pd.read_csv('movies_metadata.csv')
credits = pd.read_csv('credits.csv',dtype={'id':str})
keywords = pd.read_csv('keywords.csv',dtype={'id':str})
links=pd.read_csv("links.csv")
rating=pd.read_csv("ratings_small.csv")


# In[3]:


movies


# In[4]:


credits


# In[5]:


movies.columns


# In[6]:


credits.columns


# In[7]:


links.columns


# In[8]:


rating.columns


# In[9]:


keywords.columns


# In[10]:


movies.nunique()


# In[11]:


rating.nunique()


# In[12]:


movies.info()


# In[13]:


credits.info(
)


# In[14]:


links.info()


# In[15]:


rating.info()


# In[16]:


movies_d = movies.merge(credits, on ="id")


# In[17]:


movies_d


# In[18]:


movies_df = movies_d.merge(keywords, on ="id")


# In[19]:


movies_df


# In[20]:


links=links.merge(rating,on="movieId")


# In[21]:


links


# In[22]:


links.info()


# In[23]:


links['movieId'] = links.imdbId.astype(str)


# In[24]:


links.rename({'movieId': 'id'}, axis=1, inplace=True)


# In[25]:


links


# In[26]:


movies_df1=movies_df.merge(links, on ="id")


# In[27]:


movies_df1.head(2)


# In[28]:


movies_df.shape


# In[29]:


movies_fd = movies_df[['id', 'title','overview','genres','cast','crew','keywords']]


# In[30]:


movies_fd.head(5)


# In[31]:


movies_fd.dropna(inplace = True)


# In[32]:


movies_fd.isna().sum()


# In[33]:


movies_fd.shape


# In[34]:


movies_fd.duplicated().sum()


# In[35]:


movies_fd.drop_duplicates(keep=False, inplace=True)


# In[36]:


movies_fd.shape


# In[37]:


import ast


# In[38]:


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 


# In[39]:


movies_fd['genres'] = movies_fd['genres'].apply(convert)
movies_fd.head()


# In[40]:


movies_fd["keywords"][1]


# In[41]:


movies_fd['keywords'] = movies_fd['keywords'].apply(convert)
movies_fd.head()


# In[42]:


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 


# In[43]:


movies_fd['cast'] = movies_fd['cast'].apply(convert3)
movies_fd.head()


# In[44]:


movies_fd["crew"][1]


# In[45]:


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 


# In[46]:


movies_fd['crew'] = movies_fd['crew'].apply(fetch_director)
movies_fd.head()


# In[47]:


movies_fd['overview'] = movies_fd['overview'].apply(lambda x:x.split())


# In[48]:


movies_fd


# In[49]:


movies_fd["genres"]


# In[50]:


def reducingspace(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


# In[51]:


movies_fd['cast'] = movies_fd['cast'].apply(reducingspace)
movies_fd['crew'] = movies_fd['crew'].apply(reducingspace)
movies_fd['genres'] = movies_fd['genres'].apply(reducingspace)
movies_fd['keywords'] = movies_fd['keywords'].apply(reducingspace)


# In[52]:


movies_fd.head(2)


# In[53]:


movies_fd['tags'] = movies_fd['overview'] + movies_fd['genres']  + movies_fd['cast'] + movies_fd['crew']+ movies_fd['keywords']


# In[54]:


movies_fd


# In[55]:


fnmovies = movies_fd.drop(columns=['overview','genres','keywords','cast','crew'])


# In[56]:


fnmovies['tags'] = fnmovies['tags'].apply(lambda x: " ".join(x))
fnmovies.head()


# In[57]:


import nltk


# In[58]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[59]:


def stem(text):
    w= [ ]
    
    for i in text.split():
        w.append(ps.stem(i))
    return " ".join(w)


# In[60]:


ps.stem("moving")


# In[61]:


fnmovies['tags'] = fnmovies['tags'].apply(stem)


# In[62]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=50,stop_words='english')


# In[63]:


vector = cv.fit_transform(fnmovies['tags'])


# In[64]:


vector


# In[65]:


vector = cv.fit_transform(fnmovies['tags']).toarray()


# In[66]:


vector[1]


# In[67]:


len(cv.get_feature_names())


# In[68]:


cv.get_feature_names()


# In[69]:


vector.shape


# In[70]:


from sklearn.metrics.pairwise import cosine_similarity


# In[71]:


sim= cosine_similarity(vector)


# In[73]:


print(sim)


# In[77]:


fnmovies[fnmovies['title'] == 'The Lego Movie'].index[0]


# In[78]:


def recommend_movie_system(movie):
    index = fnmovies[fnmovies['title'] == movie].index[0]
    distances = sorted(list(enumerate(sim[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(fnmovies.iloc[i[0]].title)


# In[98]:


recommend_movie_system("GoldenEye")


# In[97]:


fnmovies["title"].head(12)


# In[107]:


fnmovies.shape


# In[103]:


shortfnmovies=fnmovies


# In[128]:


sm=shortfnmovies.loc[shortfnmovies.index<1000]


# In[129]:


from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer(max_features=1000,stop_words='english')


# In[130]:


sm.shape


# In[131]:


vector11 = cv1.fit_transform(sm['tags']).toarray()


# In[132]:


sim11= cosine_similarity(vector11)


# In[133]:


import pickle


# In[134]:


pickle.dump(sm,open('movie_smalllist.pkl','wb'))
pickle.dump(sim11,open('similarity11.pkl','wb'))


# In[135]:


def recommend_movie_system1(movie):
    index = sm[sm['title'] == movie].index[0]
    distances = sorted(list(enumerate(sim11[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(sm.iloc[i[0]].title)


# In[136]:


recommend_movie_system1("Jumanji")


# In[ ]:




