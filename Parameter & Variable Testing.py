#!/usr/bin/env python
# coding: utf-8

# In[42]:


from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.classifiers.fundamentals import Sector 
from quantopian.research import run_pipeline
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.filters import Q1500US, Q500US
import pandas as pd
import numpy as np
import random as random
import itertools
from itertools import combinations
import sklearn
from sklearn.cluster import KMeans
import quantopian.pipeline.factors as pfactors
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


def make_pipeline():
    base_universe = Q500US()
    ROA = Fundamentals.roa.latest
    ROE = Fundamentals.roe.latest
    EV_EBITDA = Fundamentals.ev_to_ebitda.latest
    Enterprise_value = Fundamentals.enterprise_value.latest
    PE = Fundamentals.pe_ratio.latest
    PB = Fundamentals.pb_ratio.latest
    PS = Fundamentals.ps_ratio.latest
    
    
    
    rsi = pfactors.RSI(inputs=[USEquityPricing.close], window_length=14)
    
    #pipeline variables....winsorize to omit outliers, and zscore to normalize
    
    return Pipeline(
      columns={
            'RSI' : rsi.zscore().winsorize(min_percentile =0.05, max_percentile= 0.95),
            #'PE': PE.zscore().winsorize(min_percentile = 0.05, max_percentile = 0.95),
            'Enterprise value' : Enterprise_value.zscore().winsorize(0.05, 0.95),
            #'ROA' : ROA.zscore().winsorize(0.05, 0.95),
            'ROE': ROE.zscore().winsorize(0.05, 0.95),
            'PB': PB.zscore().winsorize(0.05, 0.95),
            'PS': PS.zscore().winsorize(0.05, 0.95),
            #'EV_EBITDA':EV_EBITDA.zscore().winsorize(0.05, 0.95)
            }
                    )

result = run_pipeline(make_pipeline(),'2015-05-05', '2015-05-05')

train_set_dropna = result.dropna(axis=0)
train_set = train_set_dropna.loc[:,~train_set_dropna.columns.str.contains('RSI')]
    #train_set = train_set.iloc[:, 1:]
kmeans = KMeans(n_clusters=5).fit(train_set)
labels = kmeans.labels_
#print labels
values = np.array(labels)

values = values.reshape((-1, 1))
train_set['values'] = values

train_set['RSI'] = train_set_dropna['RSI']

# only trade with sizeable cluster information, 25 stocks per cluster

train_set = train_set.groupby('values').filter(lambda x: len(x) > 25)


#long the cluster with bottom 10% RSI value and short the cluster with top 10% RSI value

train_set_long = train_set.loc[train_set['RSI'] < train_set.groupby('values').RSI.transform(lambda x: x.quantile(.1))]
train_set_short = train_set.loc[train_set['RSI'] > train_set.groupby('values').RSI.transform(lambda x: x.quantile(.9))]
longs = train_set_long.index.tolist()
#print(longs)
shorts = train_set_short.index.tolist()

print(train_set_short)
print (train_set_long)

#Plot 2 of the 4 variables to observe intra and inter cluster distances

plt.scatter(train_set_short['Enterprise value'], train_set_short['ROE'] ,c=train_set_short['values'], cmap = 'rainbow')
#plt.show()
#print (shorts)



# In[20]:


train_set_short.groupby('values')['RSI'].mean()


# In[44]:


train_set_long.groupby('values')['RSI'].mean()


# In[46]:


train_set_long.groupby('values')['Enterprise value','PE'].describe()


# In[29]:


from sklearn.preprocessing import MinMaxScaler


mms = MinMaxScaler()
mms.fit(train_set_dropna.loc[:,~train_set_dropna.columns.str.contains('RSI')])
data_transformed = mms.transform(train_set_dropna.loc[:,~train_set_dropna.columns.str.contains('RSI')])

#Find elbow point to minimize SSD
Sum_of_squared_distances = []
K = range(1,50)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[44]:



#Silhouette Analysis

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from kmeansplots import kmeans_plot, silhouette_plot


X = train_set_dropna.loc[:,~train_set_dropna.columns.str.contains('RSI')]


#Find silhouette score for each cluster number and for different selected variables
# silhouette score provides information on inter and intracluster distances

for n_clusters in range(2, 10):
    clusterer = KMeans(n_clusters=n_clusters)
    y = clusterer.fit_predict(X)


    print(n_clusters, round(silhouette_score(X, y), 2))


# In[ ]:





# In[ ]:




