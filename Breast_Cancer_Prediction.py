#!/usr/bin/env python
# coding: utf-8

# Classification and Clustering of Breast Cancer for prediction of diagnosis

# #Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import plotly.offline as py
from matplotlib.lines import Line2D
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
plt.style.use('ggplot')


# In[2]:


# Importing dataset
df=pd.read_csv('F:/data.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape 


# In[6]:


#Information about the dataset
df.info()


# In[7]:


#Description about the dataset
df.describe()


# In[8]:


# dropping unncessary columns
df.drop("Unnamed: 32",axis=1,inplace=True)


# In[9]:


#checking missing values
df.isnull().sum()


# In[10]:


#Checking duplicate values
df.duplicated().sum()


# Covariance

# In[11]:


df.cov()


# Correlation

# In[12]:


df.corr()


# Visual Representation ofCorrelation

# <b>Correlation Meterics</b>

# In[13]:


fig, ax = plt.subplots(figsize=(20,15)) 
sns.heatmap(df.corr(),ax=ax,annot=True,linewidth=.5)


# In[14]:


#Univariate Analysis


# In[15]:


fig= px.histogram(df, x='diagnosis',color='diagnosis',height=500,width=500, barmode='relative')
fig.show()


# Inference:Benign is more diagnosed than Malignant

# In[16]:



import plotly.express as px

fig1 = px.bar(df, x='diagnosis', y='texture_worst', height=500,width=500,title='Stacked Bar Chart - Hover on individual items')
fig1.show()


# Inference: the texture_worst for Benign is higher than Malignant 

# In[17]:


#PieChart


# In[18]:



fig = px.pie(df, values='compactness_worst', names='diagnosis', title='Relation')
fig.show()


# Inference:The percentage of Compactness_worst is higher for Malignant than Benign

# In[19]:


#Multivariate Analysis


# In[20]:


#positive Correlation
fig,ax=plt.subplots(2,2,figsize=(10,10))
fig.tight_layout(pad=5.0)
sns.scatterplot(x='perimeter_mean',y='radius_worst',data=df,hue='diagnosis',ax=ax[0][0])
sns.scatterplot(x='area_mean',y='radius_worst',data=df,hue='diagnosis',ax=ax[1][0])
sns.scatterplot(x='texture_mean',y='texture_worst',data=df,hue='diagnosis',ax=ax[0][1])
sns.scatterplot(x='area_worst',y='radius_worst',data=df,hue='diagnosis',ax=ax[1][1])
ax[0,0].set_title("Fig (1)")
ax[0,1].set_title("Fig (2)")
ax[1,0].set_title("Fig (3)")
ax[1,1].set_title("Fig (4)")
plt.show()


# Inference: Fig(1) : The perimeter_mean and radius_mean is higher for Malignant<br />
#            Fig(2) : The texture_mean and texture_worst is lower for Benign than malignant <br />
#            Fig(3) : The radius_worst and area_mean is heigher for Malignant and Benign<br />
#            Fig(4) : The radius_worst and area_worst is lower for benign than malignant

# In[21]:


#Negative Correlation


# In[22]:


fig,ax=plt.subplots(2,2,figsize=(10,10))
fig.tight_layout(pad=5.0)
sns.scatterplot(x='area_mean',y='fractal_dimension_mean',data=df,hue='diagnosis',ax=ax[0][0])
sns.scatterplot(x='radius_mean',y='smoothness_se',data=df,hue='diagnosis',ax=ax[1][0])
sns.scatterplot(x='smoothness_se',y='perimeter_mean',data=df,hue='diagnosis',ax=ax[0][1])
sns.scatterplot(x='area_mean',y='smoothness_se',data=df,hue='diagnosis',ax=ax[1][1])
ax[0,0].set_title("Fig (1)")
ax[0,1].set_title("Fig (2)")
ax[1,0].set_title("Fig (3)")
ax[1,1].set_title("Fig (4)")


# Plots

# In[23]:


#Implot


# In[24]:


sns.lmplot(x="radius_mean", y="texture_mean",hue="diagnosis", data=df,
           markers=["o", "x"], palette="Set1");


# Inference:

# In[25]:


#Pair plot


# In[26]:


import seaborn as sns
sns.set_theme(style="ticks")
sns.pairplot(data=df[['diagnosis' ,'radius_mean','texture_mean','perimeter_mean','area_mean']], palette="Set1",hue="diagnosis", height=3,diag_kind="hist")
plt.show()
#sns.pairplot(data=df[['diagnosis' ,'radius_mean','texture_mean','perimeter_mean','area_mean',hue="diagnosis")]],hue="diagnosis", height=3, diag_kind="hist")


# In[27]:


#boxen plot


# In[28]:


sns.set_theme(style="whitegrid")
sns.boxenplot(x="diagnosis", y="perimeter_mean",
              color="r",
              scale="linear", data=df)


# In[29]:


#count plot
for column in df.select_dtypes(include='object'):
    if df[column].nunique() < 10:
        sns.countplot(y=df.diagnosis, data=df)
        plt.show()


# <b style="font_size : 20px">Distribution Plot</b>

# In[30]:


import plotly.figure_factory as ff


hist_data = [df['radius_mean']]
group_labels = ['breast_cancer'] # name of the dataset

fig = ff.create_distplot(hist_data, group_labels)
fig.show()


# Scatter plot using plotly

# In[31]:


fig = px.scatter(df,x='radius_mean',y='perimeter_mean',color='diagnosis',height=500,width=700,size_max=60)
fig.show()


# In[32]:


fig2=px.scatter(df,x='texture_worst',y= 'symmetry_worst',color='diagnosis',height=500,width=700,size_max=60)
fig2.show()


# <b>Data preprocessing</b>

# In[33]:


X=df.iloc[:,2:].values
y=df.iloc[:,1].values


# In[34]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
labelencode = LabelEncoder()
y=labelencode.fit_transform(y)


# In[35]:


#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)


# In[36]:


#applying standard scaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[37]:


# pip install keras


# In[38]:


# # pip install tensorflow
# pip install pywrap_tensorflow


# In[39]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[40]:


classifier = Sequential()


# In[41]:


#adding the input and first hidden layer

classifier.add(Dense(16, activation='relu', kernel_initializer='glorot_uniform',input_dim=30))

#adding second layer
classifier.add(Dense(6, activation='relu', kernel_initializer='glorot_uniform'))

#adding the output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


# In[42]:


classifier.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[43]:


classifier.fit(X_train,y_train,batch_size=100,epochs=150)


# In[44]:


y_pred=classifier.predict(X_test)
y_pred = (y_pred>0.5)


# In[45]:


from sklearn.metrics import accuracy_score


print(f"The test accuracy is very high i.e.{accuracy_score(y_test,y_pred)}")


# In[46]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[47]:


print('CLUSTERING ON raidus_mean  AND texture_mean')
X = df[['radius_mean','texture_mean']]

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X.loc[:, 'radius_mean'], X.loc[:, 'texture_mean'], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.show()


# In[48]:


kmeans = KMeans(n_clusters=3, random_state=0)
df['cluster'] = kmeans.fit_predict(df[['radius_mean', 'texture_mean']])
# get centroids
centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]
## add to df
df['cen_x'] = df.cluster.map({0:cen_x[0], 1:cen_x[1], 2:cen_x[2]})
df['cen_y'] = df.cluster.map({0:cen_y[0], 1:cen_y[1], 2:cen_y[2]})
# define and map colors
colors = ['#DF2020', '#81DF20', '#2095DF']
df['c'] = df.cluster.map({0:colors[0], 1:colors[1], 2:colors[2]})


# In[51]:


fig, ax = plt.subplots(1, figsize=(8,8))
# plot data
plt.scatter(df.radius_mean, df.texture_mean, c=df.c, alpha = 0.6, s=10)
# plot centroids
plt.scatter(cen_x, cen_y, marker='^', c=colors, s=70)
# plot lines
for idx, val in df.iterrows():
    x = [val.radius_mean, val.cen_x,]
    y = [val.radius_worst, val.cen_y]
    plt.plot(x, y, c=val.c, alpha=0.2)
# legend
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i+1), 
                   markerfacecolor=mcolor, markersize=5) for i, mcolor in enumerate(colors)]
legend_elements.extend([Line2D([0], [0], marker='^', color='w', label='Centroid - C{}'.format(i+1), 
            markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(colors)])
legend_elements.extend(cent_leg)
plt.legend(handles=legend_elements, loc='upper right', ncol=2)
# x and y limits
plt.xlim(0,200)
plt.ylim(0,200)
# title and labels
plt.title('Pokemon Stats\n', loc='left', fontsize=22)
plt.xlabel('Attack')
plt.ylabel('Defense')
#on hold


# In[ ]:




