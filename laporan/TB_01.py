#import library
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;

import time;

import pickle;

from sklearn.cluster import KMeans;
from sklearn.metrics import silhouette_score;

from sklearn.model_selection import train_test_split;

from sklearn.cluster import AgglomerativeClustering;
import scipy.cluster.hierarchy as sch;

#load data
mall = pd.read_csv("https://raw.githubusercontent.com/shrk-sh-ioai/tubes-ppdp/main/mall-customer-dt/mall-customers.csv", sep=",", encoding='cp1252');

#checking data - NaN value
dfCheck = mall[mall.isna().any(axis=1)];
print("Data with NaN values :"); print(dfCheck);

#cheking data - Minimum value
print("Minimum values :"); print(mall.min());

#checking data - Maximum value
print("Maximum values :"); print(mall.max());

#preparing data

# spending score classification ----------
spend_classes = ["1","2","3","4","5","6","7","8","9","10"];

spend_items = pd.IntervalIndex.from_tuples([(0, 11), (11, 21), (21, 31), (31, 41), (41, 51), (51,61), (61,71), (71,81), (81,91), (91,101)], closed='left');

mall['Spending_Class'] = np.array(spend_classes)[
    pd.cut(mall["Spending Score (1-100)"],
    bins = spend_items).cat.codes
];

# gender classification ----------
gender_classes = ['0','1'];

gender_items = [
    (mall['Gender'] == "Female"),
    (mall['Gender'] == "Male"),
];

mall['Gender_Class'] = np.select(gender_items, gender_classes);

# age classification ----------
# 15-24 tahun: Kelompok usia muda
# 25-34 tahun: Kelompok usia pekerja awal
# 35-44 tahun: Kelompok usia paruh baya
# 45-54 tahun: Kelompok usia pra-pensiun
# 55-64 tahun: Kelompok usia pensiun
# 65 tahun ke atas: Kelompok usia lanjut

age_classes = ['0','1','2','3','4','5'];

age_items = [
    (mall['Age'] >= 15) & (mall['Age'] <= 24),
    (mall['Age'] >= 25) & (mall['Age'] <= 34),
    (mall['Age'] >= 35) & (mall['Age'] <= 44),
    (mall['Age'] >= 45) & (mall['Age'] <= 54),
    (mall['Age'] >= 55) & (mall['Age'] <= 64),
    (mall['Age'] >= 65),
];

mall['Age_Class'] = np.select(age_items, age_classes);

print(mall);

# searching best-k ----------
mall_x = mall[[
    'Gender_Class',
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]];

mall_x_np = np.array(mall_x.values);

print(mall_x);

# kmeans model train ---------
intertia = [];
silhouette_coefficients = [];

k_range = range(2,10);
for k in k_range :
    
    kmeans_model = KMeans(
        n_clusters = k,
        random_state=0
    ).fit(mall_x_np);
    
    intertia.append(kmeans_model.inertia_);
    
    score = silhouette_score(
        mall_x_np,
        kmeans_model.labels_,
        metric='euclidean'
    );
    
    silhouette_coefficients.append(score);

# elbow Method Result Visualization
plt.plot(k_range, intertia, marker= "o");

plt.xlabel('k'); plt.xticks(np.arange(2, 10));
plt.ylabel('Inertia'); plt.title('Elbow Method');

plt.show();

# silhouette Coefficient Result Visualization
plt.plot(k_range, silhouette_coefficients, marker= "o");

plt.xlabel('k'); plt.xticks(np.arange(2, 10));
plt.ylabel("Silhouette Coefficient"); plt.title("AVG Silhouette Coefficient");

plt.show();

# covariant
print(mall.cov());

# model-load
start = time.time();
kmeans_model = KMeans(n_clusters=6, random_state=0).fit(mall_x_np);

mall = mall[['Gender_Class', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']];
mall['class'] = kmeans_model.predict(np.array(mall.values));

score = silhouette_score(mall_x_np, kmeans_model.labels_,  metric='euclidean')
print(score)

print('Time : ', time.time()-start);

# save kmeans model
pickle.dump(kmeans_model, open('kmeans_model_mall','wb'));

# Object Cluster Variable
object_cluster = kmeans_model.labels_;

print(object_cluster);

# Object Centroids
centroids = kmeans_model.cluster_centers_;

print(centroids);

 analisis clustering

mall0 = mall.loc[(mall['class'] == 0)];
print('Cluster 1 : \nRata Rata Umur : ', mall0['Age'].mean(),'\nRata Rata Annual Income (k$) :', mall0['Annual Income (k$)'].mean(), '\nRata Rata Spending Score (1-100) :', mall0['Spending Score (1-100)'].mean(), "\n");

mall1 = mall.loc[(mall['class'] == 1)];
print('Cluster 2 : \nRata Rata Umur : ', mall1['Age'].mean(),'\nRata Rata Annual Income (k$) :', mall1['Annual Income (k$)'].mean(), '\nRata Rata Spending Score (1-100) :', mall1['Spending Score (1-100)'].mean(), "\n");

mall2 = mall.loc[(mall['class'] == 2)];
print('Cluster 3 : \nRata Rata Umur : ', mall2['Age'].mean(),'\nRata Rata Annual Income (k$) :', mall2['Annual Income (k$)'].mean(), '\nRata Rata Spending Score (1-100) :', mall2['Spending Score (1-100)'].mean(), "\n");

mall3 = mall.loc[(mall['class'] == 3)];
print('Cluster 4 : \nRata Rata Umur : ', mall3['Age'].mean(),'\nRata Rata Annual Income (k$) :', mall3['Annual Income (k$)'].mean(), '\nRata Rata Spending Score (1-100) :', mall3['Spending Score (1-100)'].mean(), "\n");

mall4 = mall.loc[(mall['class'] == 4)];
print('Cluster 5 : \nRata Rata Umur : ', mall4['Age'].mean(),'\nRata Rata Annual Income (k$) :', mall4['Annual Income (k$)'].mean(), '\nRata Rata Spending Score (1-100) :', mall4['Spending Score (1-100)'].mean(), "\n");

mall5 = mall.loc[(mall['class'] == 5)];
print('Cluster 6 : \nRata Rata Umur : ', mall5['Age'].mean(),'\nRata Rata Annual Income (k$) :', mall5['Annual Income (k$)'].mean(), '\nRata Rata Spending Score (1-100) :', mall5['Spending Score (1-100)'].mean(), "\n");

# penerapan model

# load kmeans model
loaded_model = pickle.load(open('kmeans_model_mall','rb'));

#load data
mall_test = pd.read_csv("https://raw.githubusercontent.com/shrk-sh-ioai/tubes-ppdp/main/mall-customer-dt/mall-customers.csv", sep=",", encoding='cp1252');

mall_test = mall_test[['Gender','Age','Annual Income (k$)','Spending Score (1-100)']];

# gender classification ----------
gender_classes = ['0','1'];

gender_items = [
    (mall_test['Gender'] == "Female"),
    (mall_test['Gender'] == "Male"),
];

mall_test['Gender_Class'] = np.select(gender_items, gender_classes);
mall_test = mall_test[['Gender_Class','Age','Annual Income (k$)','Spending Score (1-100)']];

mall_test['class'] = loaded_model.predict(np.array(mall_test.values));

print(mall_test);

plt.figure(figsize=(10, 7));
plt.title("Dendrogram Mall - Single Linkage");
dend = sch.dendrogram(sch.linkage(mall_x_np, method='single'));
plt.show();

start = time.time();
agglo_model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single');
agglo_model.fit_predict(mall_x_np);

labels = agglo_model.labels_;
print(labels);
print('Time : ', time.time()-start);

score = silhouette_score(mall_x_np, agglo_model.labels_,  metric='euclidean')
print(score)

agglo_model.n_clusters

labels = agglo_model.labels_
df_labels = pd.DataFrame({'cls': labels})

df = mall_x.join(df_labels)

df_pola = df.groupby(['cls']).describe()
print(df_pola)

plt.figure(figsize=(10, 7));
plt.title("Dendrogram Mall - Complete Linkage");
dend = sch.dendrogram(sch.linkage(mall_x_np, method='complete'));
plt.show();

start = time.time();
agglo_model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete');
agglo_model.fit_predict(mall_x_np);
score = silhouette_score(mall_x_np, agglo_model.labels_,  metric='euclidean');
print(score);

print('Time : ', time.time()-start);

agglo_model.n_clusters

labels = agglo_model.labels_
df_labels = pd.DataFrame({'cls': labels})

df = mall_x.join(df_labels)

df_pola = df.groupby(['cls']).describe()
print(df_pola);

plt.figure(figsize=(10, 7));
plt.title("Dendrogram Mall - Centroid Linkage");
dend = sch.dendrogram(sch.linkage(mall_x_np, method='centroid'));
plt.show();

start = time.time();

agglo_model = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward');
agglo_model.fit_predict(mall_x_np);
score = silhouette_score(mall_x_np, agglo_model.labels_,  metric='euclidean');
print(score);

print('Time : ', time.time()-start);

agglo_model.n_clusters

labels = agglo_model.labels_
df_labels = pd.DataFrame({'cls': labels})

df = mall_x.join(df_labels)

df_pola = df.groupby(['cls']).describe()
print(df_pola);