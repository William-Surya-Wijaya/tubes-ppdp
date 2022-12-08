import numpy as np
import pandas as pd

# Baca dataset iris.csv
dt_iris = pd.read_csv('iris.csv', delimiter = ',')

#Ambil & print rekord bag atas
print(dt_iris.head())

# Pilih fitur-fitur (4 atribut) yang akan dikelompokkan (tidak 
# mengikutsertakan atribut spesies)
dt_iris_4kol = dt_iris[['sepal_length', 'sepal_width','petal_length', 'petal_width']]
dt_iris_4kol.head()
#dt_iris_4kol.values

# Buat numpy array X dari dataframe dt_iris_4kol
X = np.array(dt_iris_4kol.values)

#Import library k-Means
from sklearn.cluster import KMeans

# Manual kelas k-Means:
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

#Lakukan clustering (fit) terhadap X dgn jumlah cluster = 3
kmeans_model = KMeans(n_clusters=3, random_state=0).fit(X)

# Eksperimen:
# Ubahlah nilai n_clusters = 4, 5, 6, .... amati hasil-hasil di bawah 

# Simpan hasil clustering berupa nomor klaster tiap objek/rekord di
# varialbel klaster_objek
klaster_objek = kmeans_model.labels_

# Simpan hasil clustering berupa centroid (titik pusat) tiap kelompok
# di variabel centroids
centroids = kmeans_model.cluster_centers_


# Import library
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

#clustering kmeans dengan pemeriksaan kualitas hasil cluster menggunakan elbow method dan koefisien silhouette
#clustering kmeans dilakukan dengan menjalankan algoritmanya menggunakan nilai k 2 hingga 15
intertia = []
silhouette_coefficients = []
K = range(2,10)
for k in K:
    kmeans_model = KMeans(n_clusters=k, random_state=0).fit(X)
    intertia.append(kmeans_model.inertia_)
    score = silhouette_score(X, kmeans_model.labels_,  metric='euclidean')
    silhouette_coefficients.append(score)

#visualisasi hasil elbow method    
plt.plot(K, intertia, marker= "o")
plt.xlabel('k')
plt.xticks(np.arange(2, 10))
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

#visualisasi hasil perhitungan koefisien Silhouette   
plt.plot(K, silhouette_coefficients, marker= "o")
plt.xlabel('k')
plt.xticks(np.arange(2, 10))
plt.ylabel("Silhouette Coefficient")
plt.title("AVG Silhouette Coefficient")
plt.show()


# Model terbaik clustering (berupa centroids) dapat disimpan dan dimanfaatkan
# untuk mencari kelompok dari objek-objek baru 

#Lakukan clustering (fit) terhadap X dgn jumlah cluster = 3
kmeans_model = KMeans(n_clusters=3, random_state=0).fit(X)

#Baca rekord-rekord bunga Irish yg belum dikeathui cluster-nya
dt_baru_iris = pd.read_csv('iris_new_data.csv', delimiter = ',')

#Ambil & print rekord bag atas
print(dt_baru_iris.head())

# Buat numpy array X_new
X_new = np.array(dt_baru_iris.values)

#Cari (predict) cluster dari rekord-rekord bunga Iris yg baru
kmeans_model.predict(X_new)

#Simpan model clustering (agar dapat digunakan lagi lain kali)
import pickle
# pickle.dump(model, open(filename, 'wb')) #Saving the model
pickle.dump(kmeans_model, open('kmeans_model_irish', 'wb'))

# Baca model dan gunakan kembali untuk memprediksi cluster objek baru
loaded_model = pickle.load(open('kmeans_model_irish', 'rb'))

#Cari (predict) cluster dari rekord-rekord bunga Iris yg baru
loaded_model.predict(X_new)

# Jika jumlah atribut banyak dan ingin dicari fitur yg sesuai untuk clustering
# Salah satu cara: membandingkan variance antar atribut
#
dt_iris_4kol.cov()

# Terlihat atribut sepal_width memiliki nilai kovariance kecil dibandingkan 
# yang lain, di sini sepal_width akan diabaikan (tidak digunakan)

dt_iris_3kol = dt_iris[['sepal_length', 'petal_length', 'petal_width']]

X = np.array(dt_iris_3kol.values)

# Pencarian jumlah kelompok terbaik
intertia = []
silhouette_coefficients = []
K = range(2,10)
for k in K:
    kmeans_model = KMeans(n_clusters=k, random_state=0).fit(X)
    intertia.append(kmeans_model.inertia_)
    score = silhouette_score(X, kmeans_model.labels_,  metric='euclidean')
    silhouette_coefficients.append(score)

#visualisasi hasil elbow method    
plt.plot(K, intertia, marker= "o")
plt.xlabel('k')
plt.xticks(np.arange(2, 10))
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

#visualisasi hasil perhitungan koefisien Silhouette   
# Pada hasil plot terlihat bahwa koefisien pada k=3 lebih baik dibanding
# menggunakan seluruh (4) atribut
plt.plot(K, silhouette_coefficients, marker= "o")
plt.xlabel('k')
plt.xticks(np.arange(2, 10))
plt.ylabel("Silhouette Coefficient")
plt.title("AVG Silhouette Coefficient")
plt.show()

# Kesimpulan: Sebaiknya pengelompokan dilakukan dgn 3 atribut saja
# Catatan: Jika model akan disimpan dan nantinya digunakan untuk memprediksi
# kelompok dari rekord-rekord baru, maka dataset baru harus memiliki 3 atribut
# (karena model dibuat dg 3 atribut)