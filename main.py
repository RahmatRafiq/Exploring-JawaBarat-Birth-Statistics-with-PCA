import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Membaca file csv menggunakan pandas
data = pd.read_csv("datakelahirann.csv")

# Check for null values
print(data.isnull().sum())

# Menampilkan 5 baris pertama dari data
print(data.head())

# Mengubah kolom 'status_kelahiran' menjadi data numerik
le = LabelEncoder()
# data['status_kelahiran'] = le.fit_transform(data['status_kelahiran'])
data['jenis_kelamin'] = le.fit_transform(data['jenis_kelamin'])
data['nama_provinsi'] = le.fit_transform(data['nama_provinsi'])
data['nama_kabupaten_kota'] = le.fit_transform(data['nama_kabupaten_kota'])
data['satuan'] = le.fit_transform(data['satuan'])

# Menentukan jumlah fitur yang ingin digunakan
pca = PCA(n_components=3)

# Menyesuaikan PCA dengan data
pca.fit(data)

# Transformasi data dengan fitur yang dipilih
data_pca = pca.transform(data)

# Memisahkan data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(data_pca,
                                                    data['status_kelahiran'],
                                                    test_size=0.2)

# Membuat model logistik regresi
model = LogisticRegression()

# Menyesuaikan model dengan data latih
model.fit(X_train, y_train)

# Membuat prediksi dengan data uji
y_pred = model.predict(X_test)

# Menghitung akurasi model
acc = accuracy_score(y_test, y_pred)
print("Akurasi: ", acc)

# Membuat matriks konfusi
conf_mat = confusion_matrix(y_test, y_pred)
print("Matriks Konfusi: \n", conf_mat)

# Membuat tabel klasifikasi
class_rep = classification_report(y_test, y_pred)
print("Tabel Klasifikasi: \n", class_rep)
