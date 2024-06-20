from statistics import mean
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("tabel data pollutan")
data = pd.read_excel("Kualitas Udara.xlsx")

print(data.head(10))
print("=======================================================")

print("atribut yang dipake")
prediksi = data.loc[:, ['pm10', 'so2', 'co', 'o3','no2']]
print(prediksi.head())
print("=======================================================")

# missing value
print("missing value")
print("=======================================================")
print(prediksi.isna().sum())
print("=======================================================")

# Penanganan Data Missing Value
print("Penanganan Missing Value")
prediksi_cleaned = prediksi.dropna()
print("Data tanpa missing value:")
print(prediksi_cleaned)

# outlier
print("outlier")
print("=======================================================")

outliers = []

def detect_outlier(data):
    outliers = []
    threshold = 3
    mean_value = data.mean()
    std_dev = data.std()
    
    for x in data:
        z_score = (x - mean_value) / std_dev
        if np.abs(z_score) > threshold:
            outliers.append(x)
    return outliers

# Mengecek outlier untuk setiap kolom
for col in prediksi_cleaned.columns:
    outliers = detect_outlier(prediksi_cleaned[col])
    print("outlier kolom", col, ":", outliers)
    print("total outlier", col, ":", len(outliers))
    print()

# Penanganan Outlier untuk Mengganti outlier dengan nilai rata-rata (mean)
for col in prediksi_cleaned.columns:
    outliers = detect_outlier(prediksi_cleaned[col])
    rata = mean(prediksi_cleaned[col])
    prediksi_cleaned[col] = prediksi_cleaned[col].replace(outliers, rata)

# Menampilkan data setelah penanganan outlier
print("Data setelah penanganan outlier:")
print(prediksi_cleaned)


# Data yang sudah dibersihkan dan diolah
X = prediksi_cleaned  # Fitur
y = data['categori']  # Target

# split menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  model KNN
knn_model = KNeighborsClassifier(n_neighbors=3) 

# Latih model KNN menggunakan data latih30
knn_model.fit(X_train, y_train)

# Fungsi untuk menerima input dari pengguna dan melakukan prediksi
def predict_knn():
    print("Masukkan nilai untuk setiap fitur:")
    pm10 = float(input("PM10: "))
    so2 = float(input("SO2: "))
    co = float(input("CO: "))
    o3 = float(input("O3: "))
    no2 = float(input("NO2: "))

    # Buat array fitur dari input pengguna
    user_input = [[pm10, so2, co, o3, no2]]

    # Lakukan prediksi menggunakan model KNN
    prediction = knn_model.predict(user_input)

    # Tampilkan hasil prediksi
    print("Prediksi kualitas udara:", prediction[0])

# Panggil fungsi untuk meminta input dan melakukan prediksi
predict_knn()

# Hitung akurasi model
accuracy = accuracy_score(y_test, y_train)
print("Akurasi model KNN:", accuracy)
