from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from statistics import mean

# from prediksi2 import predict_knn

app = Flask(__name__)
# load data
data = pd.read_excel("Kualitas Udara.xlsx")

# atribut yg dipake
prediksi = data.loc[:, ['pm10', 'so2', 'co', 'o3','no2']]

# missing value
prediksi.isna().sum()

# handling 
prediksi_cleaned = prediksi.dropna()

# oulier
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

# cek tiap kolom
for col in prediksi_cleaned.columns:
    outliers = detect_outlier(prediksi_cleaned[col])
   
# handling outlier dgn mean
for col in prediksi_cleaned.columns:
    outliers = detect_outlier(prediksi_cleaned[col])
    rata = mean(prediksi_cleaned[col])
    prediksi_cleaned[col] = prediksi_cleaned[col].replace(outliers, rata)

# Data yang sudah dibersihkan dan diolah
X = prediksi_cleaned  # Fitur
y = data['categori']  # Target

# split menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  model KNN
knn_model = KNeighborsClassifier(n_neighbors=3) 

# Latih model KNN menggunakan data latih30
knn_model.fit(X_train, y_train)


@app.route("/")
def index():
    return render_template('beranda.html')

@app.route("/prediksi")
def prediksi():
    return render_template('formprediksi.html')

@app.route("/prediksi2")
def prediksi2():
    return render_template('prediksi.html')

@app.route("/output")
def output():
    return render_template('output.html')

# @app.route('/static/<path:path>')
# def send_static(path):
#     return send_from_directory('static', path)

# @app.route("/")
# def index():
#     if request.method == 'POST':
#         pm10 = float(request.form['pm10'])
#         so2 = float(request.form['so2'])
#         co = float(request.form['co'])
#         o3 = float(request.form['o3'])
#         no2 = float(request.form['no2'])
#         prediction = predict_knn(pm10, so2, co, o3, no2)
#         return render_template('prediksi.html', prediction=prediction)
#     return render_template('index.html')

# @app.route("/prediksi")
# def prediksi():
#     # pm10 = float(request.form['pm10'])
#     # so2 = float(request.form['so2'])
#     # co = float(request.form['co'])
#     # o3 = float(request.form['o3'])
#     # no2 = float(request.form['no2'])
#     # prediction = predict_knn(pm10, so2, co, o3, no2)
#     # return render_template('prediksi.html')

if __name__ == '__main__':
     app.run(debug=True)