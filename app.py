from flask import Flask, render_template, \
    request, redirect, url_for, session, flash, get_flashed_messages
import pandas as pd
from sklearn.cluster import KMeans
from io import StringIO

application = Flask(__name__)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_scatter_plot(df):
    # Membuat scatter plot 2D (sesuaikan dengan jumlah klaster yang dihasilkan)
    plt.scatter(df['nama.barang'], df['kuantum'], c=df['klaster'], cmap='viridis')
    plt.xlabel('Nama Barang')
    plt.ylabel('Total Kuantum')
    plt.title('Hasil Klasterisasi')

    # Menyimpan plot sebagai gambar
    plot_path = 'static/image/cluster_plot.png'
    plt.savefig(plot_path)
    plt.close()

    return plot_path

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/', methods=['POST'])
def process_data():
    if 'file_name' not in request.files:
        print('gagal')
        return redirect(request.url)

    file = request.files['file_name']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Baca file CSV
        content = file.stream.read().decode("utf-8")
        try:
            csv_data = StringIO(content)
            df = pd.read_csv(csv_data)
            print('Original CSV data:', df)

            # Praproses data: Jumlahkan 'kuantum' berdasarkan 'nama.barang'
            df_sum = df.groupby('nama.barang')['kuantum'].sum().reset_index()
            print('Summed Data:', df_sum)

            # Proses data menggunakan algoritma K-Means
            features = df_sum[['kuantum']]
            kmeans = KMeans(n_clusters=2)
            df_sum['klaster'] = kmeans.fit_predict(features)

            plot_path = create_scatter_plot(df_sum)

            result_html = df_sum.to_html()
            return render_template('result.html', result=result_html, plot_path=plot_path)

        except Exception as e:
            print(f"Error processing CSV: {e}")
            return redirect(request.url)
    return redirect(request.url)

if __name__ == '__main__':
    application.run(debug=True)