from flask import Flask, request, jsonify
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import tempfile
import os
import threading

app = Flask(__name__)

# Route untuk menampilkan pesan Halo dari server playroom saat mengakses root
@app.route('/')
def hello():
    return jsonify({'message': 'Halo dari server playroom'})

def compare_images_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

    similarity_index, _ = ssim(img1_gray, img2_gray, full=True)
    return similarity_index

def process_comparison(image1, image2, result_callback):
    try:
        # Menggunakan tempfile untuk menyimpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file1:
            img1_path = temp_file1.name
            image1.save(img1_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file2:
            img2_path = temp_file2.name
            image2.save(img2_path)

        # Menghitung SSIM
        ssim_score = compare_images_ssim(img1_path, img2_path)

        # Menentukan hasil berdasarkan SSIM
        result = "benar" if ssim_score > 0.5 else "salah"

        # Menghapus file sementara setelah digunakan
        os.remove(img1_path)
        os.remove(img2_path)

        # Menyampaikan hasil melalui callback
        result_callback(ssim_score, result)
    
    except Exception as e:
        result_callback(error=str(e))

@app.route('/compare', methods=['POST'])
def compare():
    def result_callback(ssim_score=None, result=None, error=None):
        if error:
            return jsonify({'error': error}), 500
        return jsonify({
            'ssim_score': round(ssim_score * 100, 2),
            'result': result
        })

    # Mengambil file gambar dari request
    image1 = request.files['image1']
    image2 = request.files['image2']

    # Jalankan perbandingan gambar dalam thread terpisah
    thread = threading.Thread(target=process_comparison, args=(image1, image2, result_callback))
    thread.start()

    # Mengembalikan respons awal agar koneksi tidak tertutup
    return jsonify({'message': 'Proses perbandingan sedang berlangsung, hasil akan dikirimkan setelah selesai.'})

# Pastikan aplikasi berjalan pada port yang sesuai untuk Vercel
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
