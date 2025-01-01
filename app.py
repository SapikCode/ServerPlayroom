from flask import Flask, request, jsonify
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import tempfile
import os

app = Flask(__name__)

# Route untuk menampilkan pesan Halo dari server playroom saat mengakses root
@app.route('/')
def hello():
    return jsonify({'message': 'Halo dari server playroom'})

def compare_images_ssim(img1_path, img2_path):
    # Membaca gambar menggunakan PIL (Pillow)
    img1 = Image.open(img1_path).convert('L')  # Konversi ke grayscale
    img2 = Image.open(img2_path).convert('L')  # Konversi ke grayscale

    # Mengubah ukuran img2 agar sama dengan img1
    img2_resized = img2.resize(img1.size)

    # Mengonversi gambar ke numpy array
    img1_array = np.array(img1)
    img2_array = np.array(img2_resized)

    # Menghitung SSIM
    similarity_index, _ = ssim(img1_array, img2_array, full=True)
    return similarity_index

@app.route('/compare', methods=['POST'])
def compare():
    try:
        # Mengambil file gambar dari request
        image1 = request.files['image1']
        image2 = request.files['image2']
        
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

        # Mengembalikan hasil SSIM dan hasil perbandingan
        return jsonify({
            'ssim_score': round(ssim_score * 100, 2), 
            'result': result
        })
    
    except Exception as e:
        # Jika ada error, kembalikan error message
        return jsonify({'error': str(e)}), 500

# Pastikan aplikasi berjalan pada port yang sesuai untuk Vercel
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
