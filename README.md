# Proyek Akhir: Prediksi Performa dan Status Kelulusan Mahasiswa

## Business Understanding

Institusi pendidikan menghadapi tantangan dalam memastikan keberhasilan akademik mahasiswa. Salah satu permasalahan utama adalah tingginya angka mahasiswa yang tidak menyelesaikan studi (dropout) serta kurangnya sistem monitoring berbasis data untuk mendeteksi risiko tersebut sejak dini.

### Permasalahan Bisnis

Institusi pendidikan menghadapi beberapa tantangan utama:

- Tingginya angka mahasiswa dropout
- Sulitnya memantau status akademik mahasiswa secara keseluruhan (Dropout, Enrolled, Graduate)
- Belum adanya sistem prediksi otomatis untuk mendeteksi mahasiswa berisiko
- Tidak adanya dashboard monitoring performa akademik yang terintegrasi

Dampak bisnis:

- Penurunan reputasi dan peringkat institusi
- Inefisiensi dalam alokasi beasiswa dan sumber daya
- Penurunan pendapatan dari biaya pendidikan (tuition fees)

### Cakupan Proyek

Cakupan proyek ini meliputi:

1. Data Preparation
   - Mengolah dataset data.csv
   - Transformasi label:
     - Dropout = 0
     - Enrolled = 1
     - Graduate = 2

2. Exploratory Data Analysis (EDA)
   - Analisis distribusi status mahasiswa
   - Analisis hubungan faktor-faktor akademik terhadap kelulusan:
     - Scholarship holder
     - Tuition fees up to date
     - Admission grade
     - Age at enrollment
     - Performa unit kurikuler

3. Machine Learning Modeling
   - Model: Logistic Regression
   - Preprocessing:
     - Label Encoding pada target
     - Feature scaling menggunakan StandardScaler
     - Data split: 80% training, 20% testing (stratified)
   - Output:
     - Prediksi status mahasiswa (Dropout, Enrolled, Graduate)

4. Business Dashboard (Streamlit)
   - Visualisasi performa akademik mahasiswa
   - Analisis faktor-faktor yang memengaruhi kelulusan
   - Prediksi status mahasiswa berbasis input user
   - Halaman rekomendasi berbasis insight data

### Persiapan

Dataset:
dataset/data.csv

Setup environment:
```bash
# Membuat environment baru
conda create -n student-performance-env python=3.10 -y

# Mengaktifkan environment
conda activate student-performance-env

# Install dependencies utama
conda install pandas numpy matplotlib scikit-learn -y

# Install streamlit
pip install streamlit
```
Menjalankan Dashboard:
```bash
streamlit run app.py
```
## Business Dashboard

Dashboard dibuat menggunakan Streamlit dan memiliki 3 fitur utama:

1. Analysis Page

- Menampilkan distribusi status mahasiswa:
  - Dropout
  - Enrolled
  - Graduate
- Visualisasi faktor yang memengaruhi performa:
  - Scholarship holder vs status
  - Tuition fees vs status
  - Performa akademik semester awal
- Insight dari data eksplorasi

2. Prediction Page

- Input:
  - Data mahasiswa (36 fitur), seperti:
    - Admission grade
    - Age at enrollment
    - Curricular units performance
    - Financial status

- Output:
  - Prediksi status mahasiswa:
    - Dropout
    - Enrolled
    - Graduate
  - Akurasi model ditampilkan di sidebar

3. Recommendations Page

- Ringkasan insight dari analisis data
- Rekomendasi strategis untuk institusi pendidikan

### Link Dashboard

https://student-performance-analysis-dueqjb6ymosmuvpijadgtx.streamlit.app/

## Conclusion

Berdasarkan hasil analisis dan pemodelan menggunakan Logistic Regression, ditemukan beberapa faktor utama yang memengaruhi status mahasiswa:

- Performa akademik di semester awal (sangat krusial)
- Status beasiswa (scholarship holder)
- Ketepatan pembayaran biaya pendidikan (tuition fees up to date)

Model ini membantu institusi dalam melakukan deteksi dini terhadap mahasiswa yang berisiko dropout sehingga intervensi dapat dilakukan lebih cepat.

### Rekomendasi Action Items

1. Intervensi Akademik Dini
   - Fokus pada mahasiswa dengan performa rendah di semester 1 dan 2
   - Berikan program remedial atau mentoring

2. Optimalisasi Dukungan Finansial
   - Evaluasi dan perluas program beasiswa
   - Prioritaskan mahasiswa dengan risiko finansial tinggi

3. Monitoring Pembayaran
   - Berikan pengingat pembayaran
   - Sediakan opsi pembayaran fleksibel

4. Monitoring Berbasis Data (WAJIB)
   - Gunakan dashboard secara rutin
   - Identifikasi mahasiswa berisiko tinggi secara proaktif
   - Lakukan tindakan preventif sebelum mahasiswa dropout