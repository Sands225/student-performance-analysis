# Proyek Akhir: Prediksi Performa dan Status Kelulusan Mahasiswa

## Business Understanding

Institusi pendidikan menghadapi tantangan dalam memastikan keberhasilan akademik mahasiswa. Salah satu permasalahan utama adalah tingginya angka mahasiswa yang tidak menyelesaikan studi (dropout) serta kurangnya sistem monitoring berbasis data untuk mendeteksi risiko tersebut sejak dini.

### Permasalahan Bisnis

Institusi pendidikan menghadapi beberapa tantangan utama:

- Tingginya angka mahasiswa dropout
- Sulitnya memantau status akademik mahasiswa (Dropout vs Graduate) secara menyeluruh
- Belum adanya sistem prediksi otomatis untuk mendeteksi mahasiswa berisiko
- Tidak adanya dashboard monitoring performa akademik yang terintegrasi

Dampak bisnis:

- Penurunan reputasi dan peringkat institusi
- Inefisiensi dalam alokasi beasiswa dan sumber daya
- Penurunan pendapatan dari biaya pendidikan (tuition fees)

### Cakupan Proyek

Cakupan proyek ini meliputi:

1. Data Preparation
   - Mengolah dataset `data.csv`
   - Menghapus kelas *Enrolled* — fokus pada klasifikasi binary:
     - Dropout = 0
     - Graduate = 1

2. Exploratory Data Analysis (EDA)
   - Analisis distribusi status mahasiswa (Graduate vs Dropout)
   - Analisis hubungan faktor-faktor akademik terhadap kelulusan:
     - Scholarship holder
     - Tuition fees up to date
     - Admission grade
     - Age at enrollment
     - Performa unit kurikuler semester 1 & 2

3. Machine Learning Modeling
   - Melatih 4 model dan memilih yang terbaik berdasarkan Test Accuracy:
     - Logistic Regression (`max_iter=1000`, `random_state=42`)
     - Decision Tree (`max_depth=6`, `random_state=42`)
     - Random Forest (`n_estimators=100`, `random_state=42`)
     - Gradient Boosting (`n_estimators=100`, `random_state=42`)
   - Preprocessing:
     - Label Encoding pada target (Dropout=0, Graduate=1)
     - Feature scaling menggunakan StandardScaler (khusus Logistic Regression)
     - Data split: 80% training, 20% testing (stratified, `random_state=42`)
   - Output:
     - Prediksi status mahasiswa: **Dropout** atau **Graduate**
     - Perbandingan Test Accuracy & CV Accuracy (5-fold) seluruh model

4. Business Dashboard (Streamlit)
   - Visualisasi performa akademik mahasiswa
   - Perbandingan performa seluruh model
   - Prediksi status mahasiswa berbasis input user menggunakan best model
   - Halaman rekomendasi berbasis insight data

### Persiapan

**Sumber Data:**\
[students_performance.csv](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

**Setup Environment:**

Proyek ini menggunakan **Python 3.10**. Ikuti langkah berikut untuk menyiapkan environment dan menjalankan proyek.

```bash
# 1. Membuat virtual environment baru
conda create -n student-performance-env python=3.10 -y

# 2. Mengaktifkan environment
conda activate student-performance-env

# 3. Install seluruh dependencies dari requirements.txt
pip install -r requirements.txt
```

> Pastikan file `requirements.txt` tersedia di direktori proyek. Contoh isi `requirements.txt`:
> ```
> pandas
> numpy
> matplotlib
> scikit-learn
> streamlit
> joblib
> ```

**Struktur Direktori:**

```
.
├── dashboard.py               # File utama Streamlit dashboard
├── student_performance.ipynb  # Notebook EDA & modeling
├── requirements.txt           # Daftar dependencies
├── dataset/
│   └── data.csv               # Dataset utama
└── model/
    ├── logreg_model.pkl        # Model tersimpan (opsional)
    ├── scaler.pkl              # Scaler tersimpan (opsional)
    └── label_encoder.pkl       # Label encoder tersimpan (opsional)
```

**Menjalankan Notebook (EDA & Modeling):**

```bash
jupyter notebook student_performance.ipynb
```

**Menjalankan Streamlit Dashboard:**

```bash
streamlit run dashboard.py
```

Setelah perintah di atas dijalankan, dashboard akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

## Business Dashboard

Dashboard dibuat menggunakan Streamlit dan memiliki 4 halaman utama:

### 1. Analysis Page

Menampilkan distribusi dan analisis data mahasiswa **Graduate & Dropout**:

- Metrik ringkasan: total mahasiswa, jumlah Graduate, jumlah Dropout
- **Viz 1 — Status Distribution:** pie chart proporsi Graduate vs Dropout
- **Viz 2 — Grade Trajectory Sem 1 → 2:** scatter plot nilai semester 1 vs 2 berdasarkan status
- **Viz 3 — Financial Factors:** stacked bar chart Tuition & Scholarship vs status
- **Viz 4 — Courses Passed:** box plot MK lulus semester 1 & 2 per status
- **Viz 5 — Enrollment Age Distribution:** histogram distribusi usia pendaftaran
- **Viz 6 — Debt & Gender Risk Profile:** stacked bar chart debt status & gender vs outcome

### 2. Model Comparison Page

Menampilkan perbandingan performa keempat model yang dilatih:

- Tabel Test Accuracy & CV Accuracy (5-fold) semua model
- Bar chart perbandingan visual antar model
- Highlight best model secara otomatis
- Confusion matrix & classification report best model

### 3. Prediction Page

- **Input:** 15 fitur utama mahasiswa, antara lain:
  - Previous Qualification Grade & Admission Grade
  - Age at Enrollment
  - Performa kurikuler Sem 1 & 2 (enrolled, evaluations, passed, grade)
  - Status finansial (scholarship, tuition, debtor)
  - Atribut demografis (gender, displaced, attendance, international)
- **Output:**
  - Prediksi status: **Dropout** atau **Graduate**
  - Confidence score (%)
  - Bar chart probabilitas per kelas
  - Pesan rekomendasi sesuai hasil prediksi
- Model yang digunakan adalah **best model** hasil seleksi otomatis dari Model Comparison

### 4. Recommendations Page

- Ringkasan performa best model (accuracy, classification report, confusion matrix)
- 6 temuan utama berbasis analisis data
- Action plan berprioritas (High / Medium / Low)
- Gantt chart implementation roadmap (9 bulan)

## Menjalankan Sistem Machine Learning

Sistem Machine Learning dapat diakses sekaligus dengan Dashboard yang telah dibuat pada page `recommendations` pada link dibawah ini.

### Link Dashboard

https://student-performance-analysis-rww3uavtdzm7lwjcisfmg6.streamlit.app/

## Conclusion

Berdasarkan hasil analisis dan pemodelan dengan klasifikasi binary (Graduate vs Dropout), ditemukan beberapa faktor utama yang memengaruhi status mahasiswa:

- Performa akademik di semester awal — nilai dan jumlah MK lulus (prediktor terkuat)
- Status beasiswa (scholarship holder)
- Ketepatan pembayaran biaya pendidikan (tuition fees up to date)
- Status utang (debtor) yang memperparah risiko dropout
- Usia masuk >23 tahun sebagai indikator risiko tambahan

Selain itu, dilakukan proses pelatihan dan evaluasi terhadap empat algoritma machine learning dengan skema pembagian data sebesar 80% untuk training dan 20% untuk testing menggunakan teknik *stratified sampling*. Evaluasi model juga diperkuat dengan *cross-validation* 5-fold untuk mengukur kestabilan performa.

Adapun hasil performa masing-masing model adalah sebagai berikut:

- **Logistic Regression**  
  Model ini dikonfigurasi dengan parameter *max_iter = 1000* serta menggunakan proses *feature scaling* melalui *StandardScaler*. Model ini menghasilkan **Test Accuracy sebesar 0.9146** dan **CV Accuracy sebesar 0.9081**, yang menunjukkan performa yang tinggi sekaligus stabil.

- **Random Forest**  
  Dengan konfigurasi *n_estimators = 100*, model ini juga menunjukkan performa yang sangat baik dengan **Test Accuracy sebesar 0.9146** dan **CV Accuracy sebesar 0.9043**.

- **Gradient Boosting**  
  Model dengan *n_estimators = 100* ini mencatatkan **Test Accuracy sebesar 0.9118** dan **CV Accuracy sebesar 0.9053**, sehingga masih tergolong kompetitif dibandingkan model lainnya.

- **Decision Tree**  
  Model ini dibatasi menggunakan parameter *max_depth = 6*. Hasil evaluasi menunjukkan performa paling rendah di antara seluruh model, dengan **Test Accuracy sebesar 0.8981** dan **CV Accuracy sebesar 0.8888**.

### Rangkuman Performa Model

| Model                | Test Accuracy | CV Accuracy (5-fold) |
|---------------------|--------------|----------------------|
| Logistic Regression | 0.9146       | 0.9081               |
| Random Forest       | 0.9146       | 0.9043               |
| Gradient Boosting   | 0.9118       | 0.9053               |
| Decision Tree       | 0.8981       | 0.8888               |


Berdasarkan hasil evaluasi, **Logistic Regression** dan **Random Forest** memiliki nilai *Test Accuracy* tertinggi, yaitu sebesar 0.9146. Namun, **Logistic Regression** dipilih sebagai model terbaik karena memiliki nilai *Cross-Validation Accuracy* yang paling tinggi (0.9081), yang menunjukkan tingkat generalisasi dan kestabilan model yang lebih baik.

Model terbaik ini kemudian diintegrasikan ke dalam sistem dashboard untuk memprediksi status kelulusan mahasiswa. Implementasi model ini diharapkan dapat membantu institusi dalam melakukan deteksi dini terhadap mahasiswa yang berisiko mengalami *dropout*, sehingga intervensi dapat dilakukan secara lebih cepat dan tepat sasaran.
### Rekomendasi Action Items

1. **🔴 Intervensi Akademik Dini (High)**
   - Bangun sistem early-warning otomatis yang memonitor nilai & jumlah MK lulus di Sem 1
   - Berikan program remedial atau mentoring bagi mahasiswa di bawah threshold

2. **🔴 Optimalisasi Dukungan Finansial (High)**
   - Identifikasi mahasiswa dengan SPP tunggak di minggu ke-4 setiap semester
   - Tawarkan cicilan atau beasiswa darurat sebelum semester berakhir

3. **🔴 Program Konseling Debt (High)**
   - Konseling khusus untuk mahasiswa berstatus debtor
   - Hubungkan ke sumber beasiswa eksternal dan program manajemen keuangan

4. **🟡 Program Pendampingan Non-Tradisional (Medium)**
   - Jadwal fleksibel dan mentoring karier untuk mahasiswa usia >23 tahun
   - Perluas kuota beasiswa internal dan sosialisasikan beasiswa eksternal

5. **🟢 Monitoring Berbasis Data — SIAK Integration (Low)**
   - Integrasikan best model ke Sistem Informasi Akademik (SIAK)
   - Tampilkan risk-score real-time per mahasiswa per semester
   - Gunakan dashboard secara rutin untuk identifikasi mahasiswa berisiko secara proaktif