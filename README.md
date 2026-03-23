# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Jaya Jaya Maju

## Business Understanding

Perusahaan Jaya Jaya Maju, yang telah berdiri sejak tahun 2000 dan memiliki lebih dari 1000 karyawan, menghadapi tantangan serius dalam pengelolaan sumber daya manusia, khususnya terkait tingginya attrition rate (>10%).

### Permasalahan Bisnis

Perusahaan Jaya Jaya Maju menghadapi tantangan serius dalam pengelolaan SDM, terutama terkait:

- Tingginya attrition rate (>10%)
- Sulitnya mengidentifikasi faktor utama penyebab karyawan keluar
- Belum adanya sistem monitoring berbasis data
- Tidak adanya tools prediksi untuk mendeteksi karyawan berisiko resign

Dampak bisnis:

- Biaya rekrutmen & training meningkat
- Produktivitas tim menurun
- Tingginya turnover mengganggu stabilitas organisasi

### Cakupan Proyek

Cakupan dari proyek ini meliputi:

1. Data Preparation
    - Membersihkan data (missing values pada Attrition)
    - Transformasi label attrition (0 = Stayed, 1 = Left)
2. Exploratory Data Analysis (EDA)
    - Analisis hubungan attrition terhadap:
    - Department
    - Job Role
    - OverTime
    - Income
    - Age
    - Marital Status
    - Business Travel
    - Tenure (Years at Company)
3. Machine Learning Modeling
    - Model: Random Forest Classifier
    - Preprocessing:
    - Encoding (get_dummies)
    - Feature scaling (StandardScaler)
    - Output:
    - Prediksi attrition (binary)
    - Probabilitas attrition
4. Business Dashboard (Streamlit)
    - Monitoring KPI HR
    - Visualisasi faktor attrition
    - Interactive filtering
    - Prediksi attrition berbasis input user

### Persiapan

`employee_data.csv` <br>

Setup environment:

```bash
# Membuat environment baru
conda create -n attrition-env python=3.10 -y

# Mengaktifkan environment
conda activate attrition-env

# Install dependencies utama
conda install pandas numpy matplotlib seaborn scikit-learn -y

# Install streamlit & joblib via pip
pip install streamlit joblib
```

Menjalankan Dashboard: 
```bash
streamlit run dashboard.py
```

## Business Dashboard

Dashboard dibuat menggunakan Streamlit dan memiliki 2 fitur utama:

1. Dashboard Monitoring
    - KPI Utama:
        - Total Employees
        - Attrition Rate
        - Average Monthly Income
        - Average Tenure
        - Persentase karyawan lembur
    - Visualisasi Dashboard:
        - Attrition Distribution
            - Perbandingan jumlah karyawan keluar vs bertahan
        - Attrition by Department
            - Identifikasi departemen dengan attrition tertinggi
        - OverTime vs Attrition
            - Dampak lembur terhadap attrition
        - Attrition by Job Role
            - Role dengan risiko tertinggi
        - Income vs Attrition
            - Perbandingan gaji karyawan keluar vs bertahan
        - Age Distribution
            - Segmentasi usia dengan risiko attrition tinggi
        - Marital Status Analysis
            - Pengaruh status pernikahan
        - Business Travel
            - Dampak frekuensi perjalanan dinas
        - Correlation Heatmap
            - Hubungan antar fitur numerik
        - Feature Importance
            - Faktor paling berpengaruh dari model ML

    - Filter Interaktif
        - Department
        - Gender
        - Age Range
        - OverTime

2. Prediction Page
    - Input
        - Data lengkap karyawan
    - Output
        - Prediksi attrition
        - Probabilitas karyawan resign
        - Faktor utama penyebab karyawan resign

### Link Dashboard
```

```

## Conclusion
Dari hasil analisis dan dashboard, ditemukan beberapa faktor utama penyebab attrition, diantaranya: 
- OverTime (Lembur)
    - Karyawan lembur memiliki attrition jauh lebih tinggi
- Income (Gaji)
    - Karyawan dengan gaji rendah cenderung resign
- Job Role
    - Beberapa role memiliki attrition signifikan lebih tinggi
- Age (Usia)
    - Karyawan usia muda (early career) paling rentan keluar
- Marital Status
    - Karyawan single memiliki attrition lebih tinggi
- Business Travel
    - Frequent travel meningkatkan risiko attrition
- Tenure (Masa Kerja)
    - Karyawan baru (<2 tahun) memiliki attrition tinggi

### Rekomendasi Action Items (Optional)

1. Kurangi Beban Lembur
    - Evaluasi workload antar tim
    - Tambahkan resource pada tim dengan beban tinggi
2. Perbaiki Struktur Kompensasi
    - Benchmark gaji terhadap industri
    - Berikan insentif berbasis performa
3. Fokus pada Early Employee Retention
    - Program onboarding lebih kuat
    - Mentorship / buddy system
4. Tingkatkan Work-Life Balance
    - Flexible working
    - Program wellness
5. Monitoring Berbasis Data (WAJIB)
    - Gunakan dashboard ini secara rutin
    - Identifikasi karyawan high-risk secara proaktif
6. Kurangi Travel Berlebihan
    - Evaluasi kebutuhan business travel
    - Gunakan alternatif (online meeting)
