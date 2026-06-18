# Human Detection with HOG + SVM

> Proyek Akhir — Pengolahan dan Analisis Citra Digital  
> **Robby Azwan Saputra** · NPM 140810230008  
> Teknik Informatika, Universitas Padjadjaran

---

## Deskripsi

Implementasi sistem **Human Detection** menggunakan metode klasik computer vision:
- **HOG (Histogram of Oriented Gradients)** sebagai feature extractor
- **LinearSVC (Support Vector Machine)** sebagai classifier
- **Hard Negative Mining (HNM)** untuk meningkatkan kualitas deteksi
- **Streamlit** sebagai antarmuka web interaktif

Dua pendekatan diimplementasikan dan dibandingkan:

| Pendekatan | Deskripsi |
|---|---|
| **Custom Model + HNM** | Melatih LinearSVC dari nol dengan INRIA dataset + Hard Negative Mining |
| **OpenCV Default** | Menggunakan pretrained model bawaan OpenCV (`HOGDescriptor_getDefaultPeopleDetector`) |

---

## Struktur Proyek

```
project/
├── app.py                                          # Streamlit web app
├── hog_svm_human_model_hnm.pkl                     # Model terlatih (hasil training)
├── train_hog_svm_human_detection_hnm_fixed.ipynb   # Notebook training custom model
├── opencv_hog_default.ipynb                        # Notebook OpenCV default detector
├── README.md
└── dataset/
    ├── Train/
    │   ├── Annotations/     ← Pascal VOC XML
    │   └── JPEGImages/
    └── Test/
        ├── Annotations/
        └── JPEGImages/
```

---

## Dataset

**INRIA Person Dataset** — benchmark klasik pedestrian detection (Dalal & Triggs, 2005)

| Subset | Jumlah Gambar | Keterangan |
|---|---|---|
| Train/positive | ~614 | Gambar dengan anotasi bounding box person |
| Train/negative | ~1218 | Gambar background tanpa orang |
| Test/positive | ~288 | Gambar test berisi orang |
| Test/negative | ~453 | Gambar test background |

Download: http://pascal.inrialpes.fr/data/human/

---

## Instalasi

```bash
pip install streamlit scikit-image scikit-learn opencv-python
pip install matplotlib joblib numpy tqdm pandas
```

| Library | Kegunaan |
|---|---|
| `scikit-image` | Ekstraksi HOG, image pyramid Gaussian |
| `scikit-learn` | LinearSVC, StandardScaler, Pipeline |
| `opencv-python` | Baca/tulis gambar, HOG default detector |
| `streamlit` | Antarmuka web interaktif |
| `joblib` | Simpan dan load model `.pkl` |

---

## Penggunaan

### 1. Training Model Custom (opsional — skip jika sudah punya `.pkl`)

Buka dan jalankan semua cell di notebook:
```
train_hog_svm_human_detection_hnm_fixed.ipynb
```

Pipeline training:
1. Build dataset (positive crop + random negative crop dari INRIA)
2. Train model awal (`C=0.1`)
3. Hard Negative Mining — temukan false positive, retrain (`C=0.01`)
4. Evaluasi — confusion matrix, classification report
5. Simpan model ke `hog_svm_human_model_hnm.pkl`

### 2. OpenCV Default Detector

Buka dan jalankan:
```
opencv_hog_default.ipynb
```

Tidak perlu dataset atau training — langsung load pretrained model OpenCV dan detect.

### 3. Streamlit Web App

```bash
streamlit run app.py
```

Buka browser di **http://localhost:8501**

> Pastikan `app.py` dan `hog_svm_human_model_hnm.pkl` berada di folder yang sama.

---

## Fitur Streamlit App

Upload gambar JPG/PNG/JPEG → pipeline deteksi berjalan otomatis dengan visualisasi 6 step:

| Step | Visualisasi |
|---|---|
| 01 | Gambar original + info resolusi |
| 02 | Preprocessing — resize, grayscale, histogram piksel |
| 03 | HOG visualization — grayscale vs heatmap inferno |
| 04 | Image pyramid — semua level Gaussian pyramid |
| 05 | Raw detections sebelum NMS (box kuning) |
| 06 | Hasil final setelah NMS (box hijau) + confidence score |

### Parameter Tuning (Sidebar)

| Parameter | Default | Keterangan |
|---|---|---|
| Confidence Threshold | `1.1` | Naikkan → kurangi false positive |
| NMS IoU Threshold | `0.35` | Naikkan → buang lebih banyak box tumpuk |
| Sliding Window Step | `10` | Turunkan → lebih akurat (lebih lambat) |
| Pyramid Downscale | `1.25` | Turunkan → lebih banyak level pyramid |
| Max Image Width | `700` | Resize gambar sebelum diproses |

### Rekomendasi Setting per Kondisi

| Kondisi | Threshold | NMS IoU |
|---|---|---|
| 1 orang, foreground jelas | `1.6` | `0.5` |
| 2–3 orang | `1.2` | `0.4–0.5` |
| Banyak orang / crowd | `0.9` | `0.35` |

---

## Hasil Evaluasi

### Model Custom + HNM

| Metrik | Validation Set | Test Set |
|---|---|---|
| Accuracy | 95.5% | 87% |
| Precision (Human) | 96% | ~87% |
| Recall (Human) | 95% | ~87% |
| F1-Score | 96% | ~87% |

> Penurunan val→test adalah wajar dan tidak mengindikasikan overfit (gap terkontrol, arah konsisten).

---

## Cara Kerja

### HOG Feature Extraction

```
Window 64×128 px
  └── Cell 8×8 px
        └── Block 2×2 cell (overlap 50%)
              └── Histogram 9 bin (0°–180°)
                    └── Total fitur: 105 blocks × 36 = 3,780 dimensi
```

### Pipeline Deteksi

```
Gambar input
  └── Image Pyramid (pyramid_gaussian, downscale=1.25)
        └── Sliding Window (step=10px, window=64×128)
              └── Ekstrak HOG → LinearSVC.decision_function()
                    └── threshold filter
                          └── Non-Maximum Suppression (IoU)
                                └── Final bounding boxes
```

### Hard Negative Mining

```
1. Train model awal dengan positive + random negative
2. Jalankan detector di semua training images
3. Kumpulkan false positive (background salah detect sebagai orang)
4. Tambahkan ke dataset sebagai hard negative
5. Retrain model final → lebih robust terhadap background kompleks
```

---

## Perbandingan Pendekatan

| Aspek | Custom + HNM | OpenCV Default |
|---|---|---|
| Training | Diperlukan | Tidak perlu |
| Kecepatan | Python loop (lambat) | C++ built-in (cepat) |
| Kontrol | Penuh | Terbatas |
| Transparansi | Tinggi | Black box |
| Cocok untuk | Akademik / research | Prototyping cepat |

---

## Referensi

- Dalal, N., & Triggs, B. (2005). *Histograms of oriented gradients for human detection*. CVPR 2005.
- Tukra, S. (2019). *Hands-On Image Processing with Python*. Packt Publishing. Chapter 8.
- scikit-image HOG docs: https://scikit-image.org/docs/stable/api/skimage.feature.html
- OpenCV HOGDescriptor: https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html
- Streamlit docs: https://docs.streamlit.io

