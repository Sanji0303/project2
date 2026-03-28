# 📌 Project 2: Recommendation System & Clustering

## 📁 Cấu trúc thư mục

```
Project 2/
│
├── data/
│   └── (clean_data_2)
│
├── source code/
│   ├── Preprocessing.ipynb
│   ├── BT1_Recommendation.ipynb
│   ├── BT2_Sklearn_Clustering.ipynb
│   ├── BT2_Pyspark_Clustering.ipynb
│
├── Project2.pptx
```
---
## 🧭 Hướng dẫn xem đồ án

Để nắm bắt luồng công việc của nhóm một cách dễ dàng nhất, xin vui lòng xem các file theo thứ tự sau:

1. **`Preprocessing.ipynb`**: Tiền xử lý dữ liệu
2. **`BT1_Recommendation.ipynb`**: Thực hiện xây dựng hệ thống gợi ý
3. **`BT2_Sklearn_Clustering.ipynb`**: Thực hiện xây dựng mô hình phân cụm trên môi trường Python thông thường
4. **`BT2_Pyspark_Clustering.ipynb`**: Thực hiện xây dựng mô hình phân cụm trên pyspark
---

## 👥 Team Members & Responsibilities

### 🔹 Đặng Đức Duy – Data & Preprocessing

* Data collection & merging
* Data cleaning (price, area)
* Handle missing values & outliers
* Feature engineering:
  * `text_clean`
  * `quan_encoded`
* Text preprocessing:
  * Cleaning
  * Stopword removal
  * Tokenization

---

### 🔹 Huỳnh Lê Xuân Ánh – Recommendation System

* Content-based recommendation:
  * TF-IDF
  * Cosine similarity
  * Gensim implementation
* Hybrid recommendation:
  * Content similarity
  * Price similarity
  * Location similarity
* Formula:

```
Hybrid = 0.5 * Content + 0.25 * Price + 0.25 * Location
```

---

### 🔹 Nguyễn Thị Tuyết Vân – Clustering & Evaluation

Clustering (Scikit-learn & PySpark)

# Phân Cụm Dữ Liệu Bất Động Sản
**Scikit-learn & PySpark — So sánh và chọn mô hình tối ưu**

## 1. Giới thiệu
Dự án này tập trung vào bài toán **phân cụm dữ liệu bất động sản** nhằm chia các bất động sản thành những nhóm có đặc điểm tương đồng, từ đó hỗ trợ:
- nhận diện các phân khúc nhà ở trên thị trường,
- hỗ trợ định giá,
- cung cấp insight cho phân tích dữ liệu bất động sản.

Bài toán được triển khai trên hai hệ sinh thái:
- **Scikit-learn**: phù hợp cho thử nghiệm nhanh và dữ liệu vừa/nhỏ.
- **PySpark**: phù hợp cho xử lý dữ liệu lớn, khả năng mở rộng cao.

---

## 2. Mục tiêu
- Phân khúc bất động sản dựa trên các đặc trưng số.
- So sánh hiệu quả giữa nhiều mô hình phân cụm.
- Xác định số cụm tối ưu.
- Chọn ra mô hình tốt nhất giữa **Sklearn** và **PySpark**.

---

## 3. Dữ liệu sử dụng
Các đặc trưng dùng để phân cụm gồm:

- `gia_ban_num`: Giá bán (VND)
- `dien_tich_num`: Diện tích (m²)
- `price_per_m2`: Giá trên mỗi mét vuông
- `quan_encoded`: Mã hóa quận/huyện

### Tiền xử lý
Dữ liệu được chuẩn hóa bằng **StandardScaler** để đưa các biến về cùng thang đo trước khi thực hiện phân cụm.

---

## 4. Các mô hình được sử dụng

### Scikit-learn
- **KMeans**
- **Gaussian Mixture Model (GMM)**
- **Agglomerative Clustering**

### PySpark
- **KMeans**
- **Gaussian Mixture Model (GMM)**
- **BisectingKMeans**

---

## 5. Tiêu chí đánh giá
Mô hình được đánh giá bằng **Silhouette Score**.

- Điểm càng cao thì cụm càng tách biệt rõ.
- Điểm được dùng để:
  - chọn số cụm tối ưu `K`,
  - so sánh chất lượng giữa các mô hình.

---

## 6. Kết quả chọn K tối ưu

### 6.1. Trên Scikit-learn
- **K = 2** cho Silhouette Score cao nhất: **0.48**
- **K = 5** là lựa chọn thay thế nếu muốn phân khúc chi tiết hơn

**Kết luận:**  
→ Chọn **K = 2** cho pipeline Sklearn.

### 6.2. Trên PySpark
- **K = 2** cho Silhouette Score cao nhất: **0.6952**
- **K = 8** có thể là lựa chọn thay thế nếu muốn tăng độ chi tiết

**Kết luận:**  
→ Chọn **K = 2** cho pipeline PySpark.

---

## 7. So sánh mô hình

### 7.1. Kết quả trên Scikit-learn (K=2)
| Mô hình | Silhouette Score | Đánh giá |
|--------|------------------:|----------|
| KMeans | 0.4796 | Tốt |
| GMM | 0.3691 | Thấp hơn |
| Agglomerative | 0.5933 | Tốt nhất |

**Kết luận:**  
→ **Agglomerative Clustering** là mô hình tốt nhất trên Scikit-learn.

### 7.2. Kết quả trên PySpark (K=2)
| Mô hình | Silhouette Score | Đánh giá |
|--------|------------------:|----------|
| KMeans | 0.6952 | Tốt nhất |
| GMM | 0.6812 | Rất tốt |
| BisectingKMeans | 0.5130 | Thấp hơn |

**Kết luận:**  
→ **KMeans** là mô hình tốt nhất trên PySpark.

---

## 8. Diễn giải kết quả phân cụm

### 8.1. Scikit-learn — Agglomerative Clustering (K=2)

#### Cluster 0 — Bình dân
- Giá trung bình: ~6.49 tỷ
- Diện tích trung bình: ~47.93 m²
- Giá/m²: ~144.7 triệu
- Số lượng: 6970

#### Cluster 1 — Cao cấp
- Giá trung bình: ~19.92 tỷ
- Diện tích trung bình: ~114.42 m²
- Giá/m²: ~195.9 triệu
- Số lượng: 911

**Nhận xét:**  
Hai cụm tách biệt rõ theo **giá bán** và **diện tích**, phản ánh tương đối đúng thực tế thị trường bất động sản.

---

### 8.2. PySpark — KMeans (K=2)

#### Cụm 0 — Phổ thông / Trung bình
- Giá trung bình: ~6.49 tỷ
- Diện tích trung bình: ~47.9 m²
- Giá/m²: ~144.7 triệu
- Số lượng: 6972

#### Cụm 1 — Cao cấp
- Giá trung bình: ~19.93 tỷ
- Diện tích trung bình: ~114.5 m²
- Giá/m²: ~195.97 triệu
- Số lượng: 909

**Nhận xét:**  
- Hai cụm được phân tách rõ ràng.
- Hai biến đóng vai trò phân tách mạnh nhất là:
  - `gia_ban_num`
  - `dien_tich_num`
- Biến `quan_encoded` có giá trị trung bình gần tương đương giữa hai cụm, nên đóng góp không nhiều vào việc tách cụm.

---

## 9. So sánh Scikit-learn và PySpark

| Tiêu chí | Scikit-learn | PySpark |
|---------|--------------|---------|
| Mô hình tốt nhất | Agglomerative | KMeans |
| Silhouette Score tốt nhất | 0.5933 | 0.6952 |
| Số cụm tối ưu | K = 2 | K = 2 |
| Khả năng mở rộng | Phù hợp dữ liệu vừa/nhỏ | Phù hợp dữ liệu lớn |
| Mức độ triển khai | Đơn giản | Phức tạp hơn |

### Nhận xét tổng quan
- **PySpark** cho kết quả tốt hơn về mặt Silhouette Score.
- Cả hai hệ đều chọn **K = 2** là số cụm tối ưu.
- Nếu dữ liệu mở rộng lớn trong tương lai, **PySpark** là lựa chọn phù hợp hơn.

---

## 10. Kết luận
Mô hình tối ưu của dự án là:

# **KMeans trên PySpark**
- **K = 2**
- **Silhouette Score = 0.6952**

### Lý do lựa chọn
- Có điểm Silhouette cao nhất trong tất cả các mô hình đã thử nghiệm.
- Phân cụm rõ ràng giữa hai phân khúc:
  - nhà phổ thông / trung cấp
  - nhà cao cấp / hạng sang
- Có khả năng mở rộng tốt khi dữ liệu tăng lên.

---

## 11. Công nghệ sử dụng
- Python
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- PySpark

---

## 12. Hướng phát triển
- Bổ sung thêm đặc trưng như:
  - số phòng ngủ,
  - số tầng,
  - loại hình nhà ở,
  - vị trí địa lý chi tiết hơn.
- Thử nghiệm thêm các phương pháp giảm chiều như PCA, t-SNE.
- Đánh giá thêm bằng các chỉ số khác ngoài Silhouette Score.
- Xây dựng pipeline tự động cho dữ liệu lớn trên Spark.

---

## 13. Tác giả
Dự án được thực hiện phục vụ mục đích học máy và phân tích dữ liệu bất động sản.
