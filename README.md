# Hệ Thống Phát Hiện Giao Dịch Tài Chính Bất Thường Và Nghi Vấn Gian Lận
### (Financial Fraud Detection)

Đồ án cuối kỳ môn **Trí tuệ nhân tạo** — Trường Đại học Công nghệ Thông tin, ĐHQG-HCM
Lớp: CS106.F31.CN2.TTNT · GVHD: Nguyễn Đình Hiển

**Nhóm 5:**
- Ngô Thị Vân Giang — 25730020
- Lê Vân Anh — 25730008
- Bùi Thị Hoàng — 25730029
- Huỳnh Trường An — 25730004
- Trương Nam Anh — 25730010

---

## 1. Bài toán

Dữ liệu giao dịch gian lận có đặc điểm **mất cân bằng nghiêm trọng**: một mô hình dự đoán mọi giao dịch đều hợp lệ vẫn đạt Accuracy rất cao nhưng vô dụng trong thực tế. Đề tài xây dựng một hệ thống học máy phân loại giao dịch thành:

- **Input:** 14 đặc trưng hành vi (số tiền, giờ giao dịch, tần suất, khoảng cách, loại dịch vụ...)
- **Output:** xác suất gian lận → áp ngưỡng quyết định → nhãn 0 (hợp lệ) / 1 (gian lận)
- **Tiêu chí đánh giá:** Precision, Recall, F1, **F2-Score** (ưu tiên chính) và **PR-AUC** — không dùng Accuracy.

## 2. Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Tổng số giao dịch | 30.058 |
| Số cột thuộc tính gốc | 23 |
| Số thẻ tín dụng / điểm chấp nhận thanh toán | 909 thẻ / 693 merchant, tại 50 bang (Hoa Kỳ) |
| Thời gian ghi nhận | 21/06 – 30/06/2020 (10 ngày) |
| Số tiền giao dịch | 1 – 6.600,44 USD (trung vị 46,28 USD) |
| Giao dịch hợp lệ / gian lận | 29.925 / 133 (0,44%) |
| Tỷ lệ mất cân bằng | ~1 : 225 |

**Lỗi dữ liệu nguồn đã phát hiện và khôi phục:** tọa độ (`lat/long/merch_lat/merch_long`) bị mất dấu thập phân; mã thẻ `cc_num` bị làm tròn khiến chỉ còn 309 mã (khôi phục thành 909 thẻ bằng định danh chủ thẻ); cột `unix_time` bị ghi đè hằng số (tính lại từ thời gian giao dịch).

**Phát hiện chính từ EDA:**
- 3 loại dịch vụ `shopping_net`, `misc_net`, `grocery_pos` chiếm 82/133 vụ gian lận (61,7%). `shopping_net` có tỷ lệ fraud cao nhất (1,43%), thấp nhất là `personal_care` (0,18%) — kiểm định Chi-square p = 1,28×10⁻²³ (có ý nghĩa thống kê).
- 78,2% giao dịch gian lận nằm trong vùng giá trị bất thường (outlier) của số tiền giao dịch → **không loại bỏ outlier** mà chỉ giới hạn (clip) theo IQR sau khi chia tập, để tránh mất chính các giao dịch gian lận.
- Giao dịch gian lận có tốc độ di chuyển và tỷ lệ chi tiêu so với trung bình thẻ cao hơn rõ rệt (trung vị `speed_kmh` 59,4 so với 21,2 km/h; `amt_vs_card_avg` 1,94 so với 0,67).

## 3. Quy trình xử lý dữ liệu

1. **Thu thập:** nạp CSV từ Google Sheets bằng pandas.
2. **Làm sạch:** chuẩn hóa số, sửa lỗi tọa độ/mã thẻ/thời gian, kiểm tra dữ liệu thiếu (không có dòng nào bị loại).
3. **Feature Engineering — 14 đặc trưng:** `amt`, `age`, `hour`, `amt_vs_card_median`, `amt_vs_card_avg`, `txn_freq_hour`, `txn_count_recent`, `distance_km`, `speed_kmh`, `amt_log`, `amt_zscore_card`, `amt_per_km`, `is_night`, `category`.
4. **Chia dữ liệu (stratify, random_state=42):** Train&Val 24.046 (106 fraud) / Test 6.012 (27 fraud); Train&Val → Fit 19.236 (85 fraud) / Validation 4.810.
5. **Chống Data Leakage:** mọi thống kê theo thẻ, IQR-clip, StandardScaler, TargetEncoder đều **fit trên tập Fit**, áp dụng lại cho Validation/Test.
6. **Tinh chỉnh siêu tham số:** RandomizedSearchCV, CV 5-fold, scoring = `average_precision`, cố định quy trình cho toàn bộ 6 thử nghiệm.
7. **Chọn ngưỡng quyết định:** trên tập Validation, tối ưu F2-Score với ràng buộc Precision ≥ 0,6.
8. **Đánh giá cuối:** một lần duy nhất trên tập Test.

## 4. Mô hình & xử lý mất cân bằng

| Chiến lược | Mô hình |
|---|---|
| `class_weight` (balanced / thủ công 1:k) | Random Forest |
| `scale_pos_weight` (0,1–1,0 lần tỷ lệ mất cân bằng) | XGBoost, LightGBM |
| SMOTE nhẹ (2–5%, trong Pipeline theo từng fold CV) | XGBoost |
| `auto_class_weights = "Balanced"` | CatBoost |
| Stacking (Top-3 base learners + meta-learner Logistic Regression) | XGBoost + LightGBM + CatBoost |

## 5. Sáu thử nghiệm (TN1–TN6)

Cùng một cách chia tập, cùng bộ thuật toán và quy trình tinh chỉnh — chỉ thay đổi **tập đặc trưng đầu vào**, nhằm tách riêng ảnh hưởng của phương pháp mã hóa và của việc loại bỏ đặc trưng nhiễu:

| TN | Biểu diễn đặc trưng | Số đặc trưng | Best PR-AUC (mô hình) |
|---|---|---|---|
| TN1 | 6 đặc trưng gốc, chưa engineering | 6 | 0,5016 (Random Forest) |
| TN2 | OneHotEncoder | 26 | 0,8092 (LightGBM) |
| **TN3** | **TargetEncoder** | **14** | **0,8197 (LightGBM)** ⭐ |
| TN4 | TN3 trừ `is_night`, `amt_vs_card_median`, `distance_km`, `amt` | 10 | 0,8231 (LightGBM) |
| TN5 | TN3 trừ `is_night`, `amt_vs_card_median`, `distance_km` | 11 | 0,8205 (LightGBM) |
| TN6 | TN3 trừ `amt`, `amt_vs_card_median`, `distance_km` | 11 | 0,8068 (LightGBM) |

**Kết luận từ so sánh:** kỹ thuật tạo đặc trưng quan trọng hơn phương pháp mã hóa (TN1→TN3: PR-AUC tăng 63%, chỉ nhờ feature engineering); loại bỏ đặc trưng (TN4–TN6) **không cải thiện mô hình có ý nghĩa** — chênh lệch PR-AUC so với TN3 (≤0,0034) nhỏ hơn sai số do tập Test chỉ có 27 mẫu gian lận (1 mẫu ≈ 3,7% Recall).

**Permutation Importance (đo trên tập Validation, mô hình LightGBM/TN3):** `amt_log` chi phối gần như toàn bộ hiệu quả (0,5778 — gấp hơn 3 lần đặc trưng đứng thứ 2 là `age`), tiếp theo `category`, `amt_vs_card_avg`, `hour`; các đặc trưng bị loại ở TN4–TN6 (`distance_km`, `amt_vs_card_median`, `is_night`) có ảnh hưởng thấp, giải thích vì sao loại bỏ chúng không cải thiện kết quả.

## 6. Kết quả (tập Test, TN3 — 14 đặc trưng, TargetEncoder)

| Mô hình | Threshold | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FNR | FPR |
|---|---|---|---|---|---|---|---|---|---|
| **LightGBM** ⭐ | 0,0634 | **0,9091** | 0,7407 | **0,8163** | **0,7692** | 0,9828 | 0,8197 | 25,93% | **0,03%** |
| Stacking | 0,9827 | 0,7000 | **0,7778** | 0,7368 | 0,7609 | 0,9598 | 0,7820 | 22,22% | 0,15% |
| Random Forest | 0,2498 | 0,6897 | 0,7407 | 0,7143 | 0,7299 | 0,9516 | 0,7475 | 25,93% | 0,15% |
| XGBoost | 0,2545 | 0,5882 | 0,7407 | 0,6557 | 0,7042 | 0,9758 | 0,7398 | 25,93% | 0,23% |
| CatBoost | 0,8407 | 0,3922 | 0,7407 | 0,5128 | 0,6289 | 0,9711 | 0,6895 | 25,93% | 0,52% |

Trên tập Test (6.012 giao dịch, 27 gian lận, 5.985 hợp lệ): **LightGBM** phát hiện đúng 20/27 giao dịch gian lận, chỉ báo nhầm 2/5.985 giao dịch hợp lệ — thấp nhất trong 5 mô hình.

## 7. Mô hình đề xuất: LightGBM + TargetEncoder (TN3)

**Lý do lựa chọn** (không chỉ dựa vào PR-AUC cao nhất tuyệt đối):
- **Cân bằng tốt nhất** trên cả F1 (0,8163) và F2 (0,7692) — cao nhất trong toàn bộ 6 thử nghiệm, trong khi TN4 có PR-AUC nhỉnh hơn (+0,0034) thì lại có F2 thấp hơn, FPR cao hơn và chậm hơn.
- **Chi phí nghiệp vụ thấp nhất:** Precision 90,9% → trong 22 cảnh báo chỉ 2 cảnh báo sai (FPR 0,03%, thấp hơn Stacking 5 lần).
- **Chi phí tính toán thấp:** dự đoán 6.012 giao dịch chỉ 0,036 giây; tinh chỉnh chỉ 231,85s (thấp hơn TN2 gấp 4,9 lần, TN4 gấp 9,6 lần).
- Chênh lệch PR-AUC với TN4 (0,0034) nhỏ hơn sai số thống kê do tập Test chỉ có 27 mẫu gian lận → không đủ căn cứ đánh đổi.

## 8. Hạn chế

- Dữ liệu mô phỏng, chỉ quan sát trong 10 ngày — chưa phản ánh đầy đủ các phương thức gian lận biến đổi theo thời gian.
- Tập Test chỉ có 27 giao dịch gian lận → sai số thống kê giữa các cấu hình gần nhau khá lớn (1 mẫu ≈ 3,7% Recall).
- Đặc trưng `distance_km` chứa nhiễu do chất lượng dữ liệu tọa độ.
- Xác suất dự đoán của các mô hình **chưa được hiệu chuẩn tốt** (Calibration Curve lệch xa đường "Perfect calibration") — ảnh hưởng đến việc diễn giải xác suất tuyệt đối, dù không ảnh hưởng đến khả năng phân loại theo ngưỡng.
- Không gian siêu tham số được khảo sát bằng RandomizedSearchCV với số lần thử giới hạn, chưa đảm bảo tối ưu toàn cục.

## 9. Hướng phát triển

- Mở rộng dữ liệu thực tế nhiều tháng; bổ sung đặc trưng thiết bị/IP/lịch sử tranh chấp.
- Hiệu chuẩn xác suất bằng `CalibratedClassifierCV`; chọn ngưỡng vận hành theo chi phí kinh doanh thay vì chỉ F2-Score.
- Thử thêm CatBoostEncoder, mã hóa theo chuỗi thời gian; tối ưu hóa Bayes cho siêu tham số.
- Mở rộng Stacking với nhiều mô hình cơ sở hơn; triển khai chấm điểm thời gian thực với giám sát lệch phân bố dữ liệu.

## 10. Cấu trúc thư mục

```
├── notebook/                          # Notebook xử lý dữ liệu, huấn luyện & đánh giá mô hình
├── Presentation.pptx                  # Slide thuyết trình đồ án
├── Bao_cao_do_an_mon_TTNT.pdf         # Báo cáo đầy đủ (6 chương + tài liệu tham khảo)
├── README.md                          # Tài liệu giới thiệu đồ án (file này)
```

## 11. Công cụ & thư viện

Python, pandas, scikit-learn (`ColumnTransformer`, `StandardScaler`, `TargetEncoder`, `RandomizedSearchCV`, `StackingClassifier`, `permutation_importance`), XGBoost, LightGBM, CatBoost, imbalanced-learn (SMOTE), matplotlib/seaborn, scipy (Chi-square test).

## 12. Tài liệu tham khảo chính

- He & Garcia (2009) — *Learning from Imbalanced Data*, IEEE TKDE.
- Davis & Goadrich (2006) — *The Relationship between Precision-Recall and ROC Curves*, ICML.
- Chawla et al. (2002) — *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR.
- Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System*, KDD.
- Ke et al. (2017) — *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS.
- Prokhorenkova et al. (2018) — *CatBoost: Unbiased Boosting with Categorical Features*, NeurIPS.
- Wolpert (1992) — *Stacked Generalization*, Neural Networks.

*(Danh sách đầy đủ 17 tài liệu tham khảo xem trong báo cáo PDF.)*
