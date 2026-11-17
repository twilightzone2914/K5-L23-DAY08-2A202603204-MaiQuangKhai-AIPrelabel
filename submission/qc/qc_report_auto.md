# Báo cáo QC tự động (nháp)

> **Gợi ý** do script tính; bạn xác nhận. Script chỉ đếm theo phiếu kiểm bằng mắt người.

## Mẫu
- 7 ảnh (2 ca khó + 5 ngẫu nhiên), 125 đối tượng.
- taxonomy v1.0

## Số đo
| Chỉ số | Giá trị | Mẫu số |
|---|---|---|
| Lỗi **theo đối tượng** | 2/125 = 1.6% | 125 đối tượng trong mẫu |
| **Ảnh có lỗi** (dùng cho ngưỡng) | 2/7 = 28.6% | 7 ảnh trong mẫu |
| Máy: accepted / edited / deleted | 460/621 = 74.1% / 48/621 = 7.7% / 107/621 = 17.2% | 621 box máy |
| Máy: added | 49/558 = 8.8% | 558 đối tượng sau duyệt |
| Bắt lỗi bẫy | 10/12 = 83.3% | 12 bẫy |

## Top 3 loại lỗi
| Loại lỗi | Số lỗi | Ảnh ví dụ |
|---|---|---|
| thua | 2 | img_022.jpg, img_034.jpg |

## Giới hạn thống kê
- 2 ảnh lỗi trong 7 ảnh mẫu: cận trên 95% (Wilson) của tỉ lệ ảnh lỗi thật là **64%**. Ngưỡng 2% không thể được **chứng minh** bằng mẫu nhỏ như vậy: hãy nói rõ giới hạn này trong báo cáo.

## Gợi ý quyết định
**REWORK**

- ảnh có lỗi 28.6% > 2%

Cây quyết định: Hold (guideline chưa rõ / dữ liệu hỏng) -> Rework (> ngưỡng, lỗi lặp, bắt bẫy / nhất quán thấp) -> Release.
