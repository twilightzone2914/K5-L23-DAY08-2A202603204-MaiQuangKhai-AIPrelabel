# Thông tin học viên

- Họ và tên: Mai Quang Khải
- MSHV: 2A202603204
- Lớp (L2 / L3 / khác): L23
- Tài khoản GitHub: twilightzone2914
- Repo bài nộp (URL): https://github.com/twilightzone2914/K5-L23-DAY08-2A202603204-MaiQuangKhai-AIPrelabel
- Commit chốt bài (hash): d7c5529
- Máy: Linux (Ubuntu), GPU NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.2)
- CVAT (Docker): v2.81.1 (http://127.0.0.1:8080)

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 460/621 = 74.1% / 48/621 = 7.7% / 107/621 = 17.2% (mẫu số: 621 box máy); added: 49/558 = 8.8% (mẫu số: 558 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 45/74 = 60.8% (mẫu số: 74 box máy) / 45/50 = 90.0% (mẫu số: 50 đối tượng gold)
- Bắt bẫy: 10/12 = 83.3% (lượt đầu 7/12 = 58.3%; 2 bẫy img_003 và img_034 giữ theo nhãn đúng thực tế của người duyệt)
- Thời gian: gán tay 7.50 phút/ảnh (30 phút / 4 ảnh); duyệt có pre-label 4.29 phút/ảnh (103 phút / 24 ảnh)
- Lỗi QC: 28.6% theo ảnh (2/7 ảnh mẫu, cận trên Wilson 95%: 64%), 1.6% theo đối tượng (2/125 đối tượng trong mẫu)
- **Quyết định:** Rework — lý do một dòng: Tỉ lệ ảnh có lỗi trong mẫu QC là 2/7 = 28.6% > ngưỡng Release 2% (còn box thừa #272 trên thùng hàng ở img_022 và box < 20 px #545 ở img_034 cần dọn sạch).
