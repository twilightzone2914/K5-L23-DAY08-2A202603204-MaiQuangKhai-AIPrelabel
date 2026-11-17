# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603204 - Mai Quang Khải

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Bản v1.1 chốt toàn bộ 7 câu hỏi mở ở mục 6 của v1.0 và bổ sung quy tắc xử lý lỗi đặc trưng của YOLO26, YOLOE-26s và SAM 2.1.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Phân loại theo khung sườn và bàn đạp: có bàn đạp + khung ống mảnh $\rightarrow$ `bicycle`; có sàn để chân liền khối, vỏ yếm nhựa và gắn biển số $\rightarrow$ `motorbike`. | `img_021.jpg`, `img_028.jpg` | `CAL-02` |
| 2 | Khối xe máy đỗ dày đặc gần/xa camera: từng xe hay `ignore_region`? | Nếu từng xe cao $\ge 20\text{ px}$ và nhìn rõ $\ge 20\%$ thân xe (tách được yên/đầu/bánh) thì gán từng `motorbike` (như hàng xe tiền cảnh `img_006.jpg`, `img_034.jpg`). Nếu cụm $\ge 5$ xe dính chặt ở xa hoặc cao $< 20\text{ px}$ không tách được cá thể $\rightarrow$ bao bằng một `ignore_region`. | `img_006.jpg`, `img_018.jpg`, `img_034.jpg` | `CAL-03` |
| 3 | Hàng nhô ra (tre, rổ, quang gánh): box/mask có gồm hàng không? | Giỏ/thùng gắn cố định vào khung xe (`img_003.jpg`, `img_006.jpg`) tính vào box/mask xe; hàng rời nhô quá $20\%$ chiều dài xe hoặc quang gánh/đòn gánh của người bán hàng rong **không** gộp vào box `person` (chỉ ôm thân người). | `img_003.jpg`, `img_006.jpg`, `img_014.jpg` | `CAL-04` |
| 4 | Người ngồi trong xe ba bánh có mui che / vật bị che khuất: ngưỡng $\ge 20\%$ đánh giá thế nào? | Gán `person` khi nhìn thấy rõ **đầu + vai** hoặc **nửa thân trên** (tương đương $\ge 20\%$ diện tích cơ thể, ví dụ người lái và khách trên `img_003.jpg`, `img_013.jpg`). Nếu chỉ lộ bóng mờ hoặc bị cột/tường che $> 80\%$ (như đầu xe máy sau cột gạch ở `img_034.jpg`) thì **không gán** (xoá box máy nếu có). | `img_003.jpg`, `img_013.jpg`, `img_034.jpg` | `CAL-03` |
| 5 | Kính chắn gió / cửa sổ xe có thuộc mask xe không? | **Có.** Kính chắn gió và khung mui thuộc mask của `three_wheeler` / `car` / `bus`; tuy nhiên phần cơ thể người lái hoặc hành khách nhìn thấy trực tiếp qua hông xe hở phải được loại trừ khỏi mask xe bằng điểm click âm (`--neg`). | `img_004.jpg`, `img_013.jpg`, `img_017.jpg` | `CAL-04` |
| 6 | Box máy chồng nhau quá nhiều trong ảnh đêm / chợ: gộp hay giữ? | Giữ riêng box cho từng đối tượng thật có thể phân biệt được (`img_022.jpg`); xoá ngay các box trùng lặp (duplicate) do YOLOE/YOLO26 sinh đè lên cùng một xe hoặc bắt nhầm thùng hàng/sạp chợ dưới mái bạt. | `img_022.jpg`, `img_013.jpg` | `CAL-05` |
| 7 | Có đặt ngưỡng `conf` tối thiểu để tự động chấp nhận box máy không cần kiểm mắt không? | **Tuyệt đối không.** Thực nghiệm cho thấy YOLO26 vẫn bắt nhầm thùng hàng dưới mái bạt thành `person` với `conf = 0.62` (`img_022.jpg` #272) và YOLOE gán nhầm xe tải 4 bánh thành `three_wheeler` với `conf >= 0.50` (`img_003.jpg`). Mọi box ở mọi mức `conf` đều phải kiểm bằng mắt qua checklist 4 bước. | `img_003.jpg`, `img_022.jpg` | `CAL-01`, `CAL-05` |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_003.jpg`, `img_013.jpg` | Xe ba gác mái che ở tiền cảnh và xe tải thùng 4 bánh ở xa (`img_003.jpg` #15, bẫy #1). | YOLO26 gọi tuk-tuk (`img_013.jpg`) là `car`; YOLOE (`conf 0.15`) bắt đúng xe ba gác nhưng gọi nhầm xe tải nhỏ 4 bánh ở xa (`img_003.jpg`) là `three_wheeler`. | Gán `three_wheeler` cho xe ba gác/tuk-tuk; sửa nhãn xe tải 4 bánh ở xa (`img_003.jpg` #15) thành `truck` bất kể gợi ý của YOLOE. | **Mục 2 & 4.3 (`three_wheeler` vs `truck`):** Xác nhận số bánh và buồng lái; xe tải thùng 4 bánh có cabin kín luôn là `truck` (`CAL-01`). |
| `img_034.jpg` | Đầu xe máy đỗ khuất sau cột gạch bên trái (`[100.6, 421.6, 45.1×49.4]`, bẫy #5) và xe máy ở rất xa (`#545`, cao `17.4 px`). | YOLO26 sinh box trên phần đầu xe bị che $> 80\%$ sau cột và các box xe nhỏ cao $< 20\text{ px}$ ở cuối phố. | Xoá box xe bị cột che $> 80\%$ và xoá (hoặc gom vào `ignore_region`) các box đơn lẻ có chiều cao $< 20\text{ px}$. | **Mục 4.1.2 & 4.1.6:** Không giữ box máy cho vật lộ diện $< 20\%$ sau vật cản hoặc cao $< 20\text{ px}$ (`CAL-03`). |
| `img_014.jpg`, `img_022.jpg` | Người bán hàng gánh đôi quang gánh (`img_014.jpg` #144), người ở sát mép phải (`#136`, `can_review`), và thùng hàng dưới bạt chợ đêm (`img_022.jpg` #272). | YOLO26 giữ box chuẩn trên thân người gánh hàng nhưng bắt nhầm thùng xốp dưới mái bạt ở `img_022.jpg` thành `person` (`conf = 0.62`). | Giữ box `person` ôm sát cơ thể người (bỏ phần đòn gánh nhô xa), giữ `#136` ở mép ảnh vì cao `84 px >= 20 px`, và xoá box giả `#272` trên thùng hàng. | **Mục 4.1.3, 4.1.5 & 4.1.7:** Quang gánh rời không tính vào box `person`; rà soát kỹ false positive `person` dưới mái bạt chợ đêm (`CAL-04`, `CAL-05`). |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt 7 câu hỏi mở mục 6; bổ sung quy tắc phân biệt `three_wheeler` vs `truck` ở xa, quy tắc loại trừ quang gánh khỏi `person`, quy tắc tách người lái khỏi mask SAM của xe, và quy tắc xoá box `< 20 px` / bị che `> 80%` / false positive chợ đêm. | Đồng bộ quyết định từ `CAL-01`, `CAL-02`, `CAL-03`, `CAL-04`, `CAL-05` sau khi duyệt 24 ảnh batch và kiểm QC 7 ảnh mẫu. |
