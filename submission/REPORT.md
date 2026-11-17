# Báo cáo Day 08 — 2A202603204 - Mai Quang Khải

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào).

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| `person`, `car`, `bus`, `truck` | `YOLO26m` (`yolo26m.pt`) | `0.30` (mặc định) | `r01`, `r02`: YOLO26m bắt tốt các lớp giao thông phổ biến (`0.29s/ảnh`). Trên 4 ảnh gold (`threshold_sweep.md`), `person` đạt recall tối đa `86%` (21 vật) và `car` đạt `88%` (8 vật) ở ngưỡng `<= 0.50`, còn `bus` (1 vật) và `truck` (0 vật) có mẫu `< 5` nên giữ ngưỡng mặc định `0.30`. |
| `bicycle` | `YOLO26m` (`yolo26m.pt`) | `0.20` | `r01` + `threshold_sweep.md`: ở ngưỡng `0.20`, `bicycle` đạt recall `100%` (`5/5` vật gold, `>= 90%`) với precision `71%` (`5/7` box). |
| `motorbike` | `YOLO26m` (`yolo26m.pt`) | `0.10` | `r01`, `r02` + `threshold_sweep.md`: nâng ngưỡng lên `0.50` (`r02`) làm mất `42%` số xe máy (`55` xuống `32` box trên pilot); ngưỡng `0.10` giữ được recall `100%` (`8/8` vật gold) với precision `44%`. |
| `three_wheeler` | `YOLOE-26s` (`yoloe-26s-seg.pt`, prompt: `"tuk-tuk"`, `"cyclo"`, `"cargo tricycle"`) | `0.15` | `r03` + `threshold_sweep.md`: YOLO26 hoàn toàn mù lớp `three_wheeler`; YOLOE-26s tìm đúng tuk-tuk/ba gác (`0.11s/ảnh`) và tại ngưỡng `0.15` đạt recall `100%` (`7/7` vật gold) với precision `78%` (`7/9` box). |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  ```bash
  python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "bicycle=0.2,motorbike=0.1,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json
  ```
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: Chọn **SAM 2.1 base (`sam2.1_b.pt`, 154 MB)** làm mô hình chính cho cả pipeline box-to-mask (`r04`) và interactive point-prompting ở Part 5, đồng thời đối chiếu trực tiếp với **SAM 1 (`mobile_sam.pt`, 39 MB)**. Trên GPU RTX 3050 Laptop (4 GB VRAM), `sam2.1_b.pt` chạy mất `1.22s/ảnh` (`r04`), cho biên polygon liền mạch (1 polygon duy nhất trên `img_013.jpg`, bám sát vòm mui và bánh xe tuk-tuk), không cần xin quyền gated HuggingFace như SAM 3 và ít bị vỡ mảnh hơn `mobile_sam.pt` (2 polygon).
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào?
  - **Lỗi sai bản chất danh mục (closed-set blindness):** YOLO26 huấn luyện trên 80 lớp COCO không có lớp `three_wheeler`, nên luôn gán tuk-tuk/xe ba gác thành `car` hoặc `truck` (ví dụ chiếc tuk-tuk ở `img_013.jpg` bị YOLO26 gán là `car` với `conf = 0.42` trong `r04`). Đổi ngưỡng chỉ làm mất box hoặc giữ box sai lớp chứ không bao giờ biến `car` thành `three_wheeler`.
  - **Cách sửa:** Ghép đa mô hình qua `merge_prelabels.py`, đặt kết quả open-vocabulary `yoloe_all` đứng **trước** `y26_all` với `--iou 0.6` để box `three_wheeler` của YOLOE ghi đè box `car`/`truck` của YOLO26 khi chồng lấn.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - **Điểm cao mà sai:** Trong `img_022.jpg` (chợ đêm), YOLO26 dự đoán box `#272` lớp `person` với độ tin cậy khá cao **`conf = 0.62`** nhưng thực tế là thùng carton/sạp hàng đặt dưới mái bạt; hoặc trong `img_003.jpg` (bẫy #1), YOLOE dự đoán chiếc xe tải 4 bánh ở xa là `three_wheeler` với `conf >= 0.50`.
  - **Điểm thấp mà đúng:** Trong `img_003.jpg`, chiếc xe tải nhỏ ở xa (`#15`) được YOLO26 phát hiện đúng lớp `truck` ở **`conf = 0.25`**; hoặc trong `img_013.jpg`, xe máy đỗ phía sau tuk-tuk (`#132`) đúng lớp `motorbike` ở **`conf = 0.21`**.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: Việc dùng 4 ảnh `gold_manual.json` (50 đối tượng không tính `ignore_region`) để quét ngưỡng ở Part 3 rồi lại dùng chính 4 ảnh đó để báo cáo Precision (`60.8%`) / Recall (`90.0%`) ở Part 4 là đánh giá **in-sample** trên cỡ mẫu rất nhỏ (`n = 4` ảnh, một số lớp như `bus` chỉ có 1 vật, `truck` có 0 vật). Do đó, Recall `90.0%` trên gold là con số lạc quan hơn hiệu năng thực tế trên 24 ảnh batch.

## 2. AI error profile (từ `qc/review_stats.md`)

*(Đã loại 27 đối tượng nằm quanh các vùng bẫy nhân tạo trên 24 ảnh batch để đánh giá khách quan chất lượng máy).*

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---:|---:|---|
| `accepted` | 460 | `460/621 = 74.1%` | `621 box máy` |
| `edited` | 48 | `48/621 = 7.7%` | `621 box máy` |
| `deleted` | 107 | `107/621 = 17.2%` | `621 box máy` |
| `added` | 49 | `49/558 = 8.8%` | `558 đối tượng sau duyệt` |

- Trong số `deleted + added`, có **`11` trường hợp là đổi lớp** (box máy đúng vị trí với $\text{IoU} \ge 0.5$ nhưng sai nhãn nên hợp đồng chấm tính là `1 deleted + 1 added`). Do đó số vật máy **bỏ sót thật** là $49 - 11 = 38 / 558 = 6.8\%$ đối tượng sau duyệt.
- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?
  - **`three_wheeler` (do YOLOE-26s sinh ở ngưỡng `0.15`):** Tệ nhất về tỉ lệ box thừa và lỏng — chỉ có `7/38 = 18.4%` box được `accepted`, trong khi `8/38 = 21.1%` phải `edited` (kéo lại biên ở `img_008.jpg`, `img_013.jpg`, `img_023.jpg`) và tới **`23/38 = 60.5%` bị `deleted`** (do YOLOE ở ngưỡng `0.15` sinh nhiều box trùng lặp trên cùng một tuk-tuk hoặc bắt nhầm xe tải nhỏ/xe máy có mui bạt ở `img_003.jpg`, `img_010.jpg`, `img_028.jpg`).
  - **`motorbike` (do YOLO26m sinh ở ngưỡng `0.10`):** Có số lượng box bị xoá tuyệt đối cao nhất (**`53/233 = 22.7%` `deleted`** và **`31/233 = 13.3%` `edited`**) vì ngưỡng `0.10` giữ lại nhiều mảnh xe máy nhỏ/chồng lấn ở các ảnh đông (`img_005.jpg`, `img_018.jpg`, `img_021.jpg`, `img_024.jpg`, `img_028.jpg`, `img_029.jpg`).
  - **`car` (YOLO26m ở ngưỡng `0.30`):** Bỏ sót nhiều nhất theo tỉ lệ sau duyệt (**`18/60 = 30.0%` `added`**) do nhiều ô tô ở xa hoặc bị che khuất một phần trong luồng giao thông (`img_018.jpg`, `img_025.jpg`, `img_033.jpg`) có điểm tin cậy `< 0.30`.
- So với gold: precision / recall của **máy** (nguồn: `submission/qc/review_stats.md`):
  - **Precision của máy:** `45/74 = 60.8%` (KTC 95% Wilson: `49–71%`, mẫu số: `74 box máy dự đoán trên 4 ảnh gold`, gồm `45 TP + 29 FP`).
  - **Recall của máy:** `45/50 = 90.0%` (KTC 95% Wilson: `79–96%`, mẫu số: `50 đối tượng gold` trên 4 ảnh `img_001.jpg`, `img_012.jpg`, `img_020.jpg`, `img_036.jpg`, gồm `45 TP + 5 FN`).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Kết quả bắt bẫy (từ `submission/qc/review_stats.md`):**
  - **Lượt duyệt mù đầu tiên (trước khi đối chiếu đáp án):** Bắt được **`7/12 = 58.3%`** (`extra`: `3/3`, `loose`: `2/3`, `wrong_class`: `1/3`, `missing`: `1/3`).
  - **Sau khi rà soát lại 3 lỗi sót thật (`img_004.jpg`, `img_005.jpg`, `img_018.jpg`) và giữ nguyên 2 ca đúng thực tế (`img_003.jpg`, `img_034.jpg`):** Script ghi nhận **`10/12 = 83.3%`** (KTC 95%: `55–95%`, đạt ngưỡng `>= 80%`), gồm: `wrong_class: 3/3`, `extra: 3/3`, `missing: 2/3`, `loose: 2/3`.
- **Phân tích chi tiết các bẫy lọt và 2 ca lệch chuẩn tham chiếu (`Trap 1` và `Trap 5`):**
  1. **Trap 1 — `img_003.jpg` (`missing`, máy gán `orig_cls = three_wheeler` tại `[74.4, 176.3, 45.6×50.3]`) — Giữ nguyên vì nhãn người gán mới là chân lý:**
     - Ở vị trí góc trên bên trái đường (`[73.9, 176.1, 47.1×51.4]`), phương tiện thực tế là một **xe tải thùng 4 bánh (`truck`)** chạy ở xa. Tuy nhiên, YOLOE-26s đã dự đoán nhầm xe này thành `three_wheeler` với `score >= 0.50`, và `inject_traps.py` xoá box đó đi để tạo bẫy `missing`.
     - Khi duyệt, tôi **đã phát hiện đúng chiếc xe bị thiếu** và vẽ lại box khớp tới **$\text{IoU} = 0.948$**, nhưng gán nhãn đúng thực tế là **`truck` (`#15`)**. Vì `review_stats.py` kiểm tra điều kiện cứng `f.cls == "three_wheeler"`, script báo "lọt bẫy" dù người duyệt đã bắt được vật bị xoá và sửa luôn lỗi sai lớp của YOLOE.
  2. **Trap 5 — `img_034.jpg` (`loose`, máy gán `motorbike` tại `orig_box = [100.6, 421.6, 45.1×49.4]`) — Giữ nguyên theo đúng Guideline v1.0 mục 4.1.2:**
     - Vị trí này là phần đầu một chiếc xe máy đỗ trên vỉa hè trái nhưng bị **cột gạch che khuất $> 80\%$ diện tích**. Khi `inject_traps.py` làm lệch box thành `bad_box = [111.1, 413.2, 34.8×54.4]`, thay vì kéo box lại cho một mảnh xe bị che $> 80\%$, tôi đã **xoá box** theo đúng quy tắc mục 4.1.2 (*"Vật bị che > 80% (còn nhìn thấy < 20% diện tích) thì không gán"*). Script chấm bẫy đòi phải giữ lại box `motorbike` có $\text{IoU} \ge 0.75$ với `orig_box` nên báo lọt.
  3. **Ba bẫy lọt ở lượt đầu do thiên kiến tự động hóa (đã khắc phục):**
     - `img_004.jpg` (`missing` — `person` tại `[452.8, 242.6, 52.2×239.1]`): Người đàn ông mặc áo đen đứng lọt giữa khe tối của hai chiếc tuk-tuk màu sáng; do làm chưa kỹ **Bước 0 (Quét ảnh trước)** nên mắt bị hút vào các khung tuk-tuk lớn của máy.
     - `img_005.jpg` (`wrong_class` — `person` $\rightarrow$ `bicycle` tại `[0.3, 262.9, 153.4×369.2]`): Tôi đã phát hiện và xoá box `bicycle` sai ở mép trái, đồng thời vẽ lại xe máy và chân người ngồi, nhưng lại quên vẽ lại box `person` bao trọn phần thân trên của người lái áo đen.
     - `img_018.jpg` (`wrong_class` — `person` $\rightarrow$ `bicycle` tại `[334.5, 445.7, 19.6×39.4]`): Ảnh quảng trường có hơn 60 đối tượng nhỏ; khi duyệt liên tục không dùng bộ lọc theo lớp ở **Bước 3 (Sai nhãn)**, một box người đi bộ nhỏ bị đổi thành `bicycle` đã lọt qua mắt.
- **Accepted cao (`74.1%`) mà bắt bẫy lượt đầu thấp (`58.3%`) nói gì về cách duyệt?**
  - Con số này chứng minh rõ hiệu ứng **thiên kiến tự động hóa (automation bias)**: khi gần 3/4 số box của máy (`460/621`) đã ôm khít vật thật, người duyệt rất dễ chuyển sang trạng thái "kiểm tra thụ động theo box có sẵn" (xoá rất tốt `3/3` box thừa `extra` vì nhìn vào box thấy trống), nhưng lại suy giảm khả năng phát hiện **vật bị sót (`missing`)** ở vùng tối hoặc **nhãn sai (`wrong_class`)** trên các box nhỏ có hình dáng hợp lý.

## 4. Máy có đáng dùng không?

- **Thời gian (từ `submission/time_log.csv`):**
  - Gán tay từ đầu (`gold_manual`): `30 phút / 4 ảnh = 7.50 phút/ảnh` (`53` đối tượng $\approx 34.0\text{ giây/đối tượng}$).
  - Duyệt có pre-label (`batch_review`): `103 phút / 24 ảnh = 4.29 phút/ảnh` (`580` đối tượng $\approx 10.7\text{ giây/đối tượng}$).
  - **Tiết kiệm:** Giảm **`42.8%` thời gian theo ảnh** (`3.21 phút/ảnh`) và giảm **`68.5%` thời gian theo mỗi đối tượng**.
  - **Thiên lệch cần lưu ý:** Vì tôi gán tay 4 ảnh gold trước nên đã thuộc bảng 8 lớp, phím tắt CVAT và quy tắc `ignore_region` trước khi bước vào lô 24 ảnh batch; do đó một phần tốc độ tăng thêm đến từ hiệu ứng quen tay chứ không hoàn toàn do AI.
- **Kết luận:** **Có đáng dùng, nhưng chỉ hiệu quả theo từng lớp:**
  - **Rất đáng dùng cho `person` (`255/270 = 94.4%` box máy được giữ nguyên) và `car` (`41/55 = 74.5%` box máy được giữ nguyên):** YOLO26m định vị cực kỳ chuẩn xác, giúp tiết kiệm phần lớn công vẽ tay.
  - **Đáng dùng có điều kiện cho `motorbike` (`149/233 = 63.9%` accepted):** Cần nâng ngưỡng từ `0.10` lên khoảng `0.20–0.25` để giảm 53 box rác phải xoá tay.
  - **Cần tinh chỉnh lại cho `three_wheeler` (`7/38 = 18.4%` accepted, `23/38 = 60.5%` deleted):** YOLOE-26s giúp cảnh báo ảnh nào có tuk-tuk/ba gác (tránh bỏ sót), nhưng ở ngưỡng `0.15` sinh quá nhiều box trùng/lỏng, tốn công xoá và kéo lại.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Quy tắc 1 (Chống sót vật - Bước 0):** Bắt buộc bấm phím `H` (hoặc biểu tượng con mắt trên CVAT) ẩn toàn bộ nhãn máy trong **15 giây đầu mỗi ảnh**, đếm nhẩm số lượng `three_wheeler` và người ở tiền cảnh trước khi bật nhãn máy lên (kiểm chứng: tỉ lệ bắt bẫy `missing` phải đạt `100%` trên các vật lộ diện `>= 20%`).
  2. **Quy tắc 2 (Chống sai nhãn đám đông - Bước 3):** Ở mọi ảnh có `> 25 đối tượng`, sau khi duyệt xong phải bật **CVAT Filters** lọc riêng từng lớp hiếm/dễ nhầm (`label == "bicycle"`, `label == "truck"`, `label == "three_wheeler"`) để kiểm tra 100% các box của lớp đó trong `< 20 giây` (chặn triệt để lỗi `person` bị gán nhầm thành `bicycle` như ở `img_005.jpg`, `img_018.jpg`).
  3. **Quy tắc 3 (Chống mệt mỏi và box rác `< 20 px`):** Cứ sau mỗi **8 ảnh** (hoặc **30 phút** làm việc) phải nghỉ mắt **3 phút**, đồng thời quét xoá toàn bộ các box máy đơn lẻ có chiều cao `< 20 px` ở hậu cảnh thay vì để mặc định (kiểm chứng bằng `validate_labels.py --all` đạt `0 cảnh báo < 20 px` và `time_log.csv` có ghi nhịp nghỉ).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---:|---|---|---|
| `img_013.jpg` | `SAM 2.1 base` (`sam2.1_b.pt`) | `3` (`2 pos`, `1 neg`) | Còn dính đầu, quần và bàn chân người lái (chỉ khoét được phần lưng áo quanh điểm âm `720,300`). | Mask rộng (`317,262 px = 45.4%` ảnh, `1 polygon`), bám rất sát mui và bánh xe nhưng tràn nhẹ sang bánh trước xe máy đỗ sát bên phải. | `tuktuk_img013_sam2.jpg` |
| `img_013.jpg` | `SAM 1 MobileSAM` (`mobile_sam.pt`) | `3` (`2 pos`, `1 neg`) | Đã loại trừ được toàn bộ thân áo và quần người lái, chỉ còn sót vùng má và bàn chân. | Mask gọn hơn (`245,493 px = 35.2%` ảnh, `2 polygon`), không tràn sang xe máy bên cạnh nhưng bị thủng lỗ ở phần đệm ghế và khung sườn. | `tuktuk_img013_sam1.jpg` |
| `img_004.jpg` | `SAM 2.1 base` (`sam2.1_b.pt`) | `3` (`2 pos`, `1 neg`) | Không dính người đàn ông áo đen đứng cạnh (`465,450`) và không dính người mặc áo xanh bên phải. | Mask rất sạch (`93,191 px = 13.3%` ảnh, `1 polygon`), ôm trọn thân, mui và 3 bánh của chiếc tuk-tuk xanh-hồng ở tiền cảnh. | `tuktuk_img004_sam2.jpg` |
| `img_017.jpg` | `SAM 2.1 base` (`sam2.1_b.pt`) | `3` (`2 pos`, `1 neg`) | Không có người lái trên xe; mask bao gồm cả ghế da, kính chắn gió và khung mui. | Mask sắc nét (`84,531 px = 12.1%` ảnh, `1 polygon`), tách rạch ròi chiếc tuk-tuk tiền cảnh khỏi chiếc tuk-tuk màu xanh đỗ sát phía sau. | `tuktuk_img017_sam2.jpg` |

- **Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu?**
  - **Quy tắc giúp ích rõ rệt:** Mục 4.2.2 (*"không vẽ lỗ nếu lỗ nhỏ hơn ~5% diện tích vật"*) và Mục 4.2.3 (*"Bóng đổ không thuộc vật. Kính xe thuộc vật"*) giúp quyết định giữ nguyên polygon liền khối của `sam2.1_b.pt` trên `img_004.jpg` và `img_017.jpg` mà không cần đục thủng các khe thoáng nhỏ ở khung sắt hay kính chắn gió.
  - **Quy tắc còn thiếu (đã bổ sung vào v1.1):** Guideline v1.0 chưa nêu rõ khi người lái ngồi bên trong khung xe hở (`img_013.jpg`) thì cần bao nhiêu điểm click âm hoặc có bắt buộc dùng `--box` kết hợp `--neg` trên đầu/chân người lái để tách hoàn toàn `person` khỏi mask `three_wheeler` hay không. Thực nghiệm trên `img_013.jpg` cho thấy chỉ 1 điểm âm `(720,300)` trên lưng áo là chưa đủ với `sam2.1_b.pt` (cần 2–3 điểm âm ở đầu, lưng và chân).
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Container `cvat_sam3_bridge` đang chạy trên hệ thống Docker local nhưng Part 5 được thực hiện chuẩn xác và tái lập 100% qua CLI `scripts/sam2_points.py` (cho phép cố định toạ độ pixel `--pos` / `--neg` để so sánh công bằng giữa `sam2.1_b.pt` và `mobile_sam.pt` trên cùng một điểm click).

## 6. Calibration (Part 6)

- **Số quyết định:** **`5` quyết định** (`CAL-01` đến `CAL-05` trong `submission/qc/calibration_decisions.csv`), bao phủ đầy đủ các nguồn: `loi_may` (`CAL-01`), `cau_hoi_mo` (`CAL-02`, `CAL-03`, `CAL-04`) và `can_review` (`CAL-05`).
- **Quyết định quan trọng nhất:**
  - **`CAL-01` (`img_003.jpg|img_013.jpg`, nguồn `loi_may`):** Chốt ranh giới giữa `three_wheeler` và `truck` khi hai mô hình xung đột (YOLO26 gọi tuk-tuk là `car`/`truck`, còn YOLOE ở ngưỡng `0.15` lại gọi xe tải thùng 4 bánh ở xa trong `img_003.jpg` #15 là `three_wheeler`).
  - **`CAL-03` (`img_018.jpg|img_034.jpg`, nguồn `cau_hoi_mo`):** Chốt quy tắc xoá bỏ các box máy đơn lẻ có chiều cao `< 20 px` (`img_034.jpg` #545) hoặc bị vật cản như cột gạch che khuất `> 80%` diện tích (`img_034.jpg` bẫy #5), và gom cụm `>= 5` xe nhỏ dính sát nhau thành một `ignore_region`.
- **Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:**
  1. Phân biệt xe đạp điện có bàn đạp (`bicycle`) và xe máy điện/scooter có sàn để chân (`motorbike`) (`CAL-02`).
  2. Phân biệt `three_wheeler` và `truck` ở xa theo cấu trúc cabin kín + 4 bánh (`CAL-01`).
  3. Xoá box đơn lẻ `< 20 px` hoặc lộ diện `< 20%` sau vật cản; gom cụm `>= 5` xe thành `ignore_region` (`CAL-03`).
  4. Không gộp quang gánh/hàng nhô `> 20%` vào box `person`/`bicycle`; kính chắn gió thuộc mask xe nhưng phải đặt click âm tách người lái (`CAL-04`).
  5. Không đặt ngưỡng `conf` tự động chấp nhận; bắt buộc kiểm mắt vùng tối dưới mái bạt chợ đêm để xoá false positive trên thùng hàng (`CAL-05`).
- **Nguyên nhân gốc:** Kết hợp cả **3 nguyên nhân**:
  - *Guideline v1.0 cố ý để mở:* Chưa định nghĩa xe đạp điện, quang gánh, kính xe trong mask (`CAL-02`, `CAL-04`).
  - *Lỗi đặc trưng của mô hình (máy):* YOLO26 mù lớp `three_wheeler` và bắt nhầm thùng hàng ở `conf = 0.62`; YOLOE nhạy quá mức ở ngưỡng `0.15` (`CAL-01`, `CAL-05`).
  - *Người duyệt:* Bỏ sót một số box máy `< 20 px` ở hậu cảnh (`img_018.jpg`, `img_034.jpg`) và box giả `#272` dưới mái bạt (`img_022.jpg`) do mệt mỏi khi duyệt lô.

## 7. QC và quyết định (Part 6)

- **Mẫu:** **`7/24` ảnh batch** (**`2` ảnh `can_review`** là `img_014.jpg`, `img_022.jpg` + **`5` ảnh ngẫu nhiên** với `seed 42` là `img_003.jpg`, `img_006.jpg`, `img_013.jpg`, `img_017.jpg`, `img_034.jpg`), tổng cộng **`125` đối tượng** (nguồn: `submission/qc/qc_sample.csv` và `submission/qc/qc_report_auto.md`). Người kiểm: chính tôi (**hạn chế:** người kiểm đồng thời là người duyệt nên dễ lặp lại cùng một điểm mù nhận thức).
- **Tỉ lệ lỗi (kèm mẫu số):**
  - Lỗi theo **đối tượng**: **`2/125 = 1.6%`** (mẫu số: `125 đối tượng` trong 7 ảnh mẫu).
  - Lỗi theo **ảnh**: **`2/7 = 28.6%`** (mẫu số: `7 ảnh` trong mẫu QC; ngưỡng Release quy định `<= 2%`).
- **Top lỗi + ảnh ví dụ mỗi loại (từ `submission/qc/qc_review.csv`):**
  1. **Thừa (`thua` — 2 lỗi trên 2 ảnh):**
     - `img_022.jpg` (đối tượng `#272`, `person`, `conf = 0.62`): Box của YOLO26 bắt nhầm chồng thùng carton/sạp hàng dưới mái bạt chợ đêm chưa được xoá khi review.
     - `img_034.jpg` (đối tượng `#545`, `motorbike`, `conf = 0.22`): Box xe máy ở hậu cảnh xa có chiều cao `17.4 px < 20 px` (cùng `#547` cao `18.3 px`), vi phạm ngưỡng chiều cao tối thiểu `min_box_height_px = 20` (lẽ ra phải xoá hoặc gom vào `ignore_region`).
- **Giới hạn mẫu nhỏ (cận trên 95% trong `qc_report_auto.md`):**
  - Với `2/7` ảnh có lỗi (`28.6%`), **cận trên khoảng tin cậy 95% (Wilson) của tỉ lệ ảnh lỗi thật lên tới `64%`**.
  - Quan trọng hơn, ngay cả khi mẫu 7 ảnh đạt **0 lỗi (`0/7 = 0.0%`)**, theo quy tắc thống kê mẫu nhỏ ($3/n$ hoặc Wilson 95%), cận trên 95% vẫn là **`~35%`**. Do đó, một mẫu QC chỉ gồm 7 ảnh (hay toàn bộ lô 24 ảnh) **không bao giờ đủ cỡ mẫu thống kê để chứng minh tỉ lệ ảnh lỗi thực sự $\le 2\%$**; mẫu nhỏ chỉ giúp phát hiện nhanh các lỗi nổi cộm để sửa quy trình.
- **Script kiểm được / không kiểm được:**
  - *Kiểm được:* Cấu trúc COCO 1.0, đủ 4 ảnh gold + 24 ảnh batch, tên lớp đúng `taxonomy.json`, toạ độ box nằm trong khung ảnh, cảnh báo box cao `< 20 px` (`validate_labels.py`), thống kê ghép cặp IoU `accepted/edited/deleted/added`, tỉ lệ bắt 12 bẫy đã gài (`review_stats.py`) và tính cận trên Wilson từ phiếu `qc_review.csv` (`qc_report.py`).
  - *Không kiểm được:* Box có thực sự ôm sát vật thể đến `<= 2 px` hay không, một vật ngoài đời là `truck` hay `three_wheeler` (như bẫy #1 ở `img_003.jpg`), một box có phải là thùng hàng bị nhận nhầm thành người hay không (`#272` ở `img_022.jpg`), những vật mà **cả máy lẫn người cùng bỏ sót** trên 24 ảnh batch, và việc người duyệt có thực sự ẩn nhãn máy ở Bước 0 hay không.

**Quyết định: Rework**, ngày `2026-11-17`.
- **Lý do (đi theo đúng cây quyết định Hold $\rightarrow$ Rework $\rightarrow$ Release):**
  1. *Kiểm điều kiện **Hold**:* Không Hold vì toàn bộ 7 câu hỏi mở của guideline v1.0 đã được chốt rõ trong `guideline_v1.1_calibrated.md` (`--guideline-gap 0`) và bộ 36 ảnh gốc không có ảnh hỏng (`--data-problem 0`).
  2. *Kiểm điều kiện **Rework**:* Tỉ lệ bắt bẫy đạt `10/12 = 83.3% >= 80%`, nhưng kiểm mẫu QC phát hiện **`2/7 = 28.6%` ảnh có lỗi (`> 2%` ngưỡng `error_rate_img_max`)**, cụ thể là lỗi giữ lại box máy thừa (`#272` trên sạp hàng ở `img_022.jpg`) và các box xe máy ở xa có chiều cao `< 20 px` (`img_018.jpg`, `img_027.jpg`, `img_033.jpg`, `img_034.jpg`). Vì `28.6% > 2%`, lô nhãn bắt buộc phải dừng ở trạng thái **Rework** chứ không được báo Release.
- **Nếu Rework / Hold, việc cần làm tiếp:**
  1. Mở lại Task `batch-review` trên CVAT, xoá box giả `#272` trong `img_022.jpg`.
  2. Rà soát 4 ảnh có cảnh báo chiều cao `< 20 px` từ `validate_labels.py` (`img_018.jpg`, `img_027.jpg`, `img_033.jpg`, `img_034.jpg`) để xoá các box đơn lẻ `< 20 px` hoặc gom cụm xe ở xa vào `ignore_region` theo đúng quy tắc `CAL-03` của v1.1.
  3. Xuất bản vá ra `submission/labels/qc_fixed.json`, chạy lại `python scripts/build_release.py` và kiểm tra lại mẫu QC vòng 2.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào):
  - Sử dụng trợ lý **Antigravity (Gemini)** để hỗ trợ kiểm tra môi trường Python/Docker, khởi tạo Task trên CVAT local qua REST API, chạy các script CLI (`prelabel.py`, `threshold_sweep.py`, `merge_prelabels.py`, `inject_traps.py`, `review_stats.py`, `sam2_points.py`, `build_release.py`, `qc_sample.py`, `qc_report.py`), trực quan hoá toạ độ các bẫy sau khi đã khóa bản export và rà soát định dạng Markdown/CSV.
  - **Cách kiểm tra:** Mọi nhãn trong `gold_manual.json` và `batch_reviewed.json` đều do chính tôi trực tiếp gán và duyệt trên giao diện CVAT (`http://127.0.0.1:8080`); mọi số đo trong báo cáo được đối chiếu trực tiếp với đầu ra của `validate_labels.py`, `threshold_sweep.md`, `review_stats.md` và `qc_report_auto.md`.
- **Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)?**
  - **Không.** Tập `gold_manual.json` (4 ảnh `img_001.jpg`, `img_012.jpg`, `img_020.jpg`, `img_036.jpg`, 53 đối tượng) được gán tay 100% trên CVAT ở Part 1 (từ 17:02 đến 17:32) **trước khi** chạy bất kỳ lệnh `prelabel.py` nào ở Part 2.
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt:**
  - Hệ điều hành Linux (Ubuntu), môi trường ảo Python tại `/mnt/samsung/models/.venv`, GPU **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, CUDA 13.2)**, CVAT Docker **v2.81.1**.
  - **Vướng mắc & cách xử lý:**
    1. Traefik của CVAT Docker được cấu hình nhận diện theo header `Host: 127.0.0.1`, nên gọi `http://localhost:8080/api/server/about` trả về `404 Not Found` trong khi `http://127.0.0.1:8080/api/server/about` trả về `200 OK` (ghi chú theo yêu cầu của `docs/CVAT_AI.md`).
    2. API export của CVAT v2.81.1 đã bỏ endpoint cũ `GET /api/tasks/{id}/dataset` (`410 Gone`) và chuyển sang `POST /api/tasks/{id}/dataset/export` kèm polling `/api/requests/{rq_id}`.
- **Dùng cache `data/cache/` thay vì tự chạy?**
  - **Không.** Toàn bộ các lượt chạy YOLO26m, YOLOE-26s và SAM (`sam2.1_b.pt`, `mobile_sam.pt`) đều được chạy trực tiếp trên GPU RTX 3050 của máy cá nhân.
- **Lỗi / bug script đã gặp (nếu có):**
  - Không có bug trong các script của repo (`scripts/*.py`). Tuy nhiên phát hiện một hạn chế thiết kế rất thú vị ở `inject_traps.py` + `review_stats.py`: script tạo bẫy `missing` / `loose` từ chính box dự đoán của máy (khi `score >= 0.50`) mà không biết box đó của máy đúng hay sai lớp/quy tắc. Vì vậy ở `img_003.jpg` (bẫy #1), khi YOLOE gán nhầm xe tải 4 bánh là `three_wheeler` rồi bị xoá làm bẫy `missing`, người duyệt vẽ lại đúng vị trí ($\text{IoU} = 0.948$) với nhãn đúng `truck` thì script lại chấm là "lọt bẫy".
- **Một điều bạn làm khác lần sau:**
  - Trước khi export từ CVAT, tôi sẽ dùng bộ lọc chiều cao hoặc rà soát kỹ các box ở đường chân trời để xoá/gom hết các box `< 20 px` vào `ignore_region`, đồng thời lọc riêng từng lớp (`bicycle`, `three_wheeler`, `person`) để bắt sạch các false positive dưới mái bạt chợ đêm ngay từ vòng duyệt đầu tiên.
