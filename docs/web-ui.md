# Hướng dẫn Web UI

Chạy server (`python -m retarget.server`) rồi mở http://127.0.0.1:6790/. Trang đầu tiên là **viewer**. Trang **landmark picker** được mở từ viewer khi cần.

Web UI chưa có chức năng tải mô hình lên. Muốn thêm mô hình mới thì import bằng lệnh `python -m retarget.importer <file>`, sau đó tải lại trang viewer (Ctrl+R) để thấy mô hình trong danh sách.

## Viewer

Bên trái là khung 3D: kéo chuột trái để xoay, lăn chuột để zoom, kéo chuột phải để di chuyển. Bên phải là bảng điều khiển gồm các mục dưới đây.

### Model

| Thành phần | Tác dụng |
| --- | --- |
| Danh sách mô hình | Các mô hình trong `web/models/`. `<tên>` là bản convert từ file gốc, `<tên>.rigged` là bản đã được sinh thêm blendshape. `ict_neutral` là khuôn mặt mẫu dùng khi sinh blendshape, không phải mô hình để điều khiển. |
| **Load** | Tải mô hình đang chọn. |
| **Background** | Đổi màu nền: tối, trắng, xanh lá (để tách nền khi quay video). |
| **Reset pose** | Đưa mô hình về mặt trung tính và đặt mọi slider về 0. |
| Extra smoothing | Làm mượt thêm phía trình duyệt. Để 0 vì capture đã có One Euro filter; tăng lên sẽ làm chuyển động trễ hơn. |
| Mirror in viewer | Lật trái phải thêm một lần. Capture đã chạy ở chế độ gương nên thường để tắt. |

### Face rig

Mục này cho biết mô hình đang chọn đã sẵn sàng đến đâu, và chỉ hiện thao tác cần làm tiếp theo. Các nút không cần thiết sẽ không xuất hiện.

| Tình trạng của mô hình | Mục Face rig hiển thị |
| --- | --- |
| VRM, hoặc morph sẵn có đã đủ mọi blendshape | Chỉ thông tin, không có nút |
| Chưa từng sinh blendshape | **Generate blendshapes** |
| Lần trước MediaPipe không tìm được khuôn mặt | Chỉ có **Pick landmarks →** (nút màu cam) |
| Đã có điểm mốc nhưng chưa sinh, hoặc điểm mốc mới hơn kết quả | **Generate blendshapes** và link *Edit landmarks…* |
| Kết quả đã khớp với điểm mốc hiện tại | **Load `<tên>.rigged`** (nếu đang xem bản khác) và link *Edit landmarks…* |
| Đang sinh | Tiến độ; mọi nút bị khoá |

- **Generate blendshapes** chạy trên server, mất khoảng 1 phút. Nếu mô hình chưa có điểm mốc, hệ thống thử tìm khuôn mặt bằng MediaPipe trước. Xong thì viewer tự tải bản `.rigged`. Server chỉ chạy một job tại một thời điểm.
- **Pick landmarks →** mở trang chấm điểm mốc cho mô hình đó.
- *Edit landmarks…* dùng khi blendshape sinh ra trông sai, ví dụ điểm MediaPipe tìm được bị lệch: sửa điểm, lưu, rồi sinh lại.

### Input

Chọn thứ điều khiển mô hình:

- **Live capture**: mô hình làm theo các frame do chương trình capture gửi lên (webcam hoặc video). Lúc này slider bị khoá.
- **Sliders**: tự kéo từng blendshape, góc đầu và hướng nhìn để kiểm tra mô hình mà không cần camera. Frame từ capture bị bỏ qua.

Chuyển về Live capture khi không có capture nào đang chạy thì mô hình trở về mặt trung tính.

### Status

Thông tin của mô hình đang hiển thị: số morph, bone đầu và mắt tìm được, số blendshape ARKit được điều khiển (coverage, ví dụ 51/52). Dòng cuối là trạng thái kết nối với server và số frame nhận được mỗi giây khi capture đang chạy.

### Sliders

- **Head / gaze**: góc đầu (pitch, yaw, roll) và hướng nhìn, tính bằng độ.
- **Canonical inputs (ARKit 52)**: 52 blendshape, mỗi cái từ 0 đến 1.
- **Zero all sliders**: đặt tất cả về 0.

Giá trị hiện bên phải mỗi slider. Slider chỉ dùng được khi Input là Sliders.

## Landmark picker

Trang này dùng để chấm điểm mốc bằng tay. Điểm mốc cho hệ thống biết mắt, mũi, miệng của mô hình nằm ở đâu, để khớp khuôn mặt mẫu lên trước khi sinh blendshape. Thông thường MediaPipe tự tìm được; trang này chỉ cần khi MediaPipe không nhận ra khuôn mặt (hay gặp với mặt anime) hoặc tìm sai.

Bên trái là ảnh chính diện của mô hình: lăn chuột để zoom, kéo chuột phải để di chuyển, phím **F** để đưa khuôn mặt về giữa. Bên phải là tên điểm cần chấm, hình mặt mẫu có chấm đỏ ở đúng vị trí đó, các nút, dòng **Next** cho biết bước tiếp theo, và danh sách 19 điểm.

### Các bước

1. **Click** lên ảnh ở đúng chỗ chấm đỏ trên mặt mẫu. Trang tự chuyển sang điểm tiếp theo.
2. Điểm nào không nhìn thấy (lông mày bị tóc mái che, mép môi khi miệng chỉ là một nét vẽ) thì bấm **Skip** (phím **S**).
3. Chấm sai thì **kéo** điểm đến chỗ đúng, hoặc **Undo** (phím **Z**). Bấm vào một mục trong danh sách để chọn lại điểm đó.
4. **Save landmarks**. Nút chỉ bật khi có thay đổi chưa lưu và đã chấm đủ các điểm chính: bốn khoé mắt, chóp mũi, hai khoé miệng, cằm.
5. **Generate blendshapes**. Nút chỉ bật sau khi đã lưu. Mất khoảng 1 phút; trong lúc chạy, các điểm bị khoá.
6. **Open in viewer →** để xem kết quả.

Màu của điểm:

- Xanh lá: điểm đã chấm.
- Cam: điểm do MediaPipe tìm. Kiểm tra lại và kéo nếu lệch.
- Đỏ: sau khi lưu, tia chiếu từ điểm này không trúng mặt (rơi ra ngoài, hoặc lọt qua hốc mắt). Kéo nhích điểm vào trong một chút rồi lưu lại.

Nếu rời trang khi còn thay đổi chưa lưu, trang sẽ hỏi lại. Link **← Viewer** ở góc trên quay về viewer với đúng mô hình đang làm.

## Các luồng thao tác thường dùng

**Kiểm tra một mô hình mà không cần camera.** Chọn mô hình, Load, chọn Input là Sliders, rồi kéo các slider như jawOpen, eyeBlinkLeft, mouthSmileLeft, head yaw.

**Điều khiển bằng webcam.** Chạy `python -m retarget.capture.mediapipe_capture 0`, để Input là Live capture. Trong cửa sổ webcam: `c` lấy tư thế đầu hiện tại làm tư thế thẳng, `q` để thoát.

**Mô hình thiếu blendshape** (ví dụ `kizunaai` sau khi import với `--no-rig`). Chọn mô hình, bấm Generate blendshapes trong mục Face rig, đợi khoảng 1 phút, viewer tự tải bản `.rigged`.

**Mặt anime MediaPipe không nhận ra** (ví dụ `model4`). Sau khi import, log báo không tìm thấy khuôn mặt; trong viewer, mục Face rig chỉ hiện nút Pick landmarks. Bấm nút đó, chấm điểm, Save, Generate blendshapes, rồi Open in viewer.

**Blendshape sinh ra trông sai.** Trong Face rig bấm Edit landmarks…, kéo các điểm lệch về đúng chỗ, Save, rồi Generate blendshapes.

## Khi gặp lỗi

| Hiện tượng | Cách xử lý |
| --- | --- |
| Face rig báo *Busy: generating …* | Server đang sinh cho mô hình khác. Đợi xong rồi bấm lại. |
| Landmark picker báo chưa có ảnh render | Bấm **Render front view** (khoảng 10 giây). |
| Mô hình mới import không có trong danh sách | Tải lại trang (Ctrl+R). |
| Mô hình nằm ngang hoặc to nhỏ bất thường | Import lại với `--up`, `--front`, `--scale`. Dòng `auto-orient:` trong log cho biết hệ thống đã đoán hướng thế nào. |
| Avatar giật và chậm khi dùng webcam | Xem log capture: nếu chỉ khoảng 7 fps thì phòng quá tối và camera tự hạ tốc độ khung hình. Thêm ánh sáng. |
| Chuyển động rung khi ngồi yên | Trong cửa sổ webcam bấm `f` để so sánh các bộ lọc. Có thể ghi lại bằng `--dump frames.jsonl` để chỉnh tham số trong `retarget/capture/filters.py`. |
