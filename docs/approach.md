# Ý tưởng và cách triển khai

Người dùng nhìn vào webcam, một mô hình nhân vật 3D bất kỳ trên màn hình lặp lại biểu cảm khuôn mặt của họ theo real-time, mà không cần artist chỉnh tay cho từng mô hình.

Tài liệu này giải thích bài toán, ý tưởng, cách phiên bản mới được xây dựng và kết quả đạt được. Cách cài đặt và chạy nằm trong [README](../README.md), cách dùng giao diện nằm trong [web-ui.md](web-ui.md).

## 1. Bài toán

Hệ thống cũ (paper CITA 2025, branch `master`) đã điều khiển được nhân vật MMD từ webcam, nhưng còn ba hạn chế:

- **Chỉ chạy với mô hình MMD có morph đặt đúng tên tiếng Nhật.** Mô hình định dạng khác, hoặc không có morph, sẽ đứng yên.
- **Biểu cảm chỉ có hai trạng thái 0 hoặc 1.** Không có mức trung gian nên chuyển động bị giật.
- **Chưa có đánh giá định lượng.** Chưa đo độ chính xác, FPS hay so sánh với baseline.

## 2. Ý tưởng chính: một không gian biểu cảm chung

Mọi dữ liệu đi qua một **canonical expression space** gồm:

- **52 blendshape theo chuẩn ARKit** của Apple, ví dụ jawOpen (mở miệng), eyeBlinkLeft (nhắm mắt trái), browInnerUp (nhướn mày). Mỗi blendshape có trọng số từ 0 đến 1.
- **Head pose** (góc xoay đầu) và **gaze** (hướng nhìn).

Phía capture chỉ cần xuất ra không gian này; phía mô hình chỉ cần hiểu không gian này. Hai phía không cần biết gì về nhau, nên thêm một loại mô hình mới không phải sửa phần capture và ngược lại.

```text
Import (một lần cho mỗi mô hình):
  file mô hình → glTF → tìm morph và bone có sẵn → sinh blendshape còn thiếu → ADAPTER

Runtime (liên tục, từng frame):
  webcam → MediaPipe → 52 blendshape + head pose + gaze → One Euro filter
         → server → trình duyệt → ADAPTER → mô hình chuyển động
```

**Adapter** là một file JSON nhỏ đi kèm mỗi mô hình, cho biết mỗi blendshape ARKit được điều khiển bằng morph nào hoặc bone nào của mô hình đó.

## 3. Phía mô hình

### 3.1 Import mọi định dạng

Mọi mô hình được chuyển về một dạng chung là glTF (`.glb`), vì viewer, adapter và bước sinh blendshape đều đọc glTF.

| Định dạng | Cách đọc |
| --- | --- |
| PMX (MMD), OBJ | Code Python riêng |
| glTF, GLB | Code Python riêng, gộp mọi mesh thành một |
| FBX, .blend, .dae | Blender chạy ngầm để chuyển sang GLB, rồi đọc như trên |
| VRM | Giữ nguyên file, dùng các expression có sẵn trong VRM |

Nhiều file được xuất với trục sai hoặc sai đơn vị: mô hình nằm ngang, hoặc chỉ cao 17 cm. Hệ thống tự đoán lại từ bộ xương: hướng từ hông lên đầu là trục đứng, các cặp bone trái/phải cho biết phía trái, từ đó suy ra hướng mặt; chiều cao vô lý thì nhân với một luỹ thừa của 10.

### 3.2 Adapter: tận dụng thứ mô hình đã có

Adapter được tạo tự động khi import. Với mỗi blendshape, hệ thống chọn nguồn theo thứ tự ưu tiên:

1. **Expression có sẵn của VRM** (aa, blink, lookUp...).
2. **Morph do artist làm**, nhận ra qua bảng tên: tiếng Nhật của MMD (あ, まばたき...), tên VRoid, tên tiếng Anh, tên ARKit. Đây là nguồn đẹp nhất vì đúng phong cách nhân vật.
3. **Bone** cho các chuyển động cứng: xoay đầu, xoay mắt.
4. **Blendshape tự sinh** cho phần còn thiếu (mục 3.3).
5. Nếu không có bone đầu, cả mô hình xoay theo head pose.

Morph một bên (nháy mắt trái, cười lệch phải) được kiểm tra trái/phải bằng hình học, không chỉ dựa vào tên: có mô hình đặt tên theo góc nhìn của người xem, ngược với các mô hình khác.

Tỉ lệ blendshape ARKit được điều khiển gọi là **coverage**, ví dụ 34/52.

### 3.3 Sinh blendshape còn thiếu

Ý tưởng: lấy một khuôn mặt mẫu đã có đủ biểu cảm, khớp nó lên mặt mô hình, rồi chép từng biểu cảm sang. Khuôn mặt mẫu là **ICT-FaceKit** (giấy phép MIT, 53 biểu cảm theo kiểu ARKit).

**Bước 1: điểm mốc.** Render mặt mô hình chính diện trong trình duyệt, chạy MediaPipe trên ảnh đó để tìm 478 điểm mốc, rồi chiếu từng điểm từ ảnh ngược lên bề mặt 3D. Làm y hệt với khuôn mặt mẫu, ta có từng cặp điểm tương ứng. Tia chiếu bỏ qua các lớp trong suốt và xuyên qua tóc mái để chạm đúng vào da mặt.

Với mặt anime, MediaPipe thường không nhận ra khuôn mặt. Khi đó người dùng chấm 19 điểm bằng tay trên Web UI (khoé mắt, mí mắt, lông mày, mũi, miệng, cằm). Thử nghiệm trên mặt người cho thấy 19 điểm chấm tay, kể cả khi lệch vài pixel, cho kết quả ngang với 478 điểm của MediaPipe.

**Bước 2: template registration.** Xoay, dời và co giãn khuôn mặt mẫu cho khớp các điểm mốc, rồi biến dạng dần nó (non-rigid ICP) cho bám sát bề mặt mặt mô hình.

**Bước 3: deformation transfer.** Với mỗi biểu cảm của khuôn mặt mẫu, tính xem từng tam giác của nó biến dạng thế nào, rồi tìm vị trí mới cho các vertex của mô hình sao cho tam giác tương ứng biến dạng giống nhất. Các phần khác nhau được xử lý riêng:

- Da mặt đi theo khuôn mặt mẫu.
- Mảnh nhỏ nằm trên da (lông mày, lông mi) bám theo vùng da bên dưới.
- Tóc giữ nguyên, chỉ đi theo bone đầu.
- Khoang miệng, lợi, lưỡi chuyển động tự do theo vùng xung quanh.
- Nhãn cầu, răng (khối kín nhỏ) không bị biến dạng.

Toàn bộ mất khoảng một phút cho mỗi mô hình, chạy trên CPU. Kết quả là `<tên>.rigged.glb`, cùng một file adapter mới.

## 4. Phía capture

- **MediaPipe Face Landmarker** cho 52 trọng số blendshape và ma trận xoay đầu mỗi frame, khoảng 9 ms trên CPU. Hướng nhìn được suy ra từ các blendshape eyeLook. Mặc định chạy chế độ gương: người dùng nghiêng trái thì nhân vật nghiêng về cùng phía trên màn hình.
- **One Euro filter** làm mượt từng kênh. Khi đứng yên, bộ lọc lọc mạnh để hết rung. Khi cử động nhanh (nháy mắt, quay đầu), nó tự giảm lọc để không bị trễ.
- **Webcam được giữ ở 30 fps.** Nhiều webcam tự hạ xuống 7,5 fps trong phòng tối để phơi sáng lâu hơn. Ở tốc độ đó chuyển động bị giật, và không bộ lọc nào vừa mượt vừa nhanh được.
- **Server** (Python, Socket.IO) chuyển frame tới trình duyệt. **Viewer** (three.js) đọc adapter và đặt trọng số morph, góc bone cho từng frame.

## 5. Web UI

Viewer cho phép xem mô hình, chọn nguồn điều khiển (webcam hoặc slider), và theo dõi mô hình đang ở bước nào của quá trình sinh blendshape. Trang landmark picker để chấm điểm mốc khi MediaPipe không nhận ra khuôn mặt. Các nút chỉ hiện và chỉ bấm được khi phù hợp với trạng thái của mô hình, và server cũng kiểm tra lại, nên không thể làm sai thứ tự (ví dụ sinh blendshape từ các điểm chưa lưu). Chi tiết trong [web-ui.md](web-ui.md).

## 6. Kết quả hiện tại

**Độ chính xác của blendshape tự sinh.** Đo trên các khuôn mặt ICT nằm ngoài khuôn mặt mẫu, nên đáp án đã biết: giấu blendshape thật đi, cho hệ thống sinh lại, rồi đo khoảng cách trung bình giữa vertex sinh ra và vertex thật.

| Mặt thử | Mesh | Sai số trung bình |
| --- | --- | --- |
| identity 7 | cùng mesh với mẫu | 0,57 mm |
| identity 7 | mesh khác (giảm còn 50 %) | 0,23 mm |
| identity 23 | mesh khác (giảm còn 30 %) | 0,27 mm |
| identity 42, khuôn mặt cường điệu | cùng mesh với mẫu | 0,65 mm |

Để so sánh, khoảng cách giữa hai mắt khoảng 90 mm, nên sai số dưới 1 % kích thước khuôn mặt. Hai trường hợp cùng mesh cao hơn vì có lợi và lưỡi bên trong miệng chuyển động tự do.

**Coverage trên các mô hình trong repo.**

| Mô hình | Loại | Trước | Sau |
| --- | --- | --- | --- |
| HeadA.pmx, HeadB.pmx | đầu anime | 34/52 | 52/52 |
| model2.pmx | anime toàn thân | 24/52 | 51/52 |
| model3.pmx | anime toàn thân | 26/52 | 51/52 |
| model4.pmx | anime, MediaPipe không nhận ra, chấm tay | 22/52 | 51/52 |
| MaleModel.pmx, testFemale1.pmx | người, không có morph | 0/52 | 51/52 |
| FemaleModel.fbx | người, qua Blender | 0/52 | 51/52 |
| Seed-san.vrm | VRM mẫu | 24/52 | 31/52 (không sinh thêm) |

52/52 là tối đa; 51/52 là do khuôn mặt mẫu không có tongueOut.

**Hiệu năng.** MediaPipe khoảng 9 ms mỗi frame, webcam đến viewer đạt 30 fps, trên CPU. Sinh blendshape khoảng 45 đến 80 giây cho mỗi mô hình, dùng dưới 1 GB RAM.

## 7. So với proposal ban đầu

Thay đổi:

- Phần chuyển PMX sang glTF và server được viết bằng Python, không cần Blender hay Node.js. Blender chỉ còn dùng cho FBX, .blend, .dae.
- Kế hoạch chia giai đoạn vẫn giữ (P0 nền tảng, P1 sinh blendshape, P2 adapter, P3 anime và hiệu chỉnh, P4 đánh giá). P0 đến P2 đã xong, trừ việc đo baseline của hệ thống cũ. P3 đã làm phần anime, phần hiệu chỉnh chưa làm. P4 chưa bắt đầu.

Làm thêm ngoài proposal:

- Tự đoán trục và đơn vị của mô hình từ bộ xương.
- Kiểm tra trái/phải của morph bằng hình học.
- Chấm điểm mốc bằng tay trên Web UI, kèm luồng thao tác theo trạng thái (Save, Generate, Open).
- Ba sửa lỗi trong bước khớp khuôn mặt mẫu, làm sai số giảm từ 0,34 đến 0,75 mm xuống còn 0,23 đến 0,65 mm.
- Tóc giữ cứng: trước đó tóc mái bị kéo theo lông mày, nhiều hơn cả chính lông mày.
- One Euro filter và giữ webcam ở 30 fps.

## Thuật ngữ

| Thuật ngữ | Nghĩa |
| --- | --- |
| Mesh, vertex | Mesh là bề mặt mô hình 3D, gồm nhiều tam giác; vertex là các đỉnh của tam giác. |
| Blendshape, morph | Một biểu cảm lưu dưới dạng độ dịch chuyển của vertex, ví dụ jawOpen; trộn nhiều cái theo trọng số ra nét mặt. MMD gọi là morph. |
| Bone, rig | Bone là khung xương, xoay bone thì phần mesh gắn với nó xoay theo; rig là toàn bộ bone và blendshape của mô hình. |
| ARKit 52 | Bộ 52 blendshape khuôn mặt do Apple định nghĩa, được dùng rộng rãi để truyền biểu cảm. |
| Landmark (điểm mốc) | Các điểm đặc trưng trên mặt như khoé mắt, chóp mũi, khoé miệng. |
| Adapter | File JSON của mỗi mô hình, cho biết blendshape ARKit nào điều khiển morph hay bone nào. |
| Coverage | Số blendshape ARKit được adapter điều khiển, ví dụ 51/52. |
| Template registration | Biến dạng một khuôn mặt mẫu cho khớp với bề mặt mặt mô hình. |
| Non-rigid ICP | Thuật toán lặp kéo từng vùng của mesh mẫu về điểm gần nhất trên mesh đích, vẫn giữ mesh trơn. |
| Deformation transfer | Kỹ thuật chép một biến dạng (một biểu cảm) từ mesh này sang mesh khác. |
| One Euro filter | Bộ lọc làm mượt tín hiệu, lọc mạnh khi tín hiệu đứng yên và lọc nhẹ khi tín hiệu thay đổi nhanh. |
| glTF, GLB, PMX, VRM, FBX | Các định dạng mô hình 3D. GLB là glTF đóng gói một file; PMX của MMD; VRM cho avatar anime; FBX của Autodesk. |
| Baseline | Phương pháp đối chứng, dùng làm mốc so sánh. |
