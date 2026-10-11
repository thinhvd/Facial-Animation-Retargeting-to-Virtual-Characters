# Facial Animation Retargeting to Virtual Characters

Người dùng nhìn vào webcam, một nhân vật 3D trên trình duyệt lặp lại biểu cảm khuôn mặt của họ theo real-time: mở miệng, nháy mắt, nhướn mày, quay đầu, liếc mắt.

Branch này là **phiên bản mới**. Mục tiêu là nhận một mô hình 3D bất kỳ (PMX, glTF, VRM, FBX, OBJ...) và điều khiển được nó mà không cần chỉnh tay cho từng mô hình. Nếu mô hình thiếu blendshape, hệ thống tự sinh phần còn thiếu.

Ý tưởng và cách triển khai được giải thích trong [docs/approach.md](docs/approach.md).

## Hai phiên bản trong repo

| Branch | Nội dung |
| --- | --- |
| `master` | Project cũ (hệ thống của paper CITA 2025): TFLite face tracking → Node.js relay → three.js MMDLoader. Chỉ chạy với mô hình MMD. Hướng dẫn chạy nằm trong README của branch `master`. |
| `feature/p0-canonical-pipeline` | Phiên bản mới (branch này). Code mới nằm trong `retarget/` và `web/`. Code cũ vẫn được giữ nguyên để làm mốc so sánh. |

Khác nhau chính:

| | Project cũ | Phiên bản mới |
| --- | --- | --- |
| Mô hình | MMD (PMX) có morph đặt đúng tên tiếng Nhật | PMX, glTF/GLB, VRM, OBJ; FBX và .blend qua Blender |
| Biểu cảm | Vài biểu cảm, mỗi cái chỉ 0 hoặc 1 | 52 blendshape ARKit, giá trị liên tục từ 0 đến 1, cộng head pose và gaze |
| Face tracking | RFB-320 + 106 landmark, tính EAR/MAR thủ công | MediaPipe Face Landmarker, có One Euro filter |
| Mô hình thiếu morph | Đứng yên | Tự sinh blendshape từ template mặt người (ICT-FaceKit) |
| Mặt anime MediaPipe không nhận ra | Không áp dụng | Chấm khoảng 20 điểm mốc bằng tay trên Web UI |
| Server | Node.js | Python (aiohttp + Socket.IO) |

## Yêu cầu

- Linux (đã chạy trên Ubuntu). Windows và macOS chưa được thử.
- [Miniforge](https://github.com/conda-forge/miniforge) (hoặc Miniconda), Git, một webcam, khoảng 4 GB dung lượng trống.
  Nếu dùng Miniconda và conda báo chưa chấp nhận Terms of Service của kênh `defaults`, chạy hai lệnh `conda tos accept ...` mà thông báo gợi ý, hoặc cài Miniforge.
- Không cần GPU: MediaPipe chạy trên CPU khoảng 9 ms mỗi frame.

## Cài đặt

```bash
# 1. Clone đúng branch
git clone -b feature/p0-canonical-pipeline https://github.com/thinhvd/Facial-Animation-Retargeting-to-Virtual-Characters.git
cd Facial-Animation-Retargeting-to-Virtual-Characters

# 2. Môi trường Python (tên môi trường: fr)
conda env create -f environment.yml
conda activate fr
playwright install chromium

# 3. Model MediaPipe Face Landmarker
curl -L -o weights/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# 4. Template mặt ICT-FaceKit (khoảng 450 MB), dùng khi sinh blendshape
git clone --depth 1 https://github.com/USC-ICT/ICT-FaceKit data/ict-facekit
python -m retarget.convert data/ict-facekit/FaceXModel/generic_neutral_mesh.obj -o web/models/ict_neutral.glb --scale 0.01
```

Chromium ở bước 2 được dùng để render ảnh chính diện của mô hình khi tìm điểm mốc, và để chụp ảnh kiểm tra.

**Tuỳ chọn: Blender** (khoảng 1 GB, không cần quyền root), chỉ cần khi import FBX, .blend hoặc .dae:

```bash
mkdir -p data/blender && cd data/blender
curl -LO https://mirror.freedif.org/blender/release/Blender4.5/blender-4.5.14-linux-x64.tar.xz
tar -xJf blender-4.5.14-linux-x64.tar.xz && rm blender-4.5.14-linux-x64.tar.xz
cd ../..
```

Có thể tải bản Linux từ [blender.org](https://www.blender.org/download/) thay cho mirror trên. Code tự tìm Blender trong `data/blender/`; nếu máy đã cài Blender sẵn thì không cần bước này.

## Chạy thử

Mở hai terminal, ở cả hai đều chạy `conda activate fr` và `cd` vào thư mục repo.

**Terminal 1: server** (để chạy suốt)

```bash
python -m retarget.server            # http://127.0.0.1:6790/
```

**Terminal 2: các bước còn lại**

```bash
# Lần đầu tiên: tìm điểm mốc trên template ICT (khoảng 10 giây)
python -m retarget.rig.landmarks ict_neutral

# Import mô hình mẫu có sẵn trong repo
python -m retarget.importer NodeServer/models/headmodel/HeadA.pmx                       # đầu anime, khoảng 1 phút
python -m retarget.importer NodeServer/models/HumanModel/FemaleModel/testFemale1.pmx    # người, chưa có morph nào

# Điều khiển bằng webcam
python -m retarget.capture.mediapipe_capture 0
```

Mở http://127.0.0.1:6790/, chọn `HeadA.rigged` rồi bấm **Load**. Nhìn vào webcam, nhân vật sẽ cử động theo.

Một số điều nên biết:

- `importer` chuyển mô hình sang glTF, tạo adapter, và tự sinh blendshape nếu mô hình thiếu. Kết quả nằm trong `web/models/`. Tên có đuôi `.rigged` là bản đã được bổ sung blendshape.
- Trong cửa sổ webcam: `q` để thoát, `c` để đặt tư thế đầu hiện tại làm tư thế thẳng, `f` để đổi bộ lọc (euro, none, ema) và so sánh.
- Muốn dùng video thay cho webcam: `python -m retarget.capture.mediapipe_capture duong/dan/video.mp4 --realtime`.
- Mô hình bị nằm ngang hoặc to nhỏ bất thường sau khi import: xem dòng `auto-orient:` trong log, rồi chỉ định lại bằng `--up`, `--front`, `--scale`.

## Web UI

Server phục vụ hai trang:

- **Viewer** (`/viewer.html`): xem mô hình và cho nó cử động. Mục **Input** chọn nguồn điều khiển: **Live capture** (webcam) hoặc **Sliders** (kéo tay từng blendshape, không cần camera). Mục **Face rig** cho biết mô hình đang ở bước nào và chỉ hiện đúng thao tác tiếp theo: Generate blendshapes, Pick landmarks, hoặc Load bản đã sinh.
- **Landmark picker** (`/landmarks.html`): dùng khi MediaPipe không tìm được khuôn mặt (thường gặp với mặt anime). Chấm khoảng 20 điểm mốc theo hình mẫu, **Save**, **Generate blendshapes**, rồi **Open in viewer**.

Hướng dẫn chi tiết từng phần, kèm các luồng thao tác thường dùng: [docs/web-ui.md](docs/web-ui.md).

## Cấu trúc repo

```text
retarget/                    Phiên bản mới (Python)
  canonical.py               Danh sách 52 blendshape ARKit và định dạng một frame
  naming.py                  Bảng tên morph và bone (MMD, VRM, ARKit, tiếng Anh)
  mesh.py                    Cấu trúc mô hình dùng chung cho mọi định dạng
  convert.py                 Chuyển một file mô hình sang .glb trong web/models/
  adapter.py                 Tạo adapter: blendshape ARKit nào được điều khiển bằng morph hay bone nào
  importer.py                Lệnh gộp: convert → adapter → (nếu thiếu) sinh blendshape
  server.py                  Server web + Socket.IO, API cho landmark và job sinh blendshape
  formats/                   Đọc và ghi các định dạng
    pmx.py, pmx_convert.py   PMX (MMD)
    obj.py, obj_convert.py   OBJ (scan, template)
    gltf_reader.py, gltf_writer.py, gltf_convert.py   glTF/GLB
    blender.py               Gọi Blender để đọc FBX, .blend, .dae
    orient.py                Đoán hướng trục và đơn vị của mô hình từ bộ xương
  capture/
    mediapipe_capture.py     Webcam hoặc video → frame (52 blendshape, head pose, gaze) → server
    filters.py               One Euro filter làm mượt chuyển động
  rig/                       Sinh blendshape cho mô hình thiếu
    landmarks.py             Render chính diện → MediaPipe hoặc chấm tay → điểm mốc 3D trên mesh
    registration.py          Khớp template ICT lên mặt mô hình (similarity + non-rigid ICP)
    transfer.py              Deformation transfer: chép biểu cảm từ template sang mô hình
    generate.py              Ghép các bước trên, xuất <model>.rigged.glb
    ict.py, geometry.py, render.py   Đọc ICT-FaceKit, hàm hình học, render qua trình duyệt
    evaluate.py              Đo sai số trên mặt ICT đã biết đáp án
  README.md                  Ghi chú kỹ thuật chi tiết, kèm số liệu (tiếng Anh)
web/
  viewer.html                Viewer (three.js, three-vrm)
  landmarks.html             Trang chấm điểm mốc
  rigjob.js                  Phần dùng chung của hai trang cho job sinh blendshape
  models/                    Mô hình đã import (không đưa lên git)
tools/
  screenshot_viewer.py       Chụp ảnh viewer không cần mở trình duyệt
  blender_export.py          Script chạy bên trong Blender: file → .glb
  validate_glb.js            Kiểm tra file .glb bằng gltf-validator
docs/                        Tài liệu: ý tưởng (approach.md), hướng dẫn Web UI (web-ui.md)
environment.yml              Môi trường conda cho phiên bản mới
weights/                     Model của MediaPipe (tải ở bước cài đặt) và của project cũ
data/                        ICT-FaceKit, Blender (tải về, không đưa lên git)

NodeServer/                  Project cũ: Node.js relay, trang three.js
  models/                    Mô hình mẫu, dùng chung cho cả hai phiên bản
service/, vtuber_link_start.py, requirements.txt   Project cũ: face tracking bằng TFLite
```

## Tài liệu thêm

- [docs/approach.md](docs/approach.md): bài toán, ý tưởng, cách triển khai, kết quả, phần còn để ngỏ.
- [docs/web-ui.md](docs/web-ui.md): hướng dẫn sử dụng Web UI.
- [retarget/README.md](retarget/README.md): ghi chú kỹ thuật cho người sửa code, gồm định dạng adapter, các lỗi đã gặp và số liệu đánh giá.
