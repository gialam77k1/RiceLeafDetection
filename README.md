# RiceLeafDetection

## Giới thiệu

`RiceLeafDetection` là dự án nghiên cứu ứng dụng thị giác máy tính và học sâu để xử lý ảnh lá lúa, với hai mục tiêu chính:

- Phát hiện và cắt vùng lá hoặc vùng bệnh bằng mô hình YOLO.
- Phân loại ảnh lá lúa thành 4 lớp bệnh/không bệnh bằng các mô hình CNN và transfer learning.

Repository hiện được xây dựng chủ yếu dưới dạng các Jupyter Notebook phục vụ thử nghiệm, huấn luyện, đánh giá và so sánh nhiều hướng tiếp cận khác nhau.

## Mục tiêu của dự án

- Xây dựng pipeline tiền xử lý ảnh lá lúa từ dữ liệu gốc.
- Tự động phát hiện vùng lá hoặc vùng tổn thương bằng YOLO.
- Huấn luyện và fine-tuning nhiều mô hình phân loại ảnh.
- So sánh độ chính xác và tốc độ suy luận của các mô hình trên nhiều tập dữ liệu.
- Tạo nền tảng để mở rộng thành hệ thống hỗ trợ chẩn đoán bệnh lá lúa.

## Bài toán phân loại

Dựa trên các notebook hiện có, bài toán phân loại đang sử dụng 4 lớp:

- `01`: Đốm nâu (`Brown Spot`)
- `02`: Đạo ôn (`Leaf Blast`)
- `03`: Cháy lá (`Leaf Blight`)
- `04`: Bình thường (`Normal`)

Trong nhiều notebook, các lớp được ánh xạ như sau:

```python
class_name_mapping = {
    '01': 'Đốm nâu (Brown Spot)',
    '02': 'Đạo ôn (Leaf Blast)',
    '03': 'Cháy lá (Leaf Blight)',
    '04': 'Bình thường (Normal)'
}
```

## Cấu trúc thư mục hiện tại

```text
RiceLeafDetection/
|-- cnn_custom/
|   |-- CNN_Custom.ipynb
|-- fine_tuning_Desnet121/
|   |-- densenet121-pytorch-full.ipynb
|-- fine_tuning_Mobilenetv2/
|   |-- mobilenetv2-pytorch-full.ipynb
|-- fine_tuning_mobilenetv3+CBAM attention/
|   |-- Mobilenetv3_cbam_att.ipynb
|-- fine_tuning_Resnet18/
|   |-- fine-tuning-Resnet18.ipynb
|-- yolo_cat_anh/
|   |-- yolo_cat_anh.ipynb
|-- yolo_detection_fps_pipeline/
|   |-- yolo_detection_fps_pipeline.ipynb
|-- dataset/
|   |-- dataset_train/
|   |   |-- dataset.docx
|   |-- dataset_test/
|   |   |-- dataset_test.docx
|-- README.md
```

## Mô tả chi tiết các thành phần

### 1. Nhóm notebook phân loại ảnh

#### `cnn_custom/CNN_Custom.ipynb`

Notebook này triển khai một mô hình CNN tự xây dựng để phân loại bệnh lá lúa.

Đặc điểm chính:

- Tải dữ liệu từ file nén Google Drive bằng `gdown`
- Giải nén dữ liệu để huấn luyện trên Kaggle
- Tiền xử lý ảnh bằng:
  - làm sáng ảnh
  - tăng tương phản
  - làm sắc nét
  - resize về `224 x 224`
- Chia dữ liệu theo chiến lược `stratified split`
- Huấn luyện mô hình CNN tùy chỉnh bằng PyTorch

Thông tin cấu hình được thấy trong notebook:

- `IMAGE_SIZE = 224`
- `BATCH_SIZE = 128`
- `EPOCHS = 40`

Một lần chạy được lưu trong notebook cho thấy:

- `Train samples: 8279`
- `Validation samples: 1183`
- `Test samples: 2366`

#### `fine_tuning_Resnet18/fine-tuning-Resnet18.ipynb`

Notebook này fine-tuning mô hình `ResNet18` pretrained cho bài toán phân loại 4 lớp.

Đặc điểm chính:

- Tải dữ liệu từ Google Drive bằng `gdown`
- Tiền xử lý ảnh tương tự notebook CNN custom
- Chia dữ liệu thành train/validation/test
- Dùng ResNet18 làm backbone phân loại

Cấu hình chính:

- `IMAGE_SIZE = 224`
- `BATCH_SIZE = 64`
- `EPOCHS = 15`
- `PATIENCE = 3`

Một lần chạy được lưu trong notebook:

- `Train samples: 8279`
- `Validation samples: 1183`
- `Test samples: 2366`

#### `fine_tuning_Desnet121/densenet121-pytorch-full.ipynb`

Notebook fine-tuning `DenseNet121` bằng PyTorch.

Đặc điểm chính:

- Có sử dụng `WeightedRandomSampler`
- Có bước tiền xử lý ảnh bằng OpenCV
- Chia dữ liệu theo `stratified split`
- Có trực quan hóa phân bố lớp, ảnh mẫu, biểu đồ accuracy/loss và confusion matrix

Cấu hình chính:

- `IMAGE_SIZE = 224`
- `BATCH_SIZE = 64`
- `EPOCHS = 100`
- `PATIENCE = 8`

Một lần chạy được lưu trong notebook:

- `Train samples: 9462`
- `Validation samples: 1183`
- `Test samples: 1183`

#### `fine_tuning_Mobilenetv2/mobilenetv2-pytorch-full.ipynb`

Notebook fine-tuning `MobileNetV2`.

Đặc điểm chính:

- Dùng PyTorch và torchvision
- Có chia dữ liệu train/validation/test
- Có dùng `WeightedRandomSampler`
- Phù hợp cho so sánh giữa mô hình gọn nhẹ và các backbone lớn hơn

#### `fine_tuning_mobilenetv3+CBAM attention/Mobilenetv3_cbam_att.ipynb`

Notebook thử nghiệm `MobileNetV3` kết hợp cơ chế chú ý `CBAM`.

Đặc điểm chính:

- Mục tiêu là cải thiện khả năng tập trung vào đặc trưng vùng bệnh
- Có định nghĩa các module:
  - `ChannelAttention`
  - `SpatialAttention`
  - `CBAM`
  - `AttentionPooling`

Đây là notebook theo hướng nghiên cứu nâng cao hơn so với MobileNetV2/ResNet18 cơ bản.

### 2. Notebook YOLO để cắt vùng ảnh

#### `yolo_cat_anh/yolo_cat_anh.ipynb`

Notebook này dùng `YOLO` để phát hiện và cắt vùng lá cần phân tích từ dữ liệu ảnh đầu vào.

Chức năng chính:

- Tải và dùng mô hình YOLO `.pt`
- Duyệt qua các thư mục lớp `01`, `02`, `03`, `04`
- Với các lớp bệnh:
  - chạy YOLO để phát hiện bounding box
  - cắt từng vùng phát hiện được
  - lưu từng box thành ảnh riêng
- Với lớp `04`:
  - sao chép nguyên ảnh vì đây là lớp khỏe mạnh

Cấu hình quan trọng trong notebook:

- `YOLO_MODEL_PATH = "/content/drive/MyDrive/data/yolo_kaggle/best (1).pt"`
- `INPUT_DIR = "/content/drive/MyDrive/data/test_vn/test_VN"`
- `OUTPUT_PARENT_DIR = "/content/drive/MyDrive/data/data_preprocessing_cat_vung_la_lua/data_test_vn"`
- `HEALTHY_FOLDER_NAMES = ['04']`
- `CONFIDENCE_THRESHOLD = 0.25`

Điểm đáng chú ý:

- Có cơ chế fallback khi YOLO không phát hiện được trên ảnh gốc:
  - cắt ảnh thành nửa trên
  - cắt ảnh thành nửa dưới
  - chạy dự đoán lại trên từng phần

Điều này giúp tăng khả năng phát hiện trong trường hợp vùng bệnh nhỏ hoặc khó thấy trên toàn ảnh.

### 3. Notebook pipeline tổng hợp và benchmark

#### `yolo_detection_fps_pipeline/yolo_detection_fps_pipeline.ipynb`

Đây là notebook tổng hợp quan trọng nhất trong repo, dùng để:

- tải mô hình YOLO
- tải nhiều mô hình phân loại đã huấn luyện sẵn
- đánh giá hiệu năng trên nhiều tập dữ liệu
- đo tốc độ suy luận và so sánh mô hình

Cấu hình quan trọng:

```python
YOLO_MODEL_PATH = '/kaggle/input/model-yolo/best (1) (1).pt'
YOLO_CONF_THRESHOLD = 0.25
```

Các mô hình phân loại được load trong notebook:

- `ResNet18`
- `Custom_CNN`
- `Custom_CNN_2`
- `DenseNet121`
- `MobileNetV2`

Các tập dữ liệu benchmark:

- `data_01`
- `data_02`
- `data_03`
- `data_test_vn`

Tên lớp trong notebook benchmark:

```python
CLASS_NAMES = {
    '01': 'Dom nau (Brown Spot)',
    '02': 'Dao on (Leaf Blast)',
    '03': 'Chay la (Leaf Blight)',
    '04': 'Binh thuong (Normal)'
}
```

Notebook này cho thấy định hướng của dự án không chỉ dừng ở huấn luyện mô hình mà còn hướng tới đánh giá thực tế giữa:

- độ chính xác
- tốc độ suy luận
- hiệu quả của bước phát hiện + phân loại kết hợp

## Pipeline xử lý tổng thể của dự án

Luồng xử lý tổng quát của project có thể mô tả như sau:

1. Thu thập ảnh lá lúa theo từng lớp bệnh.
2. Tổ chức dữ liệu theo thư mục `01`, `02`, `03`, `04`.
3. Dùng YOLO để phát hiện và cắt vùng lá hoặc vùng bệnh.
4. Tiền xử lý ảnh:
   - làm sáng
   - tăng tương phản
   - làm sắc nét
   - resize
5. Chia dữ liệu train/validation/test theo `stratified split`.
6. Huấn luyện hoặc fine-tuning mô hình phân loại.
7. Đánh giá accuracy, loss, confusion matrix và tốc độ suy luận.
8. So sánh các mô hình trên nhiều bộ dữ liệu khác nhau.

## Tiền xử lý dữ liệu

Từ các notebook hiện có, pipeline tiền xử lý ảnh đang được dùng lặp lại khá nhất quán:

- Đọc ảnh bằng `cv2.imread`
- Chuyển đổi hoặc chuẩn hóa màu ảnh nếu cần
- Làm sáng ảnh
- Tăng tương phản bằng `CLAHE`
- Làm sắc nét ảnh
- Resize ảnh về `224 x 224`

Ví dụ, một số notebook có các hàm:

- `brighten_image`
- `enhance_contrast`
- `sharpen_image`
- `process_image`
- `split_dataset_stratified`

## Dữ liệu và tài liệu đi kèm

Thư mục `dataset/` hiện có 2 file tài liệu:

- `dataset/dataset_train/dataset.docx`
- `dataset/dataset_test/dataset_test.docx`

Nội dung trích xuất từ các file này cho thấy dự án đang tham chiếu đến dữ liệu lưu trên Google Drive.

### Link dữ liệu train

- `dataset_train`:  
  `https://drive.google.com/drive/folders/1nedqsg0LmdzjkFDkFgnI4UMfS28fndJK?usp=sharing`

### Link dữ liệu test

- `dataset_1`:  
  `https://drive.google.com/drive/folders/1v5sAyFLFTJkzGCYBnoui66-Iq5eo-BcR?usp=sharing`

- `dataset_2`:  
  `https://drive.google.com/drive/folders/1spJHrfLkBe4C5H3pj0c0HrBZKHL079do?usp=drive_link`

- `dataset_3`:  
  `https://drive.google.com/drive/folders/1xVCYwuJ3Jsfb348Kz1nakCb4KZf94O3l?usp=sharing`

- `dataset_tu_thu_thap`:  
  `https://drive.google.com/drive/folders/1RGnpLqnSAKjtT50lyjABrHQOO07-qdXd?usp=drive_link`

Điều này cho thấy repo hiện lưu notebook và tài liệu tham chiếu, còn dữ liệu lớn nhiều khả năng được quản lý ngoài repo.

## Công nghệ sử dụng

Qua các notebook, dự án đang dùng các công nghệ chính sau:

- Python
- Jupyter Notebook
- PyTorch
- Torchvision
- OpenCV
- Ultralytics YOLO
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- PIL
- `gdown` để tải dữ liệu từ Google Drive

## Môi trường phát triển

Từ metadata của notebook, dự án được phát triển chủ yếu trên:

- Kaggle Notebook
- Google Colab

Dấu hiệu nhận biết:

- đường dẫn `/kaggle/input/...`
- đường dẫn `/kaggle/working/...`
- đường dẫn `/content/drive/MyDrive/...`
- có bật GPU trong metadata của notebook

Vì vậy, nếu chạy lại trên máy cục bộ hoặc môi trường khác, cần cập nhật:

- đường dẫn dữ liệu
- đường dẫn model `.pt`
- đường dẫn checkpoint `.pth`
- cấu hình GPU/CPU

## Cách sử dụng repo

Vì project hiện được tổ chức theo notebook, cách sử dụng phù hợp nhất là:

### Bước 1. Chuẩn bị dữ liệu

- Tải dữ liệu từ các link Google Drive ở trên
- Sắp xếp dữ liệu theo đúng cấu trúc thư mục lớp `01`, `02`, `03`, `04`
- Kiểm tra lại đường dẫn trong notebook

### Bước 2. Chạy tiền xử lý YOLO

Notebook:

- `yolo_cat_anh/yolo_cat_anh.ipynb`

Mục tiêu:

- cắt vùng ảnh chứa lá hoặc vùng bệnh
- tạo bộ dữ liệu đầu vào sạch hơn cho phân loại

### Bước 3. Huấn luyện mô hình phân loại

Chọn một trong các notebook:

- `cnn_custom/CNN_Custom.ipynb`
- `fine_tuning_Resnet18/fine-tuning-Resnet18.ipynb`
- `fine_tuning_Desnet121/densenet121-pytorch-full.ipynb`
- `fine_tuning_Mobilenetv2/mobilenetv2-pytorch-full.ipynb`
- `fine_tuning_mobilenetv3+CBAM attention/Mobilenetv3_cbam_att.ipynb`

### Bước 4. Đánh giá và benchmark

Notebook:

- `yolo_detection_fps_pipeline/yolo_detection_fps_pipeline.ipynb`

Mục tiêu:

- load các model đã huấn luyện
- đánh giá trên nhiều tập test
- so sánh tốc độ suy luận giữa các mô hình

## Điểm mạnh hiện tại của project

- Có nhiều hướng mô hình để so sánh
- Có bước phát hiện/cắt vùng lá trước khi phân loại
- Có benchmark trên nhiều tập dữ liệu
- Có kết hợp mô hình nhẹ và mô hình mạnh:
  - CNN Custom
  - ResNet18
  - DenseNet121
  - MobileNetV2
  - MobileNetV3 + CBAM
- Có notebook tổng hợp đánh giá tốc độ thực tế

## Hạn chế hiện tại

Từ cấu trúc repo hiện có, một số hạn chế dễ thấy là:

- Chưa có mã nguồn dạng module `src/` rõ ràng
- Chưa có `requirements.txt`
- Chưa có file cấu hình môi trường
- Chưa có script CLI để chạy tự động
- Nhiều đường dẫn trong notebook đang hard-code theo Kaggle/Colab/Google Drive
- Kết quả thực nghiệm chưa được tổng hợp thành bảng chính thức trong repo

## Đề xuất cải thiện

Để dự án hoàn chỉnh và chuyên nghiệp hơn, có thể mở rộng thêm:

- Tạo thư mục `src/` để gom các hàm dùng chung
- Tạo `requirements.txt`
- Thêm `environment.yml` hoặc notebook setup
- Thêm bảng kết quả so sánh giữa các mô hình
- Thêm ảnh minh họa pipeline YOLO
- Thêm sơ đồ luồng xử lý dữ liệu
- Tạo script suy luận cho 1 ảnh đơn hoặc 1 thư mục ảnh
- Tạo báo cáo kết quả theo:
  - accuracy
  - precision
  - recall
  - F1-score
  - FPS / inference time

## Gợi ý cấu trúc mở rộng trong tương lai

```text
RiceLeafDetection/
|-- data/
|-- notebooks/
|-- src/
|   |-- preprocessing/
|   |-- detection/
|   |-- classification/
|   |-- evaluation/
|-- models/
|-- results/
|-- requirements.txt
|-- README.md
```

## Kết luận

`RiceLeafDetection` là một repo nghiên cứu khá rõ ràng về hướng xử lý ảnh lá lúa theo pipeline:

- phát hiện vùng lá bằng YOLO
- tiền xử lý ảnh
- phân loại bệnh bằng nhiều kiến trúc deep learning
- đánh giá hiệu năng trên nhiều tập dữ liệu

Repo phù hợp cho:

- đồ án môn học
- nghiên cứu thử nghiệm mô hình
- làm nền cho hệ thống nhận diện bệnh lá lúa thực tế

## Tác giả

Bạn có thể bổ sung tại đây:

- tên sinh viên hoặc nhóm thực hiện
- lớp, môn học
- giảng viên hướng dẫn
- trường hoặc đơn vị thực hiện
