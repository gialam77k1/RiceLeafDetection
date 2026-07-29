# RiceLeafDetection

Dự án `RiceLeafDetection` tập trung vào bài toán phát hiện vùng lá bệnh và phân loại bệnh lá lúa bằng deep learning.

## Tổng quan

Repository hiện tại được tổ chức theo hướng nghiên cứu và thử nghiệm mô hình bằng Jupyter Notebook. Từ mã nguồn hiện có, project gồm 2 nhánh chính:

- Tiền xử lý và cắt vùng lá bằng YOLO.
- Phân loại ảnh lá lúa thành 4 lớp bệnh/không bệnh bằng nhiều mô hình CNN và transfer learning.

Ngoài ra, project còn có notebook tổng hợp để đo tốc độ suy luận và so sánh nhiều mô hình trên các tập dữ liệu khác nhau.

## Bài toán và nhãn lớp

Qua các notebook hiện có, bộ dữ liệu phân loại đang được gán 4 lớp:

- `01`: Đốm nâu (`Brown Spot`)
- `02`: Đạo ôn (`Leaf Blast`)
- `03`: Cháy lá (`Leaf Blight`)
- `04`: Bình thường (`Normal`)

## Cấu trúc hiện tại

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

## Nội dung các notebook

### 1. Phân loại ảnh lá lúa

Project đang thử nghiệm nhiều mô hình phân loại:

- `cnn_custom/CNN_Custom.ipynb`
  - Mô hình CNN tự xây dựng.
  - Dữ liệu được tiền xử lý bằng làm sáng, tăng tương phản, làm sắc nét và resize về `224 x 224`.
  - Notebook cho thấy bộ dữ liệu tổng hợp có `4` lớp và số mẫu test trong lần chạy được lưu là `2366`.

- `fine_tuning_Resnet18/fine-tuning-Resnet18.ipynb`
  - Fine-tuning `ResNet18` pretrained.
  - Cấu hình trong notebook: `IMAGE_SIZE = 224`, `BATCH_SIZE = 64`, `EPOCHS = 15`, `PATIENCE = 3`.

- `fine_tuning_Desnet121/densenet121-pytorch-full.ipynb`
  - Fine-tuning `DenseNet121`.
  - Có sử dụng chia tập train/validation/test theo stratified split.
  - Lần chạy được lưu trong notebook cho thấy:
    - `Train samples: 9462`
    - `Validation samples: 1183`
    - `Test samples: 1183`

- `fine_tuning_Mobilenetv2/mobilenetv2-pytorch-full.ipynb`
  - Fine-tuning `MobileNetV2` bằng PyTorch.

- `fine_tuning_mobilenetv3+CBAM attention/Mobilenetv3_cbam_att.ipynb`
  - Thử nghiệm `MobileNetV3` kết hợp cơ chế chú ý `CBAM`.

### 2. Tiền xử lý bằng YOLO

- `yolo_cat_anh/yolo_cat_anh.ipynb`
  - Dùng `YOLO` để phát hiện vùng lá cần quan tâm.
  - Nếu thư mục là ảnh bệnh (`01`, `02`, `03`) thì notebook cắt và lưu từng bounding box.
  - Nếu là thư mục `04` (khỏe mạnh) thì ảnh được sao chép nguyên bản.
  - Có cơ chế "giải cứu" trong trường hợp YOLO không phát hiện được trên ảnh gốc: cắt ảnh thành nửa trên và nửa dưới rồi dự đoán lại.

### 3. Đánh giá tốc độ và pipeline tổng hợp

- `yolo_detection_fps_pipeline/yolo_detection_fps_pipeline.ipynb`
  - Tải mô hình `YOLO` và nhiều mô hình phân loại đã train sẵn.
  - Benchmark trên nhiều tập dữ liệu:
    - `data_01`
    - `data_02`
    - `data_03`
    - `data_test_vn`
  - Các mô hình được load trong notebook:
    - `ResNet18`
    - `Custom_CNN`
    - `DenseNet121`
    - `MobileNetV2`
    - `YOLO` cho bước phát hiện/cắt vùng lá

## Tiền xử lý dữ liệu

Nhiều notebook phân loại đang dùng pipeline tiền xử lý ảnh tương đối giống nhau:

- Đọc ảnh bằng OpenCV
- Làm sáng ảnh
- Tăng tương phản bằng CLAHE
- Làm sắc nét
- Resize về `224 x 224`
- Chia bộ dữ liệu theo `train/validation/test` với `stratified split`

## Môi trường chạy

Từ metadata và đường dẫn trong notebook, project này được phát triển chủ yếu trên:

- Kaggle Notebook
- Google Colab
- Python
- PyTorch
- OpenCV
- Ultralytics YOLO
- Matplotlib, Seaborn, scikit-learn

Nhiều đường dẫn trong notebook đang trỏ tới:

- Google Drive, ví dụ: `/content/drive/MyDrive/...`
- Kaggle Input, ví dụ: `/kaggle/input/...`
- Model YOLO `.pt`
- Checkpoint PyTorch `.pth`

Vì vậy, để chạy lại notebook, cần cập nhật đường dẫn dữ liệu và model cho phù hợp với máy của bạn.

## Quy trình xử lý của project

Luồng làm việc hiện tại của repo có thể tóm tắt như sau:

1. Chuẩn bị dữ liệu ảnh theo thư mục lớp `01`, `02`, `03`, `04`.
2. Dùng `YOLO` để cắt vùng lá bệnh hoặc đối tượng cần phân tích.
3. Tiền xử lý ảnh đã cắt.
4. Huấn luyện và fine-tuning nhiều mô hình phân loại.
5. So sánh kết quả và tốc độ suy luận bằng notebook pipeline tổng hợp.

## Ghi chú về dữ liệu

Thư mục `dataset/` hiện đang chứa:

- `dataset/dataset_train/dataset.docx`
- `dataset/dataset_test/dataset_test.docx`

Đây nhiều khả năng là tài liệu mô tả bộ dữ liệu train/test. Bạn nên mở các file này để bổ sung thêm thông tin về:

- Nguồn thu thập ảnh
- Số lượng ảnh mỗi lớp
- Cách gán nhãn
- Quy ước train/test

## Hướng phát triển tiếp

README có thể được nâng cấp thêm khi project ổn định hơn:

- Thêm bảng kết quả độ chính xác, precision, recall, F1-score
- Thêm ảnh minh họa pipeline YOLO cắt vùng lá
- Thêm hướng dẫn chạy từng notebook
- Tách code dùng chung ra thành `src/`
- Bổ sung file `requirements.txt` hoặc `environment.yml`

## Tác giả

Cập nhật thông tin nhóm thực hiện, giảng viên hướng dẫn, hoặc môn học tại đây nếu cần.
