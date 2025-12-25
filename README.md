Dự án này sử dụng bộ dữ liệu tiêu thụ điện năng hộ gia đình (lớn) để phân tích và xây dựng mô hình dự báo. Mục tiêu là dự đoán Global Active Power (Công suất tác dụng toàn phần) dựa trên các thông số điện năng khác như điện áp, cường độ dòng điện và các chỉ số đo đếm phụ (sub-metering).Điểm đặc biệt của dự án là việc sử dụng PySpark (Spark MLlib) để xử lý dữ liệu lớn và huấn luyện mô hình, đảm bảo khả năng mở rộng (scalability).

🚀 Tính năng chính
Xử lý dữ liệu lớn (Big Data Processing): 
  Làm sạch và chuẩn hóa dữ liệu đầu vào sử dụng Spark DataFrame.
  Mô hình hóa (Modeling):Sử dụng thuật toán Linear Regression từ thư viện Spark MLlib.
  Đánh giá mô hình bằng K-Fold Cross Validation.
  Pipeline xử lý đặc trưng (VectorAssembler, StandardScaler).
  Trực quan hóa: Biểu đồ đánh giá hiệu suất (R², RMSE) và so sánh thực tế/dự báo.
  Ứng dụng Desktop (Deployment): Giao diện phần mềm (GUI) xây dựng bằng Tkinter cho phép người dùng nhập thông số và nhận kết quả dự báo tức thì từ mô hình đã lưu.

🛠 Yêu cầu hệ thống
Để chạy dự án này, máy tính của bạn cần cài đặt:
  Python (3.8 trở lên)
  Java (JDK 8 hoặc 11): Bắt buộc để chạy Apache Spark.
  Các thư viện Python: Pip install pyspark pandas numpy matplotlib seaborn findspark
📂 Cấu trúc dự ánPlaintextElectric_Power_Prediction/
│
├── household_power_consumption.csv          # Dữ liệu thô (Raw Data)
├── household_power_consumption_cleaned.csv  # Dữ liệu đã làm sạch
│
├── preprocessing.ipynb       # Notebook tiền xử lý & làm sạch dữ liệu
├── linear_regression.ipynb   # Notebook huấn luyện & đánh giá mô hình Spark
├── descriptive_analysis.py   # Script phân tích mô tả dữ liệu
│
├── electric_power_lr_model/  # Thư mục chứa Model đã huấn luyện (PipelineModel)
│   ├── metadata/
│   └── stages/
│
├── predict_app.py            # Ứng dụng giao diện dự báo (Tkinter App)
└── README.md                 # Tài liệu dự án

📊 Hiệu suất Mô hình
  Mô hình Linear Regression đã đạt được kết quả rất tốt trên tập kiểm tra độc lập (Test set):
  Chỉ số (Metric)  Giá trị  Ý nghĩa
  R² (R-squared)   0.9357   Mô hình giải thích được ~93.6% sự biến thiên của dữ liệu.
  RMSE             0.2930   Sai số căn bậc hai trung bình thấp.
  MAE              0.1116   Sai số tuyệt đối trung bình rất nhỏ.
  Mô hình không có dấu hiệu bị Overfitting (kết quả trên tập Train và Test tương đương nhau).

💻 Hướng dẫn sử dụng
Bước 1: Tiền xử lý dữ liệu
  Chạy file preprocessing.ipynb để đọc file CSV gốc, xử lý dữ liệu thiếu (null), chuyển đổi kiểu dữ liệu và lưu ra file ...cleaned.csv.
Bước 2: Huấn luyện mô hình
  Chạy file linear_regression.ipynb. Notebook này sẽ:Khởi tạo Spark Session.Load dữ liệu đã làm sạch.Tạo Pipeline (VectorAssembler -> StandardScaler -> LinearRegression).Huấn luyện và đánh giá mô hình.Lưu    mô hình vào thư mục electric_power_lr_model.
Bước 3: Chạy ứng dụng Dự báo
  Đảm bảo thư mục electric_power_lr_model nằm cùng cấp với file app. Mở terminal và chạy:python predict_app.py
Giao diện ứng dụng sẽ hiện lên, cho phép bạn nhập các thông số: 
  Global Reactive Power
  Voltage
  Global Intensity
  Sub Metering 1, 2, 3
Nhấn nút "CHẠY DỰ BÁO" để xem kết quả tiêu thụ điện năng dự kiến.
