import tkinter as tk
from tkinter import ttk, messagebox
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import datetime


class PowerPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HỆ THỐNG DỰ BÁO MỨC ĐIỆN NĂNG TIÊU THỤ ")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f2f5")

        self.setup_styles()

        self.entries = {}
        self.model = None
        self.spark = None

        self.create_header()
        self.create_main_content()
        self.create_history_view()
        self.create_status_bar()

        self.root.after(100, self.init_spark_system)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        primary_color = "#2c3e50"
        accent_color = "#27ae60"
        danger_color = "#c0392b"
        text_color = "#34495e"

        # Style cho Label
        style.configure("TLabel", background="#f0f2f5", foreground=text_color, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=primary_color, foreground="white", font=("Segoe UI", 18, "bold"))
        style.configure("Result.TLabel", background="#f0f2f5", foreground="#d35400", font=("Segoe UI", 16, "bold"))

        # Style cho Button
        style.configure("TButton", font=("Segoe UI", 10, "bold"), borderwidth=0, focuscolor="none")
        style.map("TButton", background=[("active", "#bdc3c7")])  # Hiệu ứng hover

        # Custom Button Styles
        style.configure("Predict.TButton", background=accent_color, foreground="white")
        style.map("Predict.TButton", background=[("active", "#2ecc71")])

        style.configure("Clear.TButton", background=danger_color, foreground="white")
        style.map("Clear.TButton", background=[("active", "#e74c3c")])

        # Style cho TreeView
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=25)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#bdc3c7", foreground="black")

    def create_header(self):
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        header_frame.pack(fill="x", side="top")

        lbl_title = tk.Label(header_frame, text=" ỨNG DỤNG DỰ BÁO ĐIỆN NĂNG TIÊU THỤ",
                             bg="#2c3e50", fg="white", font=("Segoe UI", 20, "bold"), pady=20)
        lbl_title.pack()

    def create_main_content(self):
        main_frame = tk.Frame(self.root, bg="#f0f2f5")
        main_frame.pack(fill="x", padx=20, pady=20)

        input_frame = ttk.LabelFrame(main_frame, text="Thông số kỹ thuật đầu vào", padding=(20, 10))
        input_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        fields = [
            ("Global Reactive Power - Công suất phản kháng (kW)", "Global_reactive_power"),
            ("Voltage - Hiệu điện thế (V)", "Voltage"),
            ("Global Intensity - Cường độ dòng điện (A)", "Global_intensity"),
            ("Sub Metering 1 - Công tơ phụ số 1 (Wh)", "Sub_metering_1"),
            ("Sub Metering 2 - Công tơ phụ số 2(Wh)", "Sub_metering_2"),
            ("Sub Metering 3- Công tơ phụ số 3 (Wh)", "Sub_metering_3")
        ]

        for idx, (label_text, col_name) in enumerate(fields):
            lbl = ttk.Label(input_frame, text=label_text)
            lbl.grid(row=idx, column=0, sticky="w", pady=8, padx=5)

            ent = ttk.Entry(input_frame, width=25)
            ent.grid(row=idx, column=1, sticky="e", pady=8, padx=5)
            self.entries[col_name] = ent

        control_frame = tk.Frame(main_frame, bg="#f0f2f5")
        control_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        btn_frame = tk.Frame(control_frame, bg="#f0f2f5")
        btn_frame.pack(pady=20)

        self.btn_predict = ttk.Button(btn_frame, text="CHẠY DỰ BÁO", style="Predict.TButton", command=self.predict,
                                      state="disabled")
        self.btn_predict.pack(side="left", padx=5, ipadx=10, ipady=5)

        btn_clear = ttk.Button(btn_frame, text="XÓA DỮ LIỆU", style="Clear.TButton", command=self.clear_form)
        btn_clear.pack(side="left", padx=5, ipadx=10, ipady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=20)

        ttk.Label(control_frame, text="Kết quả dự báo (Global Active Power):", font=("Segoe UI", 12)).pack()
        self.result_var = tk.StringVar(value="---")
        lbl_result = ttk.Label(control_frame, textvariable=self.result_var, style="Result.TLabel")
        lbl_result.pack(pady=10)

    def create_history_view(self):
        history_frame = ttk.LabelFrame(self.root, text="Lịch sử Dự báo", padding=(10, 5))
        history_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        columns = ("Time", "Reactive", "Voltage", "Intensity", "Sub1", "Sub2", "Sub3", "PREDICTION")
        self.tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=8)

        headers = ["Thời gian", "Reactive (kW)", "Voltage (V)", "Intensity (A)", "Sub1", "Sub2", "Sub3", "KẾT QUẢ (kW)"]
        widths = [120, 100, 80, 80, 60, 60, 60, 120]

        for col, header, w in zip(columns, headers, widths):
            self.tree.heading(col, text=header)
            self.tree.column(col, width=w, anchor="center")

        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_status_bar(self):
        self.status_var = tk.StringVar(value="Đang khởi tạo hệ thống Spark...")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w", bg="#dfe6e9",
                              font=("Arial", 9))
        status_bar.pack(side="bottom", fill="x")

    def init_spark_system(self):
        try:
            self.spark = SparkSession.builder \
                .appName("PowerPredictionPro") \
                .master("local[*]") \
                .config("spark.ui.showConsoleProgress", "false") \
                .getOrCreate()

            model_path = "electric_power_lr_model"
            self.model = PipelineModel.load(model_path)

            self.status_var.set("Hệ thống sẵn sàng! Vui lòng nhập thông số.")
            self.btn_predict.config(state="normal")

        except Exception as e:
            messagebox.showerror("Lỗi Khởi Động", f"Không thể load Model:\n{str(e)}")
            self.status_var.set("Lỗi hệ thống!")

    def clear_form(self):
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.result_var.set("---")
        self.status_var.set("Đã xóa dữ liệu nhập.")

    def predict(self):
        if not self.model:
            return

        try:
            data_values = []
            col_names = [
                "Global_reactive_power", "Voltage", "Global_intensity",
                "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
            ]

            row_data = []

            for col in col_names:
                val_str = self.entries[col].get()
                if not val_str:
                    messagebox.showwarning("Thiếu dữ liệu", f"Vui lòng nhập: {col}")
                    return
                val_float = float(val_str)
                data_values.append(val_float)
                row_data.append(val_float)

            input_df = self.spark.createDataFrame([tuple(data_values)], col_names)

            prediction_df = self.model.transform(input_df)
            result = prediction_df.select("prediction").collect()[0][0]

            result_text = f"{result:.4f} kW"
            self.result_var.set(result_text)
            self.status_var.set("Dự báo thành công!")

            current_time = datetime.datetime.now().strftime("%H:%M:%S %d/%m")
            # Tạo một tuple chứa: Time, Inputs..., Result
            tree_values = (current_time, *row_data, result_text)

            self.tree.insert("", 0, values=tree_values)

        except ValueError:
            messagebox.showerror("Lỗi Nhập Liệu", "Vui lòng chỉ nhập số thực (dùng dấu chấm . cho số thập phân)")
        except Exception as e:
            messagebox.showerror("Lỗi Dự Báo", str(e))

    def on_closing(self):
        if self.spark:
            self.spark.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PowerPredictionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()