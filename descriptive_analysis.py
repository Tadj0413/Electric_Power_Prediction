import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("household_power_consumption.csv", na_values="?")


def clean_data(df):
    df = df.fillna({'Global_active_power': df['Global_active_power'].median()})
    df = df.fillna({'Global_reactive_power': df['Global_reactive_power'].median()})
    df = df.fillna({'Voltage': df['Voltage'].median()})
    df = df.fillna({'Global_intensity': df['Global_intensity'].median()})
    df = df.fillna({'Sub_metering_1': df['Sub_metering_1'].ffill()})
    df = df.fillna({'Sub_metering_2': df['Sub_metering_2'].ffill()})
    df = df.fillna({'Sub_metering_3': df['Sub_metering_3'].ffill()})
    return df


df_clean = clean_data(df)

duplicates = df_clean.duplicated().sum() # tim so ban ghi trung nhau
duplicates_rows = df_clean[df_clean.duplicated(keep=False)]

df_final = df_clean.drop_duplicates()

df_final["timestamp"] = pd.to_datetime(
    df_final["Date"].astype(str) + " " + df_final["Time"].astype(str),
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)
df_final.drop(columns=["Date", "Time"], inplace=True)

df_final = df_final.sort_values("timestamp")
df_final["label"] = df_final["Global_active_power"].shift(-1)
df_final = df_final.dropna(subset=["label"])


# bieu do boxplot
df_final.drop(columns=["label", "timestamp"], inplace=True)
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_final, orient='v')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f"Biểu đồ Boxplot cho các cột dữ liệu số (ĐÃ CHUẨN HOÁ) ", fontsize=16)
plt.xlabel("")
plt.show()

for col in df_final.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_final[col], orient='v')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f"Biểu đồ Boxplot cho cột {col}", fontsize=16)
    plt.ylabel(f"Giá trị của {col}", fontsize=12)
    plt.xlabel("")
    plt.show()

#Jointplot
df_sample = df_final.sample(n=8000, random_state=42)
sns.jointplot(data=df_final, x="Global_intensity", y="Global_active_power", kind="scatter", height=7, alpha=0.4)
plt.suptitle(f"Biểu đồ Jointplot giữa cột 'Global_active_power' và 'Global_intensity' ", fontsize=16)
plt.show()

sns.jointplot(data=df_final, x="Voltage", y="Global_active_power", kind="scatter", height=7, alpha=0.4)
plt.suptitle(f"Biểu đồ Jointplot giữa cột 'Global_active_power' và 'Voltage' ", fontsize=16)
plt.show()

sns.jointplot(data=df_final, x="Sub_metering_3", y="Global_active_power", kind="scatter", height=7, alpha=0.4)
plt.suptitle(f"Biểu đồ Jointplot giữa cột 'Global_active_power' và 'Sub_metering_3' ", fontsize=16)
plt.show()

# Violinplot
df_sample = df.sample(n=10000, random_state=42)
for col in df_final.columns:
    plt.figure(figsize=(6, 5))
    sns.violinplot(y=df_sample[col],inner="quartile",cut=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f"Biểu đồ Jointplot cho cột {col}", fontsize=16)
    plt.ylabel(f"Giá trị của {col}", fontsize=12)
    plt.xlabel("")
    plt.show()

#3 cot sub_metering
plt.figure(figsize=(8,5))
sns.violinplot(data=df_sample[["Sub_metering_1","Sub_metering_2","Sub_metering_3"]],inner="quartile")
plt.title("Biểu đồ Jointplot cho 3 cột Sub_metering")
plt.show()

#bieu do histogram (phan phoi)
for col in df_final.columns:
    df_final[col].hist(figsize=(10, 6), bins=20, edgecolor='black')
    plt.suptitle(f"Biểu đồ histogram cột {col}", fontsize=16)
    plt.xlabel('Giá trị', fontsize=12)
    plt.ylabel("Tần suất", fontsize=12)
    plt.tight_layout()
    plt.show()

#bieu do scatter plot
def drawScatterPlot(x, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(df_final[x], df_final[y], alpha=0.6)
    plt.title(f"Biểu đồ tán xạ giữa {x} và {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()


drawScatterPlot("Global_active_power", "Global_intensity")
drawScatterPlot("Global_active_power", "Global_reactive_power")
drawScatterPlot("Global_active_power", "Voltage")
drawScatterPlot("Global_active_power", "Global_intensity")
drawScatterPlot("Global_active_power", "Sub_metering_1")
drawScatterPlot("Global_active_power", "Sub_metering_2")
drawScatterPlot("Global_active_power", "Sub_metering_3")


#Biểu đồ Residual plot
df_final = df_final.sample(n=20000, random_state=42)
x = df_final.drop("Global_active_power", axis=1)
y = df_final["Global_active_power"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#Residuals
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.4)
plt.axhline(0)
plt.xlabel("Fitted Values (Predicted Global_active_power)")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

plt.figure(figsize=(7,5))
plt.hist(residuals, bins=50)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.4)
plt.xlabel("Fitted Values")
plt.ylabel("√|Residuals|")
plt.title("Scale-Location Plot")
plt.show()

# bieu do heatmap
# Tính ma trận tương quan giữa tất cả các cột
correlation_matrix = df_final.corr()
# (Optional: Vẽ Heatmap tương quan để trực quan hóa)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
plt.title('Ma trận tương quan giữa các yếu tố và global_active_power')
plt.show()





