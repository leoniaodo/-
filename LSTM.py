import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
data = pd.read_csv("C:/Users/Boreas/Desktop/archive/MetroPT3(AirCompressor).csv")

# 转换时间戳并归一化特征参数
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.drop(columns=['Unnamed: 0'])  # 去除不需要的列

# 定义时间步长和特征列
time_steps = 5
scaler = StandardScaler()
sensor_columns = [
    'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 
    'Oil_temperature', 'Motor_current', 'COMP', 
    'DV_eletric', 'Towers', 'MPG', 'LPS', 
    'Pressure_switch', 'Oil_level', 'Caudal_impulses'
]
data[sensor_columns] = scaler.fit_transform(data[sensor_columns])

# 添加故障数据标签
labeled_data = data.copy()
labeled_data['status'] = 0

# 故障时间戳
def to_datetime(xs):
    result = []
    format = "%Y-%m-%d %H:%M:%S"
    for x in xs:
        result.append(pd.to_datetime(x, format=format))
    return result

failure_start_time = to_datetime(["2020-04-18 00:00:00", "2020-05-29 23:30:00", "2020-06-05 10:00:00", "2020-07-15 14:30:00"])
failure_end_time = to_datetime(["2020-04-18 23:59:00", "2020-05-30 06:00:00", "2020-06-07 14:30:00", "2020-07-15 19:00:00"])

# 故障标记
def in_between(x, start, end):
    start_con = x >= start
    end_con = x <= end
    inbetween_con = start_con and end_con
    if inbetween_con:
        return 1
    else:
        return 0

# 遍历寻找故障索引
failure_indx = []
for i, (start_time, end_time) in enumerate(zip(failure_start_time, failure_end_time)):
    mask = labeled_data['timestamp'].apply(in_between, start=start_time, end=end_time)
    indx = labeled_data.index[mask == True].tolist()
    failure_indx += indx

# 更新故障标签样本信息
labeled_data['status'].iloc[failure_indx] = 1

# 划分数据集
X = labeled_data.iloc[:, 1:-1]  # 特征列
y = labeled_data['status']  # 目标变量
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 计算相关性矩阵
correlation_matrix = pd.concat([X_train, y_train], axis=1).corr()

# 可视化相关性矩阵
plt.figure(figsize=(12, 10))  # 设置图表大小
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features with Status')
plt.show()

# 重新划分训练集
X = labeled_data.iloc[:, [1, 4, 6, 7, 9]]  # 特征列
y = labeled_data['status']  # 目标变量
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
adam = Adam(learning_rate=0.001)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# 在验证集上评估模型
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 预测验证集
y_pred = model.predict(X_val)
y_pred_class = (y_pred > 0.5).astype(int)  # 将预测的概率转换为标签（0或1）

# 计算精确率、召回率和F1分数
precision = precision_score(y_val, y_pred_class)
recall = recall_score(y_val, y_pred_class)
f1 = f1_score(y_val, y_pred_class)

# 打印精确率、召回率和F1分数
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

# 绘制训练和验证的准确率
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# 绘制精确率折线图
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Precision')
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend(loc='lower right')

# 绘制召回率折线图
plt.subplot(1, 3, 3)
plt.plot(history.history['val_accuracy'], label='Recall')
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# 绘制F1分数折线图
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], marker='o')
plt.title('F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.ylim
plt.grid(True)
plt.show()

# 计算并打印分类报告
print("Classification Report:")
print(classification_report(y_val, y_pred_class))

# 绘制分类报告的热图
classification_rep = classification_report(y_val, y_pred_class, output_dict=True)
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(classification_rep).transpose(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Classification Report')
plt.show()
