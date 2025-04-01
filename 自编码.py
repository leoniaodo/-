import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score,precision_score, recall_score, f1_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
data = pd.read_csv("C:/Users/Boreas/Desktop/archive/MetroPT3(AirCompressor).csv")

# 检查缺失值
#print(data.isna().sum())

# 打印数据集信息
#data.info()

# 打印统计信息
#print("\nSummary Statistics:")
#data.describe()

# 转换时间戳并归一化特征参数
data['timestamp'] = pd.to_datetime(data['timestamp'])
data = data.drop(columns=['Unnamed: 0'])
scaler = StandardScaler()
sensor_columns = [
    'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 
    'Oil_temperature', 'Motor_current', 'COMP', 
    'DV_eletric', 'Towers', 'MPG', 'LPS', 
    'Pressure_switch', 'Oil_level', 'Caudal_impulses'
]
data[sensor_columns] = scaler.fit_transform(data[sensor_columns])

# 定义分段参数并重新采样数据
segment_length = 60  # 以秒为单位的段长度
data.set_index('timestamp', inplace=True)
segmented_data = data.resample(f'{segment_length}S').agg({
    'TP2': ['mean', 'std', 'min', 'max'],
    'TP3': ['mean', 'std', 'min', 'max'],
    'H1': ['mean', 'std', 'min', 'max'],
    'DV_pressure': ['mean', 'std', 'min', 'max'],
    'Reservoirs': ['mean', 'std', 'min', 'max'],
    'Oil_temperature': ['mean', 'std', 'min', 'max'],
    'Motor_current': ['mean', 'std', 'min', 'max'],
    'COMP': ['mean', 'std'],
    'DV_eletric': ['mean', 'std'],
    'Towers': ['mean', 'std'],
    'MPG': ['mean', 'std'],
    'LPS': ['mean', 'std'],
    'Pressure_switch': ['mean', 'std'],
    'Oil_level': ['mean', 'std'],
    'Caudal_impulses': ['mean', 'std']
})

# 处理MultiIndex列和缺失值
segmented_data.columns = ['_'.join(col).strip() for col in segmented_data.columns.values]
segmented_data.dropna(inplace=True)
segmented_data.reset_index(inplace=True)

# 准备数据
features = segmented_data.drop(columns=['timestamp']).values

# 定义自编码器模型
input_dim = features.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation='relu')(input_layer)
bottleneck = Dense(32, activation='relu')(encoder)
decoder = Dense(64, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='linear')(decoder)
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# 训练自编码器
autoencoder.fit(features, features, epochs=50, batch_size=256, shuffle=True)

#保存模型
#autoencoder.save('sparse_autoencoder_model.h5')

# 计算重建误差
reconstructed = autoencoder.predict(features)
reconstruction_error = np.mean(np.power(features - reconstructed, 2), axis=1)
print(f"reconstruction_error: {reconstruction_error}")

# 设置异常阈值
threshold = np.percentile(reconstruction_error, 95)
anomalies = reconstruction_error > threshold
y = anomalies.astype(int)
print(f"threshold: {threshold}")

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(features, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 定义分类模型
model = Sequential()
model.add(Dense(128, input_shape=(X_train_scaled.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_val_scaled, y_val))

# 在测试集上评估模型
loss, accuracy = model.evaluate(X_val_scaled, y_val)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 预测
y_pred = model.predict(X_val_scaled)
y_pred_class = (y_pred > 0.5).astype(int)  # 二值化

# 初始化存储评价指标的列表
precisions, recalls, f1_scores = [], [], []

# 在每个epoch结束后计算精确率、召回率和F1分数
for epoch in range(len(history.history['accuracy'])):
    # 预测验证集的结果
    y_pred_val = (model.predict(X_val_scaled) > 0.5).astype(int)
    
    # 计算精确率、召回率和F1分数
    precision = precision_score(y_val, y_pred_val)
    recall = recall_score(y_val, y_pred_val)
    f1 = f1_score(y_val, y_pred_val)
    
    # 将计算结果添加到列表中
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# 计算准确率
accuracy = accuracy_score(y_val, y_pred_class)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 计算MSE
mse = mean_squared_error(y_val, y_pred_class)
print(f"Mean Squared Error: {mse}")

# 计算评价指标
print("Classification Report:")
print(classification_report(y_val, y_pred_class))

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
plt.plot(precisions, label='Precision')
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend(loc='lower right')

# 绘制召回率折线图
plt.subplot(1, 3, 3)
plt.plot(recalls, label='Recall')
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# 绘制F1分数折线图
plt.figure(figsize=(6, 4))
plt.plot(f1_scores, marker='o')
plt.title('F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.ylim
plt.grid(True)
plt.show()

# 计算评价指标
print("Classification Report:")
print(classification_report(y_val, y_pred_class))

# 绘制评价指标
plt.figure(figsize=(8, 6))
sns.heatmap(pd.DataFrame(classification_report(y_val, y_pred_class, output_dict=True)), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Classification Report')
plt.show()