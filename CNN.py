import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import  precision_score, recall_score, f1_score,classification_report

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, accuracy_score

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

#添加故障数据标签
labeled_data = data.copy()
labeled_data['status'] = 0
#print(labeled_data.head(5))

#故障时间戳
def to_datetime(xs):
  result = []
  format =  "%Y-%m-%d %H:%M:%S"
  for x in xs:
    result.append(pd.to_datetime(x, format = format))
  return result
failure_start_time = to_datetime(["2020-04-18 00:00:00", "2020-05-29 23:30:00", "2020-06-05 10:00:00", "2020-07-15 14:30:00"] )
failure_end_time   = to_datetime(["2020-04-18 23:59:00", "2020-05-30 06:00:00", "2020-06-07 14:30:00", "2020-07-15 19:00:00"] )
#print(failure_start_time,"\n",failure_end_time,failure_end_time[0].minute)

#故障标记
def in_between(x, start, end):

  start_con = x >= start
  end_con = x<= end

  inbetween_con = start_con and end_con
  if inbetween_con:
    return 1
  else:
    return 0

#遍历寻找故障索引
failure_indx = []
import numpy as np
for i, (start_time, end_time) in enumerate(zip(failure_start_time, failure_end_time)):
  mask = labeled_data['timestamp'].apply(in_between, start = start_time, end = end_time)
  indx = labeled_data.index[mask == True].tolist()
  failure_indx += indx
#print(f" Found {len(failure_indx)} samples representing failure state")

#更新故障标签样本信息
labeled_data['status'].iloc[failure_indx] = 1
#print(f"We have {labeled_data['status'][labeled_data['status']==1].count()} positve samples" )
#print(f"Example of Failure state \n {labeled_data[labeled_data['status']==1].head()}")

#分离正负样本
pos_data = labeled_data[labeled_data['status'] == 1]
neg_data = labeled_data[labeled_data['status'] == 0]
#print(f"Positive dataset\n {pos_data.info()}\n")
#print(f"Negative dataset\n {neg_data.info()}\n")

#负样本抽样，抽取正样本数量相同的负样本
n_positives = int(pos_data['status'].count())
sub_neg_data = neg_data.sample(n_positives, random_state = 42)
#print(f"Negative dataset after subsampling {sub_neg_data.info()}")

#合并正负样本
merged_data = pd.concat([pos_data, sub_neg_data], axis = 0)
print(f"Merged dataset\n")
merged_data.info()

#异常值检验
def investigate_outliers(data, c):
    q1 = data[c].quantile(0.25)
    q3 = data[c].quantile(0.75)
    iqr = q3 - q1
    ll = q1 - 1.5*iqr
    ul = q3 + 1.5*iqr

    num_outliers = data[data[c] < ll][c].count()  + data[data[c] > ul][c].count()
    if num_outliers>0:
        print(f"Found {num_outliers} oulier(s) for feature {c}")
    return {'col': c, 'n_outliers': num_outliers, 'll': ll, 'ul': ul, 'q1': q1, 'q3':q3}
print("\nDropping outliers ...\n")

#异常值处理
clean_data = merged_data.copy()
for i in range(5):
  for c in clean_data.columns:
      if c not in ["Unnamed: 0","timestamp"]:
          cue = investigate_outliers(clean_data, c)
          if cue["n_outliers"] > 0 and (cue["q1"]!= cue["q3"]):
              print(f"Droping {cue['n_outliers']} from column {c}")
              clean_data = clean_data[clean_data[c]> cue["ll"]]
              clean_data = clean_data[clean_data[c]< cue["ul"]]
              print(f"{clean_data.shape[0]} samples left\n")
          elif (cue["q1"]== cue["q3"]):
              print("Skipping .. data has Q1 equals to Q3")
              print(f"{clean_data.shape[0]} rows left\n")
print("\nDropping Completed ...\n")

#二进制处理
binary_cols = ['LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']
clean_data[binary_cols] = clean_data[binary_cols].apply(np.round)

#绘制相关性矩阵
clean_data.corr().round(2)
sns.heatmap(clean_data.corr().round(2),annot=False )
plt.show()

#清理数据并保存
clean_data.to_csv('Group_14_Clean_Data.csv')
np.savez("Group_14_Clean_Data.npz", clean_data.to_numpy())

#加载数据
data = pd.read_csv('Group_14_Clean_Data.csv')

#划分数据集
X = data.iloc[:, 2:-1]
y = data.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
plt.subplot(1, 4, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# 绘制精确率折线图
plt.subplot(1, 4, 2)
plt.plot(precisions, label='Precision')
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend(loc='lower right')

# 绘制召回率折线图
plt.subplot(1, 4, 3)
plt.plot(recalls, label='Recall')
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()

# 绘制F1分数折线图
plt.subplot(1, 4, 4)
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