# 训练 SVM 模型
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# --- 1. 加载数据 ---
print("正在读取 CSV 文件 (374MB)...")
start_time = time.time()
df = pd.read_csv('train_features.csv')
print(f"加载完成，原始规模: {len(df)} 帧，耗时: {time.time() - start_time:.2f}s")

# --- 2.  (针对 57 万行数据的优化) ---
print("\n正在按比例筛选精华数据...")
df_fall = df[df.iloc[:, -1] == 1]      # 跌倒
df_normal = df[df.iloc[:, -1] == 0]    # 正常

# 设置上限：2万跌倒，4万正常 (总计6万)
max_fall_samples = 20000
max_normal_samples = 40000

if len(df_fall) > max_fall_samples:
    df_fall = resample(df_fall, replace=False, n_samples=max_fall_samples, random_state=42)
if len(df_normal) > max_normal_samples:
    df_normal = resample(df_normal, replace=False, n_samples=max_normal_samples, random_state=42)

df_balanced = pd.concat([df_fall, df_normal])
print(f"筛选后规模: 跌倒({len(df_fall)}帧), 正常({len(df_normal)}帧), 总计: {len(df_balanced)}帧")

# --- 3. 预处理 ---
X = df_balanced.iloc[:, :-1].values
y = df_balanced.iloc[:, -1].values

# 标准化特征 (非常重要！) - 训练时 fit，推理时 transform
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 4. 开始训练 (带进度查看) ---
print("\n开始训练 SVM 模型 (开启 verbose 模式)...")
print("你可以观察屏幕上的 '.' 和 '*'，它们跳动代表程序正在运行，没有死机。")

train_start = time.time()
# 增加 cache_size 以利用更多内存加速
clf = SVC(kernel='rbf',
          probability=True,
          verbose=True,
          cache_size=1000)

clf.fit(X_train, y_train)
train_duration = time.time() - train_start

# --- 5. 保存 ---
joblib.dump(clf, 'fall_svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n" + "="*40)
print(f"训练成功！总耗时: {train_duration:.2f} 秒")
print(f"最终测试集准确率: {clf.score(X_test, y_test):.4f}")
print("模型文件已生成: fall_svm_model.pkl, scaler.pkl")
print("="*40)