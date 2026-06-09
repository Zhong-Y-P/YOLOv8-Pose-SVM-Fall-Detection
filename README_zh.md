# YOLOv8-Pose-SVM-Fall-Detection
基于 YOLOv8n-Pose + SVM 的人体跌倒检测系统（支持 ROS2 部署）

## 项目简介
本项目结合**人体姿态估计**与**机器学习分类算法**，实现端到端的实时人体跌倒检测。
使用 YOLOv8n-Pose 提取人体检测框与17个骨骼关键点，融合**形态特征（宽高比）、运动特征（质心速度）、骨骼特征（关键点坐标）** 构建特征集，通过 SVM 训练二分类模型，完成跌倒/正常行为判别。
项目不仅实现本地摄像头/视频实时检测，还完成了 **ROS2 节点化部署**，支持话题通信、状态发布与告警监听，适用于智能养老、室内人体监测等场景。

## 技术栈
- 深度学习框架：Ultralytics YOLOv8-Pose
- 机器学习：SVM、StandardScaler (Scikit-learn)
- 视觉处理：OpenCV
- 数据处理：Pandas、NumPy
- 部署框架：ROS2
- 开发环境：Windows / Linux (Ubuntu)

## 核心功能
1. 离线数据集特征提取，自动生成训练用 CSV 特征文件
2. SVM 模型训练、数据均衡、特征标准化与模型保存
3. 本地摄像头/视频实时跌倒检测 + 可视化结果展示
4. ROS2 双节点部署：检测发布节点 + 状态监听告警节点
5. 模型迭代优化：扩充数据集降低「缓慢躺卧」等场景误检率

## 文件说明
- `extract_features.py`：视频特征提取脚本，生成训练数据集
- `t_svm.py`：SVM 模型训练、评估与保存
- `detect.py`：本地实时跌倒检测主程序
- 模型文件：`yolov8n-pose.pt`、`fall_svm_model.pkl`、`scaler.pkl`
