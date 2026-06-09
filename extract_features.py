#导入头文件
import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO

# 加载官方模型
model = YOLO('yolov8n-pose.pt')

# 定义处理视频的函数
def process_videos(video_dir, label, output_csv):
    all_data = []

    # 获取目录下所有视频
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        prev_centroid = None

        print(f"正在处理: {video_file}")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # 1. YOLO 推理
            results = model(frame, verbose=False, conf=0.5)

            for r in results:
                if r.keypoints is not None and len(r.boxes) > 0:
                    # 获取第一个人的框 (x, y, w, h)
                    box = r.boxes[0].xywh[0].cpu().numpy()
                    w, h = box[2], box[3]

                    # 特征 1: 宽高比 
                    aspect_ratio = w / (h + 1e-6)

                    # 特征 2: 质心速度 
                    centroid = np.array([box[0], box[1]])
                    velocity = 0
                    if prev_centroid is not None:
                        velocity = np.linalg.norm(centroid - prev_centroid)
                    prev_centroid = centroid

                    # 特征 3: 17个关键点坐标 (归一化)
                    # 使用 .xyn 获得 0-1 之间的坐标，不受分辨率影响
                    kpts = r.keypoints.xyn[0].cpu().numpy().flatten().tolist()

                    # 只有检测到完整 17 个点才记录
                    if len(kpts) == 34:
                        all_data.append([aspect_ratio, velocity] + kpts + [label])

        cap.release()

    # 保存结果
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, mode='a', index=False, header=not os.path.exists(output_csv))


# --- 执行步骤 ---
# 请修改为你的数据集实际路径
if __name__ == "__main__":
    if os.path.exists('train_features.csv'): os.remove('train_features.csv')

    print("开始提取跌倒特征...")
    process_videos("D:/file_download/archive/Fall/Raw_Video", 1, 'train_features.csv')

    print("开始提取正常特征...")
    process_videos("D:/file_download/archive/No_Fall/Raw_Video", 0, 'train_features.csv')
    print("特征提取完成！生成的 train_features.csv 可以直接用于 SVM 训练。")