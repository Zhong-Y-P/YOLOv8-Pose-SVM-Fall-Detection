import cv2
import numpy as np
import joblib
from ultralytics import YOLO

# 1. 加载所有模型
print("正在加载模型...")
yolo_model = YOLO('yolov8n-pose.pt')  # YOLO 提取骨架
svm_clf = joblib.load('fall_svm_model.pkl')  # 你训练好的 SVM
scaler = joblib.load('scaler.pkl')  # 训练时的标准化器

# 2. 打开视频源 (0 代表摄像头，也可以换成 'test.mp4' 视频文件)
cap = cv2.VideoCapture(0)
prev_centroid = None  # 用于计算质心速度

print("系统启动成功！按 'q' 键退出。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # YOLO 姿态估计推理
    results = yolo_model(frame, verbose=False, conf=0.5)

    for r in results:
        # 确保检测到了人且有关键点
        if r.keypoints is not None and len(r.boxes) > 0:
            # --- 提取特征 (必须与训练时的顺序完全一致) ---

            # 获取人体框 (x_center, y_center, width, height)
            box = r.boxes[0].xywh[0].cpu().numpy()
            w, h = box[2], box[3]

            # 特征 1: 宽高比 (Aspect Ratio)
            aspect_ratio = w / (h + 1e-6)

            # 特征 2: 质心速度 (Velocity)
            centroid = np.array([box[0], box[1]])
            velocity = 0
            if prev_centroid is not None:
                velocity = np.linalg.norm(centroid - prev_centroid)
            prev_centroid = centroid

            # 特征 3: 17个关键点归一化坐标 (34个值)
            kpts = r.keypoints.xyn[0].cpu().numpy().flatten().tolist()

            # 如果关键点完整，则进行预测
            if len(kpts) == 34:
                # 组合当前帧所有特征 [宽高比, 速度, 17个点的x,y]
                feat_vector = np.array([[aspect_ratio, velocity] + kpts])

                # --- 核心步骤：标准化并预测 ---
                feat_scaled = scaler.transform(feat_vector)
                prediction = svm_clf.predict(feat_scaled)[0]
                # 获取概率 (可选)
                prob = svm_clf.predict_proba(feat_scaled)[0][1]

                # --- 绘制 UI ---
                # 跌倒为 1 (红色)，正常为 0 (绿色)
                color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                status = "FALL DETECTED!" if prediction == 1 else "Normal"

                # 画人体框
                x1, y1, x2, y2 = r.boxes[0].xyxy[0].cpu().numpy()
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # 写状态文字
                display_text = f"{status} ({prob:.2f})"
                cv2.putText(frame, display_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 显示结果窗口
    cv2.imshow('Real-time Fall Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()