import cv2
import torch
from ultralytics import YOLO

def main():
    # 加载模型（请确保在本地有 best.pt，该文件已在 .gitignore 中忽略）
    model = YOLO("best.pt")
    
    # 读取测试视频
    video_path = "test.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # 准备输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (w, h))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 推理
        results = model(frame, conf=0.5)
        
        # 绘制检测框与关键点
        for res in results:
            boxes = res.boxes
            keypoints = res.keypoints
            
            # 画框
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            # 画关键点
            if keypoints is not None:
                for kp in keypoints:
                    points = kp.data[0]
                    for p in points:
                        x, y = int(p[0]), int(p[1])
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                        
        out.write(frame)
        cv2.imshow("Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
