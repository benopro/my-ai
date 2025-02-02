import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Dùng phiên bản nhẹ nhất

def analyze_image(image_path):
    results = model(image_path)
    results.show()  # Hiển thị ảnh có đánh dấu đối tượng

analyze_image("test.jpg")  # Thử phân tích một hình ảnh
