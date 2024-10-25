from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
from google.colab.patches import cv2_imshow
import gradio as gr
!pip install ultralytics opencv-python-headless
! pip install mediapipe

class ObjectAndPoseDetector:
    def __init__(self, yolo_model_path):
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process_image(self, image_path):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Run YOLO detection
        yolo_results = self.yolo_model(image)[0]
        
        # Draw YOLO detections
        annotated_image = image.copy()
        boxes = yolo_results.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get confidence and class
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label
            label = f"{yolo_results.names[class_id]} {confidence:.2f}"
            cv2.putText(annotated_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Run pose detection
        pose_results = self.pose.process(rgb_image)
        
        # Draw pose landmarks if detected
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            
        return annotated_image, yolo_results, pose_results
    
    def process_images(self, image_paths):
        """Process multiple images and yield results"""
        for image_path in image_paths:
            try:
                yield self.process_image(image_path)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
    
    def __del__(self):
        # Clean up MediaPipe resources
        self.pose.close()


def process_image_gradio(image_path):
    detector = ObjectAndPoseDetector("/content/best.pt")
    annotated_image, _, _ = detector.process_image(image_path)
    return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

iface = gr.Interface(
    fn=process_image_gradio,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Image(type="numpy"),
    title="Object and Pose Detection",
    description="Upload an image to detect objects and human pose."
)

iface.launch(share=True)