"""YOLO-based plate detector"""

from ultralytics import YOLO
import numpy as np
from typing import Tuple, List

from config import YOLOConfig


class PlateDetector:
    """번호판 탐지 클래스"""
    
    def __init__(self, config: YOLOConfig):
        self.model = YOLO(config.path)
        self.conf_threshold = config.confidence
        # print("✓ YOLO 모델 로드 완료")
        
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """프레임에서 번호판 탐지"""
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold, 
            verbose=False
        )
        annotated_frame = results[0].plot()
        boxes = results[0].boxes if results else []
        
        return annotated_frame, boxes