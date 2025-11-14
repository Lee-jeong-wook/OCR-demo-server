"""Plate region extraction utilities"""

import cv2
import numpy as np
from typing import Optional

from config import ProcessingConfig


class PlateExtractor:
    """번호판 영역 추출"""
    
    def __init__(self, config: ProcessingConfig):
        self.padding_ratio = config.plate_padding_ratio
        self.scale_factor = config.plate_scale_factor
    
    def extract(self, frame: np.ndarray, box) -> Optional[np.ndarray]:
        """바운딩 박스로부터 번호판 추출"""
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = frame.shape[:2]
        
        # 경계 체크
        if y1 < 0 or y2 > h or x1 < 0 or x2 > w:
            return None
        
        # 패딩
        y_pad = int((y2 - y1) * self.padding_ratio)
        x_pad = int((x2 - x1) * self.padding_ratio)
        
        y1_pad = max(0, y1 - y_pad)
        y2_pad = min(h, y2 + y_pad)
        x1_pad = max(0, x1 - x_pad)
        x2_pad = min(w, x2 + x_pad)
        
        plate_img = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if plate_img.size == 0:
            return None
        
        # 확대
        plate_img = cv2.resize(
            plate_img, 
            None, 
            fx=self.scale_factor, 
            fy=self.scale_factor, 
            interpolation=cv2.INTER_CUBIC
        )
        
        return plate_img