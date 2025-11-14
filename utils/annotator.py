"""Frame annotation utilities"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional

from config import FontConfig, ProcessingConfig


class FrameAnnotator:
    """프레임 주석 처리"""
    
    def __init__(self, font_config: FontConfig, processing_config: ProcessingConfig):
        self.font = self._load_font(font_config)
        self.min_confidence = processing_config.min_confidence
    
    @staticmethod
    def _load_font(font_config: FontConfig) -> ImageFont.FreeTypeFont:
        """한글 폰트 로드"""
        for path in font_config.paths:
            try:
                return ImageFont.truetype(path, font_config.size)
            except:
                continue
        
        print("⚠ 한글 폰트 로드 실패, 기본 폰트 사용")
        return ImageFont.load_default()
    
    def annotate(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                 confidence: float) -> np.ndarray:
        """프레임에 텍스트 주석 추가"""
        x, y = position
        label = f"{text} ({confidence:.2f})"
        if confidence < self.min_confidence:
            label += " - Low"
        
        bg_color = (0, 255, 0) if confidence >= self.min_confidence else (180, 182, 255)
        
        # 텍스트 크기
        bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), label, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 배경 사각형
        text_y = y - 40 if y - 40 > 0 else y + 40
        bg_x1, bg_y1 = x, text_y - text_height - 5
        bg_x2, bg_y2 = x + text_width + 10, text_y + 5
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        
        # 한글 텍스트
        frame = self._put_korean_text(frame, label, (x, text_y - text_height))
        
        return frame
    
    def _put_korean_text(self, img: np.ndarray, text: str, position: Tuple[int, int]) -> np.ndarray:
        """한글 텍스트 추가"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=self.font, fill=(0, 0, 0))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)