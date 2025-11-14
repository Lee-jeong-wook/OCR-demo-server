"""Base model interfaces"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseOCRModel(ABC):
    """OCR 모델 추상 클래스"""
    
    @abstractmethod
    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """번호판 이미지를 인식하여 텍스트와 신뢰도 반환"""
        pass