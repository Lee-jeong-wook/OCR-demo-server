"""Data classes for type safety"""

from dataclasses import dataclass
from typing import List


@dataclass
class DetectionResult:
    """탐지 결과"""
    class_name: str
    plate_text: str
    confidence: float
    bbox: List[int]
    status: str


@dataclass
class VideoInfo:
    """비디오 정보"""
    width: int
    height: int
    duration: int
    fps: int