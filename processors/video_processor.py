"""Video processing logic"""

import cv2
import numpy as np
import base64
import time
from pathlib import Path
from typing import Set, Tuple, List

from models.detector import PlateDetector
from models.base import BaseOCRModel
from utils.plate_extractor import PlateExtractor
from utils.annotator import FrameAnnotator
from utils.data_classes import VideoInfo, DetectionResult
from config import ProcessingConfig
import time


class VideoProcessor:
    """비디오 처리 메인 클래스"""
    
    def __init__(self, detector: PlateDetector, ocr_model: BaseOCRModel, 
                 annotator: FrameAnnotator, config: ProcessingConfig):
        self.detector = detector
        self.ocr_model = ocr_model
        self.annotator = annotator
        self.extractor = PlateExtractor(config)
        self.min_confidence = config.min_confidence
        self.jpeg_quality = config.jpeg_quality
    
    def process_video(self, video_path: str, session_id: str, socketio):
        """비디오 처리 및 실시간 전송"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            socketio.emit('error', {'message': 'Could not open video'}, room=session_id)
            return
        
        try:
            video_info = self._get_video_info(cap)
            socketio.emit('video_info', {
                'duration': video_info.duration
            }, room=session_id)
            
            stats = self._process_frames(cap, video_info, session_id, socketio)
            self._send_completion(stats, session_id, socketio)
            
        finally:
            cap.release()
            self._cleanup(video_path)
    
    @staticmethod
    def _get_video_info(cap: cv2.VideoCapture) -> VideoInfo:
        """비디오 정보 추출"""
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = int(total_frames / fps) if(total_frames > 0) else 0
        
        return VideoInfo(width, height, duration, fps)
    
    def _process_frames(self, cap: cv2.VideoCapture, video_info: VideoInfo, 
                       session_id: str, socketio) -> dict:
        """프레임 처리 루프"""
        video_start_time = time.time()
        frame_count = 0
        detected_plates: Set[str] = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            annotated_frame, detections = self._process_single_frame(
                frame, detected_plates
            )

            self._send_frame(
                annotated_frame,
                detections, detected_plates,
                session_id, socketio
            )
            
            time.sleep(1.0 / video_info.fps)
        
        return {
            'detected_plates': detected_plates,
            'video_play_time': int(time.time() - video_start_time)
        }

    
    def _process_single_frame(self, frame: np.ndarray, detected_plates: Set[str]) -> Tuple:
        """단일 프레임 처리"""
        annotated_frame, boxes = self.detector.detect(frame)
        
        detections = []
        
        for box in boxes:
            plate_img = self.extractor.extract(frame, box)
            if plate_img is None:
                continue
            
            text, confidence = self.ocr_model.recognize(plate_img)
            
            if text:
                detected_plates.add(text)
                # print(f"인식: {text} (신뢰도: {confidence:.3f})")
            
            status = "success" if confidence >= self.min_confidence else "low_confidence"
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            annotated_frame = self.annotator.annotate(
                annotated_frame, text, (x1, y1), confidence
            )
            
            cls_id = int(box.cls[0])
            detections.append(DetectionResult(
                class_name=self.detector.model.names[cls_id],
                plate_text=text,
                confidence=round(confidence, 2),
                bbox=[x1, y1, x2, y2],
                status=status
            ).__dict__)
        
        return annotated_frame, detections
    
    def _send_frame(self, annotated_frame: np.ndarray,
                    detections: List, detected_plates: Set,
                    session_id: str, socketio):
        """프레임 데이터 전송"""
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        socketio.emit('frame', {
            'frame': frame_base64,

            'detections': detections,
            'stats': {
                'total_detected': len(detected_plates)
            }
        }, room=session_id)
    
    @staticmethod
    def _send_completion(stats: dict, session_id: str, socketio):
        """완료 신호 전송"""
        socketio.emit('completed', {
            'message': 'Processing completed',
            'total_plates': len(stats['detected_plates']),
            'plates': list(stats['detected_plates']),
            'video_play_time': stats['video_play_time']
        }, room=session_id)
    
    @staticmethod
    def _cleanup(video_path: str):
        """임시 파일 정리"""
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"파일 삭제 실패: {e}")