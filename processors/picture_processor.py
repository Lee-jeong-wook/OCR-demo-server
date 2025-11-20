"""Picture processing logic"""

import cv2
import numpy as np
import base64
import time
import zipfile
import os
from pathlib import Path
from typing import Set, Tuple, List
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.detector import PlateDetector
from models.base import BaseOCRModel
from utils.plate_extractor import PlateExtractor
from utils.annotator import FrameAnnotator
from utils.data_classes import VideoInfo, DetectionResult
from config import ProcessingConfig


class PictureProcessor:
    """사진/비디오 처리 메인 클래스"""
    
    def __init__(self, detector: PlateDetector, ocr_model: BaseOCRModel, 
                 annotator: FrameAnnotator, config: ProcessingConfig):
        self.detector = detector
        self.ocr_model = ocr_model
        self.annotator = annotator
        self.extractor = PlateExtractor(config)
        self.min_confidence = config.min_confidence
        self.jpeg_quality = config.jpeg_quality

    def process_picture_to_zip(self, pic_paths: List[str], session_id: str) -> BytesIO:
        frame_list = []
        total_success = 0
        total_fail = 0
        total_detected = 0
        
        try:
            
            for idx, pic_path in enumerate(pic_paths):
                
                # 이미지 읽기
                frame = cv2.imread(pic_path)
                if frame is None:
                    continue
                
                # 프레임 처리
                stats = self._process_single_image(frame)
                frame_list.append({
                    'frame': stats['frame'],
                    'filename': ''.join(stats['name']),
                    'detections': stats['detections'],
                    'detected_count': len(stats['detections'])
                })
                
                total_success += stats['success']
                total_fail += stats['fail']
                total_detected += stats['total']

            chart_base64 = self._create_chart(
                total_success, 
                total_fail, 
                total_detected
            )

            zip_buffer, chart_data = self._create_zip_in_memory(
                frame_list, 
                chart_base64, 
                total_success, 
                total_fail, 
                total_detected
            )
            
            return zip_buffer, chart_data
            
        except Exception as e:
            print(f"처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 임시 파일 정리
            self._cleanup_multiple(pic_paths)
    
    def _process_single_image(self, frame: np.ndarray) -> dict:
        """단일 이미지 처리"""
        detected_plates: Set[str] = set()
        annotated_frame, detections, success, fail, name, total = self._process_single_frame(
            frame, detected_plates
        )
        
        return {
            'frame': annotated_frame,
            'detections': detections,
            'detected_plates': detected_plates,
            'success': success,
            'fail': fail,
            'name': name,
            'total': total
        }
    
    def _process_single_frame(self, frame: np.ndarray, detected_plates: Set[str]) -> Tuple:
        """단일 프레임 처리"""
        # 번호판 탐지
        annotated_frame, boxes = self.detector.detect(frame)
        
        detections = []
        success = 0
        fail = 0

        detect_fail = 0

        name: Set[str] = set()

        successSet: Set[int] = set()
        failSet: Set[int] = set()

        if(len(boxes)==0):
            detect_fail += 1
        
        # 탐지된 각 번호판 처리
        for box in boxes:
            plate_img = self.extractor.extract(frame, box)
            
            # OCR 인식
            text, confidence = self.ocr_model.recognize(plate_img)
            
            # 결과 처리
            if text and confidence >= self.min_confidence:
                detected_plates.add(text)
                successSet.add(text)
                status = "success"
            elif text:
                failSet.add(text)
                status = "low_confidence"
            else:
                fail += 1
                status = "fail"
            name.add(text)
            
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 프레임에 주석 추가
            annotated_frame = self.annotator.annotate(
                annotated_frame, text if text else "Unknown", (x1, y1), confidence
            )
            
            # 탐지 결과 저장
            cls_id = int(box.cls[0])
            detections.append(DetectionResult(
                class_name=self.detector.model.names[cls_id],
                plate_text=text if text else "",
                confidence=round(float(confidence), 2),
                bbox=[x1, y1, x2, y2],
                status=status
            ).__dict__)
        
        return annotated_frame, detections, len(successSet), len(failSet) + fail, list(name), (len(successSet) + len(failSet) + fail + detect_fail)
    
    def _create_chart(self, success: int, fail: int, total: int) -> str:
        """통계 차트 생성"""
        try:
            # 이미지별로 번호판이 없는 경우
            no_detection = total - (success + fail) if (success + fail) <= total else 0
            
            # 데이터 준비
            sizes = []
            labels = []
            colors = []
            
            if success > 0:
                sizes.append(success)
                labels.append(f'Success ({success})')
                colors.append('#4CAF50')
            
            if fail > 0:
                sizes.append(fail)
                labels.append(f'Low Conf/Fail ({fail})')
                colors.append('#FF9800')
            
            if no_detection > 0:
                sizes.append(no_detection)
                labels.append(f'No Detection ({no_detection})')
                colors.append('#F44336')
            
            # 차트가 비어있는 경우 처리
            if not sizes:
                sizes = [1]
                labels = ['No Data']
                colors = ['#CCCCCC']
            
            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 파이 차트
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, 
                   startangle=90, textprops={'fontsize': 10})
            ax1.set_title(f'Recognition Results\nTotal Images: {total}', fontsize=14, fontweight='bold')
            
            # 막대 차트
            bar_container = ax2.bar(['Success', 'Fail', 'No Detection'], 
                   [success, fail, no_detection], 
                   color=['#4CAF50', '#FF9800', '#F44336'])
            ax2.set_ylabel('Count', fontsize=12)
            ax2.bar_label(bar_container, fmt='{:,.0f}')
            ax2.set_title('Detection Statistics', fontsize=14, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # 통계 텍스트 추가
            stats_text = f'Total Plates Detected: {total}\n'
            stats_text += f'Success Rate: {(success/total*100) if total > 0 else 0:.1f}%'
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            
            # 이미지로 변환
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            print(f"차트 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return ""

    @staticmethod
    def _cleanup_multiple(file_paths: List[str]):
        """여러 임시 파일 정리"""
        for path in file_paths:
            try:
                if os.path.exists(path):
                    Path(path).unlink()
            except Exception as e:
                print(f"파일 삭제 실패 ({path}): {e}")
    
    def _create_zip_in_memory(self, frame_list: List[dict], chart_base64: str, 
                         total_success: int, total_fail: int, total_detected: int) -> tuple:
        """처리된 이미지들을 메모리에서 ZIP 파일로 생성하고 base64로 반환"""
        try:
            zip_buffer = BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 처리된 이미지 저장
                for idx, item in enumerate(frame_list):
                    frame = item['frame']
                    original_filename = item['filename']

                    # 이미지 인코딩
                    success, buffer = cv2.imencode('.jpg', frame, [
                        cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality
                    ])

                    if success:
                        # 파일명에서 확장자 분리
                        name_without_ext = os.path.splitext(original_filename)[0]
                        filename = f"{idx+1:03d}_{name_without_ext}.jpg"
                        zipf.writestr(filename, buffer.tobytes())

                # 차트 이미지 추가
                if chart_base64:
                    try:
                        chart_data = base64.b64decode(chart_base64.split(',')[1])
                        zipf.writestr('statistics_chart.png', chart_data)
                    except Exception as e:
                        print(f"차트 저장 실패: {e}")

            # ZIP 파일을 base64로 인코딩
            zip_buffer.seek(0)
            zip_base64 = base64.b64encode(zip_buffer.read()).decode('utf-8')

            # zip_base64와 chart_base64를 함께 반환
            return zip_base64, chart_base64
        except Exception as e:
            print(f"ZIP 생성 실패: {e}")
            raise