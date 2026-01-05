"""Flask application main"""

from flask import Flask, request, jsonify, send_file
from flask_sock import Sock
import base64
import threading
import time
import json
import uuid

from config import Config
from models.detector import PlateDetector
from models.ocr_model import CustomOCRModel
from utils.annotator import FrameAnnotator
from processors.video_processor import VideoProcessor
from processors.picture_processor import PictureProcessor
from werkzeug.utils import secure_filename
from flask_cors import CORS

import os


class PlateRecognitionApp:
    """Flask 애플리케이션"""
    
    def __init__(self, config: Config):
        self.config = config

        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = config.server.secret_key
        
        CORS(self.app, resources={r"/*": {"origins": "*"}})
        
        # Flask-Sock 초기화
        self.sock = Sock(self.app)
        
        # 모델 초기화
        self.detector = PlateDetector(config.yolo)
        self.ocr_model = CustomOCRModel(config.ocr)
        self.annotator = FrameAnnotator(config.fonts, config.processing)
        self.processor = VideoProcessor(
            self.detector, 
            self.ocr_model, 
            self.annotator,
            config.processing
        )

        self.pic_processor = PictureProcessor(
            self.detector, 
            self.ocr_model, 
            self.annotator,
            config.processing
        )
        
        self.processing_sessions = {}
        self._register_handlers()
        self._http_handlers()
    
    def _register_handlers(self):
        """WebSocket 핸들러 등록"""
        
        @self.sock.route('/ws')
        def websocket_handler(ws):
            """WebSocket 연결 핸들러"""
            session_id = str(uuid.uuid4())
            print(f"Connected to server: {session_id}")
            self.processing_sessions[session_id] = {
                'ws': ws,
                'status': 'connected'
            }
            
            try:
                # 연결 성공 메시지 전송
                ws.send(json.dumps({
                    'type': 'connected',
                    'message': 'Connected to server',
                    'session_id': session_id
                }))
                
                while True:
                    try:
                        message = ws.receive(timeout=1)
                        if message:
                            # Check if message is bytes or string
                            if isinstance(message, bytes):
                                # Binary data - convert to string if it's text-based
                                try:
                                    message = message.decode('utf-8')
                                    data = json.loads(message)
                                    self._handle_websocket_message(data, session_id, ws)
                                except (UnicodeDecodeError, json.JSONDecodeError):
                                    # Actual binary data
                                    self._handle_binary_upload(message, session_id, ws)
                            else:
                                # String message
                                data = json.loads(message)
                                self._handle_websocket_message(data, session_id, ws)
                    except TimeoutError:
                        # 타임아웃은 정상 (keep-alive)
                        continue
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        continue
                    except Exception as e:
                        print(f"Error receiving message: {e}")
                        break
                        
            except Exception as e:
                print(f"WebSocket error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if session_id in self.processing_sessions:
                    del self.processing_sessions[session_id]
    
    def _handle_websocket_message(self, data: dict, session_id: str, ws):
        """WebSocket 메시지 처리"""
        msg_type = data.get('type')
        
        if msg_type == 'buffer_image':
            self._handle_buffer_image(data, session_id, ws)
        else:
            print(f"Unknown message type: {msg_type}")
    
    def _handle_buffer_image(self, data: dict, session_id: str, ws):
        """Base64로 인코딩된 이미지 처리"""
        print(f"📥 Received buffer_image (Base64) from {session_id}")
        
        try:
            # Base64 데이터 추출
            if 'data' not in data:
                raise ValueError("'data' field is missing")
            
            base64_data = data['data']
            image_bytes = base64.b64decode(base64_data)
            
            # 즉시 수신 확인
            ws.send(json.dumps({
                'type': 'received',
                'message': 'buffer_image_received',
                'session_id': session_id
            }))
            
            # 이미지 처리
            self._process_image_bytes(image_bytes, session_id, ws)
            
        except Exception as e:
            print(f"❌ Error in buffer_image handler: {e}")
            import traceback
            traceback.print_exc()
            ws.send(json.dumps({
                'type': 'error',
                'message': f'이미지 처리 실패: {str(e)}',
                'session_id': session_id
            }))

    def _http_handlers(self):
        """HTTP 이벤트 핸들러 등록"""
        
        @self.app.route('/process_images', methods=['POST'])
        def handle_process_images():
            try:
                # 파일 유효성 검사
                if 'images' not in request.files:
                    return jsonify({'error': '이미지 파일이 없습니다'}), 400
                print(request)
                files = request.files.getlist('images')
                if not files:
                    return jsonify({'error': '이미지를 선택해주세요'}), 400
                
                # 세션 ID 생성
                session_id = f"session_{int(time.time() * 1000)}"
                
                # 파일 저장
                saved_paths = []
                allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
                
                for file in files:
                    if file and '.' in file.filename:
                        ext = file.filename.rsplit('.', 1)[1].lower()
                        if ext in allowed_extensions:
                            filename = secure_filename(f"{session_id}_{file.filename}")
                            filepath = os.path.join('./tmp_pictures/', filename)
                            file.save(filepath)
                            saved_paths.append(filepath)
                
                if not saved_paths:
                    return jsonify({'error': '유효한 이미지 파일이 없습니다'}), 400
                
                # 이미지 처리 - base64로 받음
                zip_base64, chart_base64 = self.pic_processor.process_picture_to_zip(saved_paths, session_id)
                
                # JSON 응답으로 반환
                return jsonify({
                    'success': True,
                    'zip_file': zip_base64,
                    'chart_data': chart_base64,
                    'filename': f'plate_recognition_results_{session_id}.zip'
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _handle_binary_upload(self, binary_data: bytes, session_id: str, ws):
        """바이너리 이미지 데이터 업로드 처리"""
        print(f"📥 Received binary image data from {session_id}")
        print(f"Data size: {len(binary_data) if binary_data else 0} bytes")
        
        try:
            # 즉시 수신 확인
            ws.send(json.dumps({
                'type': 'received',
                'message': 'buffer_image_received',
                'session_id': session_id
            }))
            
            # 이미지 처리
            self._process_image_bytes(binary_data, session_id, ws)
            
        except Exception as e:
            print(f"❌ Error in binary upload handler: {e}")
            import traceback
            traceback.print_exc()
            ws.send(json.dumps({
                'type': 'error',
                'message': f'이미지 처리 실패: {str(e)}',
                'session_id': session_id
            }))
    
    def _process_image_bytes(self, image_bytes: bytes, session_id: str, ws):
        """이미지 바이트 데이터 처리"""
        import cv2
        import numpy as np
        from io import BytesIO
        
        try:
            # 바이트를 numpy 배열로 변환
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            # 이미지 처리
            self.processing_sessions[session_id]['status'] = 'processing'
            
            # 처리 시작 알림
            ws.send(json.dumps({
                'type': 'upload_success',
                'message': '업로드 성공, 처리 시작',
                'session_id': session_id
            }))
            
            # 이미지 처리 (단일 이미지)
            result = self.pic_processor._process_single_image(image)
            
            # 결과 전송
            annotated_image = result['frame']
            detections = result['detections']
            
            # 이미지를 JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', annotated_image, [cv2.IMWRITE_JPEG_QUALITY, self.config.processing.jpeg_quality])
            frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            ws.send(json.dumps({
                'type': 'frame',
                'frame': frame_base64,
                'detections': detections,
                'stats': {
                    'total_detected': len(result['detected_plates']),
                    'detected_plates': list(result['detected_plates'])
                }
            }))
            
            # 완료 메시지
            ws.send(json.dumps({
                'type': 'completed',
                'message': '이미지 처리 완료',
                'total_plates': len(result['detected_plates']),
                'plates': list(result['detected_plates'])
            }))
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
            ws.send(json.dumps({
                'type': 'error',
                'message': f'이미지 처리 중 오류: {str(e)}',
                'session_id': session_id
            }))
    
    def run(self):
        """애플리케이션 실행"""
        self.app.run(
            host=self.config.server.host, 
            port=self.config.server.port, 
            debug=self.config.server.debug
        )

config = Config('config.yaml')
plate_recognition_app = PlateRecognitionApp(config)

app = plate_recognition_app.app
sock = plate_recognition_app.sock


if __name__ == '__main__':
    plate_recognition_app.run()