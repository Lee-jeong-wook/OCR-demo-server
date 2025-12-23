"""Flask application main"""

from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import base64
import threading
import time

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
        
        # SocketIO 초기화
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*", 
            max_http_buffer_size=config.server.max_buffer_size,
            async_mode='threading'
        )
        
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
        """SocketIO 이벤트 핸들러 등록"""
        
        @self.socketio.on('connect')
        def handle_connect():
            # print(f'클라이언트 연결: {request.sid}')
            emit('connected', {'message': 'Connected to server'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            # print(f'클라이언트 연결 해제: {request.sid}')
            if request.sid in self.processing_sessions:
                del self.processing_sessions[request.sid]
        
        @self.socketio.on('upload_video')
        def handle_upload_video(data):
            self._handle_video_upload(data)

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
    
    def _handle_video_upload(self, data):
        """비디오 업로드 처리"""
        try:
            session_id = request.sid
            video_data = base64.b64decode(data['video'].split(',')[1])
            
            temp_path = f'temp_video_{session_id}_{time.time()}.mp4'
            with open(temp_path, 'wb') as f:
                f.write(video_data)
            
            self.processing_sessions[session_id] = 'processing'
            
            thread = threading.Thread(
                target=self.processor.process_video,
                args=(temp_path, session_id, self.socketio)
            )
            thread.daemon = True
            thread.start()
            
            emit('upload_success', {'message': '업로드 성공, 처리 시작'})
            
        except Exception as e:
            emit('error', {'message': f'업로드 실패: {str(e)}'})
    
    def run(self):
        """애플리케이션 실행"""
        self.socketio.run(
            self.app, 
            host=self.config.server.host, 
            port=self.config.server.port, 
            debug=self.config.server.debug, 
            allow_unsafe_werkzeug=True
        )

config = Config('config.yaml')
plate_recognition_app = PlateRecognitionApp(config)

app = plate_recognition_app.app
socketio = plate_recognition_app.socketio


if __name__ == '__main__':
    plate_recognition_app.run()