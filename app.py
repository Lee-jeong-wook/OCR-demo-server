"""Flask application main"""

from flask import Flask, request
from flask_socketio import SocketIO, emit
import base64
import threading
import time

from config import Config
from models.detector import PlateDetector
from models.ocr_model import CustomOCRModel
from utils.annotator import FrameAnnotator
from processors.video_processor import VideoProcessor


class PlateRecognitionApp:
    """Flask 애플리케이션"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Flask 초기화
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = config.server.secret_key
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*", 
            max_http_buffer_size=config.server.max_buffer_size
        )
        
        # 모델 초기화
        print("모델 로딩 중...")
        self.detector = PlateDetector(config.yolo)
        self.ocr_model = CustomOCRModel(config.ocr)
        self.annotator = FrameAnnotator(config.fonts, config.processing)
        self.processor = VideoProcessor(
            self.detector, 
            self.ocr_model, 
            self.annotator,
            config.processing
        )
        
        self.processing_sessions = {}
        self._register_handlers()
        
        print("애플리케이션 초기화 완료")
    
    def _register_handlers(self):
        """SocketIO 이벤트 핸들러 등록"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f'클라이언트 연결: {request.sid}')
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f'클라이언트 연결 해제: {request.sid}')
            if request.sid in self.processing_sessions:
                del self.processing_sessions[request.sid]
        
        @self.socketio.on('upload_video')
        def handle_upload_video(data):
            self._handle_video_upload(data)
    
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

            process_time = time.time()
            
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


# Flask CLI를 위한 앱 팩토리 함수
def create_app():
    """Flask 앱 팩토리"""
    config = Config('config.yaml')
    plate_app = PlateRecognitionApp(config)
    return plate_app.app


# SocketIO도 함께 export
def create_socketio():
    """SocketIO 인스턴스 생성"""
    config = Config('config.yaml')
    plate_app = PlateRecognitionApp(config)
    return plate_app.socketio


# Flask CLI용 전역 변수
config = Config('./config.yaml')
plate_recognition_app = PlateRecognitionApp(config)
app = plate_recognition_app.app
socketio = plate_recognition_app.socketio


if __name__ == '__main__':
    plate_recognition_app.run()