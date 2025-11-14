"""Custom OCR model implementation"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import re
from typing import Tuple
from pathlib import Path
import sys

from models.base import BaseOCRModel
from config import OCRModelConfig

class CustomOCRModel(BaseOCRModel):
    def __init__(self, config: OCRModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"OCR Model using device: {self.device}")
        # ...
        self._add_easyocr_path()  # ← 1. 먼저 sys.path에 경로 추가
        self._load_model()
    
    def _add_easyocr_path(self):
        """EasyOCR 모듈 경로를 sys.path 맨 앞에 추가"""
        current_dir = Path(__file__).parent
        easyocr_path = current_dir / "easyocr_modules"

        if not easyocr_path.exists():
            raise FileNotFoundError(f"EasyOCR 모듈 폴더를 찾을 수 없습니다: {easyocr_path}")

        easyocr_str = str(easyocr_path.absolute())
        if easyocr_str in sys.path:
            sys.path.remove(easyocr_str)
        sys.path.insert(0, easyocr_str)
    
    def _load_model(self):
        """모델 로드"""
        try:
            # importlib를 사용하여 절대 경로에서 모듈 로드
            import importlib.util
            
            easyocr_path = Path(__file__).parent / "easyocr_modules"
            
            # model.py 로드
            model_spec = importlib.util.spec_from_file_location(
                "easyocr_model", 
                easyocr_path / "model.py"
            )
            easyocr_model = importlib.util.module_from_spec(model_spec)
            model_spec.loader.exec_module(easyocr_model)
            
            # utils.py 로드
            utils_spec = importlib.util.spec_from_file_location(
                "easyocr_utils", 
                easyocr_path / "utils.py"
            )
            easyocr_utils = importlib.util.module_from_spec(utils_spec)
            sys.modules['easyocr_utils'] = easyocr_utils  # 캐시에 추가
            utils_spec.loader.exec_module(easyocr_utils)
            
            Model = easyocr_model.Model
            AttnLabelConverter = easyocr_utils.AttnLabelConverter
            
            # Converter 초기화
            self.converter = AttnLabelConverter(self.config.characters)
            self.config.num_class = len(self.converter.character)
            
            # 모델 초기화
            opt = self._create_opt_object()
            model = Model(opt)
            self.model = torch.nn.DataParallel(model).to(self.device)
            
            # 체크포인트 로드
            checkpoint = torch.load(self.config.path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            print("✓ OCR 모델 로드 완료")
            
        except ImportError as e:
            raise RuntimeError(f"EasyOCR 모듈 임포트 실패: {e}\n"
                             f"models/easyocr_modules/ 폴더에 model.py와 utils.py가 있는지 확인하세요.")
        except Exception as e:
            raise RuntimeError(f"OCR 모델 로드 실패: {e}")
    
    def _create_opt_object(self):
        """설정 객체 생성"""
        class Opt:
            pass
        
        opt = Opt()
        opt.imgH = self.config.img_height
        opt.imgW = self.config.img_width
        opt.input_channel = self.config.input_channel
        opt.output_channel = self.config.output_channel
        opt.hidden_size = self.config.hidden_size
        opt.num_fiducial = self.config.num_fiducial
        opt.Transformation = self.config.transformation
        opt.FeatureExtraction = self.config.feature_extraction
        opt.SequenceModeling = self.config.sequence_modeling
        opt.Prediction = self.config.prediction
        opt.batch_max_length = self.config.batch_max_length
        opt.num_class = self.config.num_class
        
        return opt
    
    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """번호판 인식"""
        try:
            processed_img = self._preprocess_image(image)
            
            with torch.no_grad():
                length_for_pred = torch.IntTensor([self.config.batch_max_length]).to(self.device)
                text_for_pred = torch.LongTensor(1, self.config.batch_max_length + 1).fill_(0).to(self.device)
                preds = self.model(processed_img, text_for_pred)

                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, preds_index = preds_prob.max(dim=2)
                
                preds_str = self.converter.decode(preds_index, length_for_pred)
                confidence = preds_max_prob.cumprod(dim=1)[:, -1].item()
            
            cleaned_text = self._postprocess_text(preds_str[0])
            return cleaned_text, confidence
            
        except Exception as e:
            print(f"OCR 인식 오류: {e}")
            return "", 0.0
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """이미지 전처리"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        img = Image.fromarray(gray).resize(
            (self.config.img_width, self.config.img_height), 
            Image.BICUBIC
        )
        img = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return img_tensor
    
    @staticmethod
    def _postprocess_text(text: str) -> str:
        """텍스트 후처리"""
        return re.sub(r'\[.*?\]', '', text)