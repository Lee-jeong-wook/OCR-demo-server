"""Configuration management module"""

import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str
    port: int
    debug: bool
    secret_key: str
    max_buffer_size: int


@dataclass
class YOLOConfig:
    path: str
    confidence: float


@dataclass
class OCRModelConfig:
    path: str
    img_height: int
    img_width: int
    input_channel: int
    output_channel: int
    hidden_size: int
    num_fiducial: int
    transformation: str
    feature_extraction: str
    sequence_modeling: str
    prediction: str
    batch_max_length: int
    characters: str
    num_class: int = 0  # Will be calculated


@dataclass
class ProcessingConfig:
    min_confidence: float
    plate_padding_ratio: float
    plate_scale_factor: int
    jpeg_quality: int


@dataclass
class FontConfig:
    paths: list
    size: int


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self._config_path = Path(config_path)
        self._data = self._load_config()
        
        # Initialize config objects
        self.server = self._init_server_config()
        self.yolo = self._init_yolo_config()
        self.ocr = self._init_ocr_config()
        self.processing = self._init_processing_config()
        self.fonts = self._init_font_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def _init_server_config(self) -> ServerConfig:
        data = self._data['server']
        return ServerConfig(**data)
    
    def _init_yolo_config(self) -> YOLOConfig:
        data = self._data['models']['yolo']
        return YOLOConfig(**data)
    
    def _init_ocr_config(self) -> OCRModelConfig:
        data = self._data['models']['ocr']
        config_data = data['config']
        return OCRModelConfig(
            path=data['path'],
            img_height=config_data['img_height'],
            img_width=config_data['img_width'],
            input_channel=config_data['input_channel'],
            output_channel=config_data['output_channel'],
            hidden_size=config_data['hidden_size'],
            num_fiducial=config_data['num_fiducial'],
            transformation=config_data['transformation'],
            feature_extraction=config_data['feature_extraction'],
            sequence_modeling=config_data['sequence_modeling'],
            prediction=config_data['prediction'],
            batch_max_length=config_data['batch_max_length'],
            characters=config_data['characters']
        )
    
    def _init_processing_config(self) -> ProcessingConfig:
        data = self._data['processing']
        return ProcessingConfig(**data)
    
    def _init_font_config(self) -> FontConfig:
        data = self._data['fonts']
        return FontConfig(**data)