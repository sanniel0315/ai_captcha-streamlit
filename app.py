#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit + CRNN模型整合 - 優化版布局設計"""

import streamlit as st
import os
import warnings
from PIL import Image
import re
import string
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# 環境設定
os.environ['TORCH_DISABLE_EXTENSIONS'] = '1'
warnings.filterwarnings('ignore')

# 頁面配置
st.set_page_config(
    page_title="CRNN AI Tool",  # 瀏覽器標籤簡潔標題
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 兼容性函數
def safe_rerun():
    """安全的重新運行函數，兼容不同版本的Streamlit"""
    try:
        if hasattr(st, 'rerun'):
            st.rerun()
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            if 'rerun_trigger' not in st.session_state:
                st.session_state.rerun_trigger = 0
            st.session_state.rerun_trigger += 1
    except Exception as e:
        pass

# 延遲導入 PyTorch
@st.cache_resource
def import_torch_modules():
    try:
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        return torch, nn, transforms
    except Exception as e:
        st.error(f"PyTorch 導入失敗: {e}")
        return None, None, None

# 模型配置
MODEL_PATHS = [
    "best_crnn_captcha_model.pth",
    r"C:\Users\User\Desktop\Python3.8\02_emnist\trained_models\best_crnn_captcha_model.pth",
    "model.pth", 
    "crnn_model.pth"
]

CHARACTERS = string.ascii_uppercase
CAPTCHA_LENGTH_EXPECTED = 4

DEFAULT_CONFIG = {
    'IMAGE_HEIGHT': 32,
    'IMAGE_WIDTH': 128,
    'INPUT_CHANNELS': 1,
    'SEQUENCE_LENGTH': CAPTCHA_LENGTH_EXPECTED,
    'NUM_CLASSES': len(CHARACTERS),
    'HIDDEN_SIZE': 256,
    'NUM_LAYERS': 2
}

CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}

# 優化版CSS - 更緊湊的設計
st.markdown("""
<style>
    /* 隱藏默認元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    /* 隱藏Streamlit默認標題 */
    h1[data-testid="stHeader"] {display: none;}
    .stApp > header {display: none;}
    .stApp > div[data-testid="stHeader"] {display: none;}
    
    /* 確保沒有頂部間距 */
    .main > div:first-child {margin-top: 0 !important; padding-top: 0 !important;}
    .block-container {padding-top: 0 !important; margin-top: 0 !important;}
    
    /* ========== 其他界面顏色區域 ========== */
    
    /* 全局背景顏色 */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;  /* 🎨 這裡改整體背景 */
        color: #ecf0f1;  /* 🎨 這裡改全局文字顏色 */
    }
    
    /* 頂部框架顏色 */
    .compact-header {
        background: linear-gradient(135deg, #2c3e50, #34495e);  /* 🎨 這裡改頂部框架背景 */
        border: 2px solid #34495e;  /* 🎨 這裡改頂部框架邊框 */
        box-shadow: 0 6px 25px rgba(0,0,0,0.3);  /* 🎨 這裡改頂部框架陰影 */
    }
    
    /* 輸入框顏色 */
    .stTextInput > div > div > input {
        background: white !important;  /* 🎨 這裡改輸入框背景 */
        color: #2c3e50 !important;  /* 🎨 這裡改輸入框文字顏色 */
        border: 3px solid #34495e !important;  /* 🎨 這裡改輸入框邊框 */
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #27ae60 !important;  /* 🎨 改為綠色聚焦邊框 */
        box-shadow: 0 0 12px rgba(39, 174, 96, 0.4) !important;  /* 🎨 改為綠色聚焦陰影 */
    }
    
    .stTextInput > div > div > input:hover {
        border-color: #3498db !important;  /* 🎨 這裡改輸入框hover邊框顏色 */
        box-shadow: 0 0 8px rgba(52, 152, 219, 0.3) !important;  /* 🎨 這裡改hover陰影顏色 */
    }
    
    /* 進度條顏色 - 改為綠色為主 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #27ae60, #f39c12, #2ecc71) !important;  /* 🎨 綠色為主的進度條 */
    }
    
    /* 成功/錯誤訊息顏色 */
    .stSuccess {
        background: linear-gradient(135deg, rgba(39, 174, 96, 0.15), rgba(46, 204, 113, 0.1)) !important;  /* 🎨 成功訊息背景 */
        border: 2px solid #27ae60 !important;  /* 🎨 成功訊息邊框 */
        color: #27ae60 !important;  /* 🎨 成功訊息文字顏色 */
        box-shadow: 0 2px 8px rgba(39, 174, 96, 0.2) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(39, 174, 96, 0.15), rgba(46, 204, 113, 0.1)) !important;  /* 🎨 改為綠色系錯誤訊息 */
        border: 2px solid #1e8449 !important;  /* 🎨 改為深綠色邊框 */
        color: #1e8449 !important;  /* 🎨 改為深綠色文字 */
        box-shadow: 0 2px 8px rgba(39, 174, 96, 0.2) !important;
    }
    
    /* 狀態指示器顏色 */
    .status-compact {
        background: rgba(39, 174, 96, 0.15);  /* 🎨 狀態指示器背景 */
        border: 1px solid #27ae60;  /* 🎨 狀態指示器邊框 */
        color: #ecf0f1;  /* 🎨 狀態指示器文字顏色 */
    }
    
    .status-compact.error {
        background: rgba(231, 76, 60, 0.15);  /* 🎨 錯誤狀態背景 */
        border-color: #e74c3c;  /* 🎨 錯誤狀態邊框 */
        color: #e74c3c;  /* 🎨 錯誤狀態文字顏色 */
    }
    
    /* 緊湊的頂部區域 */
    .compact-header {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        border-radius: 15px;
        padding: 20px 25px;
        margin: 10px 0 20px 0;
        box-shadow: 0 6px 25px rgba(0,0,0,0.3);
        border: 2px solid #34495e;
    }
    
    /* 主標題現在使用內聯樣式，這些類別用於其他地方 */
    .section-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #ecf0f1;
        margin: 15px 0 10px 0;
        text-align: center;
    }
    
    .section-subtitle {
        font-size: 0.85rem;
        color: #3498db;
        text-align: center;
        margin: 0 0 15px 0;
        font-weight: 500;
    }
    
    /* 狀態指示器 - 更緊湊 */
    .status-compact {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(39, 174, 96, 0.15);
        border: 1px solid #27ae60;
        border-radius: 6px;
        padding: 6px 10px;
        margin: 4px 0;
        font-size: 0.8rem;
    }
    
    .status-compact.error {
        background: rgba(231, 76, 60, 0.15);
        border-color: #e74c3c;
        color: #e74c3c;
    }
    
    /* 路徑控制區域 - 單行布局 */
    .path-control-row {
        display: flex;
        gap: 8px;
        align-items: center;
        margin: 8px 0;
    }
    
    .path-buttons-compact {
        display: flex;
        gap: 4px;
    }
    
    /* 工作區域 - 最大化垂直空間 */
    .work-area-maximized {
        height: calc(100vh - 220px);
        min-height: 500px;
        margin-top: 8px;
    }
    
    /* 三欄面板 - 優化高度 */
    .panel-maximized {
        background: #2c3e50;
        border-radius: 10px;
        height: 100%;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        display: flex;
        flex-direction: column;
    }
    
    .panel-header-compact {
        background: linear-gradient(135deg, #34495e, #2c3e50);
        padding: 8px 12px;
        font-size: 0.9rem;
        font-weight: bold;
        text-align: center;
        color: #ecf0f1;
        border-radius: 10px 10px 0 0;
        flex-shrink: 0;
    }
    
    .panel-content-maximized {
        flex: 1;
        overflow-y: auto;
        padding: 8px;
        min-height: 0;
    }
    
    /* 圖片列表項目 - 更緊湊 */
    .image-item-compact {
        padding: 4px 8px;
        margin: 2px 0;
        background: #34495e;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        font-family: 'Consolas', monospace;
        font-size: 0.75rem;
        line-height: 1.2;
    }
    
    .image-item-compact:hover {
        background: #3498db;
        transform: translateX(3px);
    }
    
    .image-item-compact.active {
        background: #e94560;
        border-color: #c0392b;
        box-shadow: 0 2px 8px rgba(233, 69, 96, 0.3);
    }
    
    /* ========== 按鈕顏色區域 ========== */
    
    /* 一般按鈕顏色 */
    .stButton > button {
        background: linear-gradient(135deg, #34495e, #2c3e50) !important;  /* 🎨 這裡改一般按鈕背景 */
        color: white !important;  /* 🎨 這裡改一般按鈕文字顏色 */
        border: none !important;
        border-radius: 6px !important;
        padding: 6px 8px !important;
        font-size: 0.75rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        min-height: 32px !important;
        position: relative !important;
        overflow: hidden !important;
        line-height: 1.2 !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;  /* 🎨 這裡改hover時的顏色 */
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3) !important;  /* 🎨 這裡改hover陰影顏色 */
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 6px rgba(52, 152, 219, 0.4) !important;  /* 🎨 這裡改點擊陰影顏色 */
    }
    
    /* 主要按鈕顏色（保存按鈕等） - 改為綠色系 */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #27ae60, #2ecc71) !important;  /* 🎨 改為綠色漸變 */
        font-weight: bold !important;
        font-size: 1rem !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        border: none !important;
        color: white !important;  /* 🎨 這裡改主要按鈕文字顏色 */
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3) !important;  /* 🎨 改為綠色陰影 */
        position: relative !important;
        overflow: hidden !important;
    }
    
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2ecc71, #1e8449) !important;  /* 🎨 改為綠色hover */
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(39, 174, 96, 0.4) !important;
    }
    
    /* 按鈕水波紋效果顏色 */
    .stButton > button:before {
        background: rgba(255, 255, 255, 0.2) !important;  /* 🎨 這裡改水波紋顏色 */
    }
    
    div[data-testid="stButton"] button[kind="primary"]:before {
        background: rgba(255, 255, 255, 0.3) !important;  /* 🎨 這裡改主要按鈕水波紋顏色 */
    }
    
    /* 輸入框 - 更大字體，與標題匹配 */
    .stTextInput > div > div > input {
        background: white !important;
        color: #2c3e50 !important;
        border: 3px solid #34495e !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-size: 1.4rem !important;
        font-weight: bold !important;
        text-align: center !important;
        min-height: 48px !important;
        letter-spacing: 4px !important;
        text-transform: uppercase !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #e94560 !important;
        box-shadow: 0 0 12px rgba(233, 69, 96, 0.4) !important;
        transform: scale(1.02) !important;
    }
    
    .stTextInput > div > div > input:hover {
        border-color: #3498db !important;
        box-shadow: 0 0 8px rgba(52, 152, 219, 0.3) !important;
    }
    
    /* 圖片顯示容器 */
    .image-display-container {
        text-align: center;
        background: white;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    
    /* 控制面板區塊 */
    .control-section {
        background: #16213e;
        border-radius: 8px;
        padding: 8px;
        margin: 4px 0;
    }
    
    .control-section h4 {
        font-size: 0.85rem;
        margin: 0 0 6px 0;
        color: #ecf0f1;
    }
    
    /* 統計顯示 */
    .stat-row {
        display: flex;
        justify-content: space-around;
        text-align: center;
        padding: 4px;
    }
    
    .stat-item-compact {
        flex: 1;
    }
    
    .stat-value-compact {
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 2px;
        color: #3498db;
    }
    
    .stat-label-compact {
        font-size: 0.7rem;
        color: #bdc3c7;
    }
    
    /* 滾動條美化 */
    .panel-content-maximized::-webkit-scrollbar {
        width: 6px;
    }
    
    .panel-content-maximized::-webkit-scrollbar-track {
        background: #34495e;
        border-radius: 3px;
    }
    
    .panel-content-maximized::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #e94560, #c0392b);
        border-radius: 3px;
    }
    
    /* 響應式調整 */
    @media (max-width: 1200px) {
        .work-area-maximized {
            height: calc(100vh - 200px);
            min-height: 400px;
        }
        
        .path-control-row {
            flex-direction: column;
            gap: 4px;
        }
    }
    
    /* 進度條樣式 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #e74c3c, #f39c12, #27ae60) !important;
    }
    
    /* 三欄標題特殊效果 */
    .panel-title-left {
        background: linear-gradient(135deg, #27ae60, #2ecc71) !important;
        border-left: 5px solid #1e8449 !important;
    }
    
    .panel-title-center {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;
        border-top: 5px solid #1f618d !important;
        border-bottom: 5px solid #1f618d !important;
    }
    
    .panel-title-right {
        background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
        border-right: 5px solid #a93226 !important;
    }
    
    /* 標題動畫效果 */
    .panel-title-center:hover {
        transform: scale(1.02) !important;
        transition: transform 0.3s ease !important;
    }
    
    .panel-title-left:hover, .panel-title-right:hover {
        transform: translateY(-2px) !important;
        transition: transform 0.3s ease !important;
    }
    
    /* 成功/錯誤訊息樣式 - 增強視覺效果 */
    .stSuccess, .stError, .stWarning, .stInfo {
        padding: 10px 15px !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        margin: 8px 0 !important;
        font-weight: 500 !important;
        animation: fadeInUp 0.3s ease !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(39, 174, 96, 0.15), rgba(46, 204, 113, 0.1)) !important;
        border: 2px solid #27ae60 !important;
        color: #27ae60 !important;
        box-shadow: 0 2px 8px rgba(39, 174, 96, 0.2) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.1)) !important;
        border: 2px solid #e74c3c !important;
        color: #e74c3c !important;
        box-shadow: 0 2px 8px rgba(231, 76, 60, 0.2) !important;
    }
    
    /* 淡入動畫 */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* 保存按鈕成功狀態 - 改為綠色 */
    .stButton > button.success-pulse {
        animation: successPulse 0.6s ease !important;
    }
    
    @keyframes successPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); box-shadow: 0 0 20px rgba(39, 174, 96, 0.6); }  /* 🎨 改為綠色脈衝 */
        100% { transform: scale(1); }
    }
    
    /* 確保沒有遺漏的紅色hover效果 */
    .stWarning {
        background: linear-gradient(135deg, rgba(243, 156, 18, 0.15), rgba(230, 126, 34, 0.1)) !important;
        border: 2px solid #f39c12 !important;
        color: #f39c12 !important;
        box-shadow: 0 2px 8px rgba(243, 156, 18, 0.2) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.15), rgba(41, 128, 185, 0.1)) !important;
        border: 2px solid #3498db !important;
        color: #3498db !important;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

class SimpleCaptchaCorrector:
    @staticmethod
    def extract_label_from_filename(filename: str) -> str:
        name_without_ext, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            return ""
        match = re.search(r'([A-Z]{4})', name_without_ext)
        return match.group(1).upper() if match else ""

    @staticmethod
    def validate_label(label: str) -> bool:
        return bool(re.fullmatch(r'[A-Z]{4}', label))

    @staticmethod
    def generate_new_filename(new_label: str) -> str:
        return f"{new_label}.png"

def create_crnn_model():
    torch, nn, transforms = import_torch_modules()
    if torch is None:
        return None
    
    class CRNN(nn.Module):
        def __init__(self, img_height, img_width, num_classes, hidden_size=256, num_layers=2):
            super(CRNN, self).__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(512, 512, 2, padding=0), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
            )
            self.rnn = nn.LSTM(512, hidden_size, num_layers, bidirectional=True, batch_first=True)
            self.classifier = nn.Linear(hidden_size * 2, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            conv_features = self.cnn(x)
            conv_features = conv_features.squeeze(2).permute(0, 2, 1)
            rnn_output, _ = self.rnn(conv_features)
            rnn_output = self.dropout(rnn_output)
            output = self.classifier(rnn_output)
            return output
    
    return CRNN

class CRNNPredictor:
    def __init__(self):
        self.torch, self.nn, self.transforms = import_torch_modules()
        if self.torch is None:
            self.device = None
            self.model = None
            self.transform = None
            self.config = None
            self.is_loaded = False
            self.model_info = {}
            return
            
        self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.config = None
        self.is_loaded = False
        self.model_info = {}

    def load_model(self, model_path: str):
        if self.torch is None:
            return False
            
        try:
            if not os.path.exists(model_path):
                return False

            checkpoint = self.torch.load(model_path, map_location=self.device)
            
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                self.config = DEFAULT_CONFIG.copy()

            for key, val in DEFAULT_CONFIG.items():
                self.config.setdefault(key, val)

            CRNN = create_crnn_model()
            if CRNN is None:
                return False
                
            self.model = CRNN(
                img_height=self.config['IMAGE_HEIGHT'],
                img_width=self.config['IMAGE_WIDTH'],
                num_classes=self.config['NUM_CLASSES'],
                hidden_size=self.config['HIDDEN_SIZE'],
                num_layers=self.config['NUM_LAYERS']
            ).to(self.device)

            if 'model_state_dict' in checkpoint:
                sd_key = 'model_state_dict'
            elif 'state_dict' in checkpoint:
                sd_key = 'state_dict'
            else:
                return False

            self.model.load_state_dict(checkpoint[sd_key])
            self.model.eval()

            self.transform = self.transforms.Compose([
                self.transforms.Grayscale(self.config['INPUT_CHANNELS']),
                self.transforms.Resize((self.config['IMAGE_HEIGHT'], self.config['IMAGE_WIDTH'])),
                self.transforms.ToTensor(),
                self.transforms.Normalize([0.5] * self.config['INPUT_CHANNELS'], [0.5] * self.config['INPUT_CHANNELS'])
            ])

            self.is_loaded = True
            self.model_info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_val_captcha_acc': checkpoint.get('best_val_captcha_acc', 0),
                'idx_to_char': checkpoint.get('idx_to_char', IDX_TO_CHAR)
            }

            return True

        except Exception as e:
            return False

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        if not self.is_loaded or self.torch is None:
            return "", 0.0

        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with self.torch.no_grad():
                outputs = self.model(input_tensor)

            _, width_cnn_output, _ = outputs.shape
            seq_len = self.config['SEQUENCE_LENGTH']

            if width_cnn_output >= seq_len:
                start = (width_cnn_output - seq_len) // 2
                focused = outputs[:, start:start + seq_len, :]
            else:
                pad = seq_len - width_cnn_output
                focused = self.torch.cat([outputs, outputs[:, -1:, :].repeat(1, pad, 1)], dim=1)

            pred_indices = self.torch.argmax(focused, dim=2)[0]
            idx_to_char_map = self.model_info.get('idx_to_char', IDX_TO_CHAR)
            
            if isinstance(next(iter(idx_to_char_map.keys())), str):
                idx_to_char_map = {int(k): v for k, v in idx_to_char_map.items()}

            text = ''.join(idx_to_char_map.get(idx.item(), '?') for idx in pred_indices).upper()

            probs = self.torch.softmax(focused, dim=2)
            max_probs = self.torch.max(probs, dim=2)[0]
            confidence = float(self.torch.mean(max_probs).item())

            return text, confidence

        except Exception as e:
            return "", 0.0

def init_session_state():
    defaults = {
        'folder_images': [],
        'current_index': 0,
        'ai_predictions': {},
        'modified_count': 0,
        'modified_files': set(),
        'ai_accurate_count': 0,
        'folder_path': r"C:\Users\User\Desktop\Python3.8\02_emnist\debug_captchas_augmented_all_split\test",
        'temp_label': "",
        'list_page': 0,  # 添加列表分頁狀態
        'initialized': True
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

@st.cache_resource
def load_crnn_model():
    try:
        predictor = CRNNPredictor()
        
        model_path = None
        for file in MODEL_PATHS:
            if os.path.exists(file):
                model_path = file
                break
        
        if model_path is None:
            return None
        
        if predictor.load_model(model_path):
            return predictor
        else:
            return None
    except Exception as e:
        return None

def load_images_from_folder(folder_path: str):
    try:
        resolved_path = Path(folder_path).resolve()
        
        if not resolved_path.exists():
            st.error(f"❌ 路徑不存在: {resolved_path}")
            return False
            
        if not resolved_path.is_dir():
            st.error(f"❌ 路徑非資料夾: {resolved_path}")
            return False

        image_files_list = []
        for p in resolved_path.glob('*.png'):
            if p.is_file():
                image_files_list.append({
                    'name': p.name, 
                    'path': str(p),
                    'original_label': SimpleCaptchaCorrector.extract_label_from_filename(p.name)
                })
        
        image_files_list.sort(key=lambda x: x['name'])
        
        if not image_files_list:
            st.error(f"❌ 資料夾中沒有PNG檔案: {resolved_path}")
            return False

        st.session_state.folder_images = image_files_list
        st.session_state.current_index = 0
        st.session_state.ai_predictions = {}
        st.session_state.modified_count = 0
        st.session_state.modified_files = set()
        st.session_state.ai_accurate_count = 0
        st.session_state.temp_label = ""
        
        st.success(f"✅ 成功載入 {len(image_files_list)} 張PNG圖片")
        return True
        
    except Exception as e:
        st.error(f"❌ 載入圖片時異常: {e}")
        return False

def perform_batch_ai_prediction(predictor):
    if not st.session_state.folder_images or not predictor:
        return
    
    total_files = len(st.session_state.folder_images)
    batch_predictions = {}
    
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    for i, img_info in enumerate(st.session_state.folder_images):
        status_placeholder.info(f"🤖 AI識別中 ({i+1}/{total_files}): {img_info['name']}")
        progress_placeholder.progress((i + 1) / total_files, text=f"進度: {i+1}/{total_files}")
        
        try:
            image = Image.open(img_info['path'])
            predicted_text, confidence = predictor.predict(image)
            
            batch_predictions[i] = {
                'text': predicted_text,
                'confidence': confidence
            }
            
        except Exception as e:
            batch_predictions[i] = {'text': "ERROR", 'confidence': 0}
    
    st.session_state.ai_predictions = batch_predictions
    
    status_placeholder.success("🎯 AI批量識別完成！")
    progress_placeholder.empty()
    
    if st.session_state.folder_images:
        st.session_state.temp_label = get_default_label_for_current_image()

def get_default_label_for_current_image():
    if not st.session_state.folder_images:
        return ""
    
    current_idx = st.session_state.current_index
    current_img = st.session_state.folder_images[current_idx]
    
    # 如果已經有有效的temp_label，使用它
    if (hasattr(st.session_state, 'temp_label') and 
        st.session_state.temp_label and 
        SimpleCaptchaCorrector.validate_label(st.session_state.temp_label)):
        return st.session_state.temp_label
    
    # 使用AI預測結果（高置信度）
    if current_idx in st.session_state.ai_predictions:
        ai_pred = st.session_state.ai_predictions[current_idx]
        if (ai_pred['confidence'] > 0.7 and 
            SimpleCaptchaCorrector.validate_label(ai_pred['text'])):
            return ai_pred['text']
    
    # 使用從檔名提取的標籤
    if current_img.get('original_label'):
        return current_img['original_label']
    
    return ""

def navigate_to_image(new_index: int):
    """導航到指定的圖片索引，並自動調整分頁"""
    if not st.session_state.folder_images:
        return
    
    if 0 <= new_index < len(st.session_state.folder_images):
        st.session_state.current_index = new_index
        st.session_state.temp_label = get_default_label_for_current_image()
        
        # 自動跳轉到包含該圖片的頁面
        IMAGES_PER_PAGE = 20
        required_page = new_index // IMAGES_PER_PAGE
        st.session_state.list_page = required_page

def save_current_file(new_label: str):
    if not st.session_state.folder_images:
        return False
    
    current_idx = st.session_state.current_index
    current_file = st.session_state.folder_images[current_idx]
    
    if not SimpleCaptchaCorrector.validate_label(new_label):
        st.error("❌ 標籤必須為4個大寫英文字母")
        return False
    
    try:
        old_path = Path(current_file['path'])
        new_filename = SimpleCaptchaCorrector.generate_new_filename(new_label)
        new_path = old_path.parent / new_filename
        
        if old_path.resolve() == new_path.resolve():
            st.info(f"ℹ️ 檔名未變更: {new_filename}")
            return True
        
        if new_path.exists():
            st.warning(f"⚠️ 目標檔案 {new_filename} 已存在，將被覆寫")
        
        old_path.replace(new_path)
        
        original_label = current_file['original_label']
        if (st.session_state.ai_predictions.get(current_idx) and 
            st.session_state.ai_predictions[current_idx]['text'] == new_label and 
            original_label != new_label):
            st.session_state.ai_accurate_count += 1
        
        st.session_state.folder_images[current_idx] = {
            'name': new_filename,
            'path': str(new_path),
            'original_label': new_label
        }
        
        if current_idx not in st.session_state.modified_files:
            st.session_state.modified_count += 1
            st.session_state.modified_files.add(current_idx)
        
        st.success(f"✅ 檔案已改名為: {new_filename}")
        return True
        
    except Exception as e:
        st.error(f"❌ 保存失敗: {e}")
        return False

def render_compact_header(predictor):
    """渲染緊湊的頂部區域"""
    st.markdown('<div class="compact-header">', unsafe_allow_html=True)
    
    # 主標題 - 最大字體，特殊設計
    st.markdown('''
    <div style="
        text-align: center; 
        margin-bottom: 25px;
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        padding: 25px 30px;
        border-radius: 20px;
        border: 3px solid #3498db;
        box-shadow: 0 8px 30px rgba(52, 152, 219, 0.3);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50px;
            left: -50px;
            width: 100px;
            height: 100px;
            background: radial-gradient(circle, rgba(52,152,219,0.3), transparent);
            border-radius: 50%;
        "></div>
        <div style="
            position: absolute;
            bottom: -30px;
            right: -30px;
            width: 80px;
            height: 80px;
            background: radial-gradient(circle, rgba(39,174,96,0.2), transparent);  /* 🎨 改為綠色裝飾圓形 */
            border-radius: 50%;
        "></div>
        <h1 style="
            font-size: 3rem;
            font-weight: 900;
            margin: 0 0 10px 0;
            background: linear-gradient(45deg, #3498db, #27ae60, #2ecc71);  /* 🎨 改為藍綠漸變，移除紅色 */
            background-size: 300% 300%;
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 4s ease-in-out infinite;
            text-shadow: 0 4px 8px rgba(0,0,0,0.3);
            position: relative;
            z-index: 2;
            letter-spacing: 2px;
        ">🎯 AI驗證碼識別工具</h1>
        <p style="
            font-size: 1.2rem;
            color: #3498db;
            margin: 0;
            font-weight: 600;
            position: relative;
            z-index: 2;
            letter-spacing: 3px;
            text-transform: uppercase;
        ">CRNN模型 | 4位大寫英文字母識別</p>
        <div style="
            width: 60px;
            height: 4px;
            background: linear-gradient(90deg, #3498db, #27ae60);  /* 🎨 改為藍綠漸變，移除紅色 */
            margin: 15px auto 0;
            border-radius: 2px;
            position: relative;
            z-index: 2;
        "></div>
    </div>
    
    <style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    </style>
    ''', unsafe_allow_html=True)
    
    # AI狀態 - 單行顯示
    if predictor is not None:
        accuracy = predictor.model_info.get('best_val_captcha_acc', 0) * 100
        epoch = predictor.model_info.get('epoch', 'unknown')
        st.markdown(f'''
        <div class="status-compact">
            <span>🤖 模型已就緒</span>
            <span>準確率: {accuracy:.1f}%</span>
            <span>輪數: {epoch}</span>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-compact error">❌ 模型載入失敗</div>', unsafe_allow_html=True)
    
    # 路徑控制 - 水平布局
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 3, 1])
    
    with col1:
        if st.button("🖥️桌面", key="path_desktop", use_container_width=True):
            st.session_state.folder_path = r"C:\Users\User\Desktop"
            safe_rerun()
    with col2:
        if st.button("📥下載", key="path_downloads", use_container_width=True):
            st.session_state.folder_path = r"C:\Users\User\Downloads"
            safe_rerun()
    with col3:
        if st.button("🎯偵錯", key="path_debug", use_container_width=True):
            st.session_state.folder_path = r"C:\Users\User\Desktop\Python3.8\02_emnist\debug_captchas_adaptive_captcha_paper"
            safe_rerun()
    with col4:
        if st.button("🧪測試", key="path_test", use_container_width=True):
            st.session_state.folder_path = r"C:\Users\User\Desktop\Python3.8\02_emnist\debug_captchas_augmented_all_split\test"
            safe_rerun()
    with col5:
        folder_path = st.text_input(
            "路徑",
            value=st.session_state.folder_path,
            placeholder="PNG圖片資料夾路徑",
            key="folder_path_input",
            label_visibility="collapsed"
        )
        st.session_state.folder_path = folder_path
    with col6:
        if st.button("🚀載入", type="primary", key="load_images", use_container_width=True):
            if folder_path.strip():
                if load_images_from_folder(folder_path.strip()):
                    if st.session_state.folder_images and predictor:
                        with st.spinner("🤖 AI識別中..."):
                            perform_batch_ai_prediction(predictor)
                    safe_rerun()
            else:
                st.error("❌ 請輸入路徑")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_maximized_work_area(predictor):
    """渲染最大化的工作區域"""
    if not st.session_state.folder_images:
        # 顯示明顯的提示信息
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 80px 40px; 
            background: linear-gradient(135deg, #2c3e50, #34495e); 
            color: #ecf0f1;
            border-radius: 15px; 
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        ">
            <h2 style="color: #3498db; margin-bottom: 20px;">📂 開始使用 AI 驗證碼識別工具</h2>
            <p style="font-size: 1.1rem; margin-bottom: 15px; color: #ecf0f1;">請選擇包含 PNG 驗證碼圖片的資料夾</p>
            <p style="font-size: 0.9rem; color: #bdc3c7; margin-bottom: 25px;">
                💡 使用上方的快速按鈕（桌面、下載、偵錯、測試）<br>
                或手動輸入資料夾路徑，然後點擊「🚀載入」按鈕
            </p>
            <div style="
                background: rgba(52, 152, 219, 0.1); 
                border: 2px solid #3498db; 
                border-radius: 10px; 
                padding: 20px; 
                margin: 20px auto;
                max-width: 600px;
            ">
                <h4 style="color: #3498db; margin-bottom: 10px;">🎯 功能特色</h4>
                <ul style="text-align: left; color: #ecf0f1; line-height: 1.6;">
                    <li>🤖 <strong>AI自動識別</strong> - 使用CRNN模型識別4位大寫英文字母</li>
                    <li>📝 <strong>手動修正</strong> - 可以手動編輯AI識別結果</li>
                    <li>📊 <strong>即時統計</strong> - 顯示處理進度和AI準確率</li>
                    <li>⚡ <strong>快速導航</strong> - 支援圖片間快速切換</li>
                    <li>💾 <strong>自動保存</strong> - 修正後自動重命名檔案</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
def render_maximized_work_area(predictor):
    """渲染最大化的工作區域"""
    if not st.session_state.folder_images:
        # 顯示明顯的提示信息
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 80px 40px; 
            background: linear-gradient(135deg, #2c3e50, #34495e); 
            color: #ecf0f1;
            border-radius: 15px; 
            margin: 20px 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        ">
            <h2 style="color: #3498db; margin-bottom: 20px;">📂 開始使用 AI 驗證碼識別工具</h2>
            <p style="font-size: 1.1rem; margin-bottom: 15px; color: #ecf0f1;">請選擇包含 PNG 驗證碼圖片的資料夾</p>
            <p style="font-size: 0.9rem; color: #bdc3c7; margin-bottom: 25px;">
                💡 使用上方的快速按鈕（桌面、下載、偵錯、測試）<br>
                或手動輸入資料夾路徑，然後點擊「🚀載入」按鈕
            </p>
            <div style="
                background: rgba(52, 152, 219, 0.1); 
                border: 2px solid #3498db; 
                border-radius: 10px; 
                padding: 20px; 
                margin: 20px auto;
                max-width: 600px;
            ">
                <h4 style="color: #3498db; margin-bottom: 10px;">🎯 功能特色</h4>
                <ul style="text-align: left; color: #ecf0f1; line-height: 1.6;">
                    <li>🤖 <strong>AI自動識別</strong> - 使用CRNN模型識別4位大寫英文字母</li>
                    <li>📝 <strong>手動修正</strong> - 可以手動編輯AI識別結果</li>
                    <li>📊 <strong>即時統計</strong> - 顯示處理進度和AI準確率</li>
                    <li>⚡ <strong>快速導航</strong> - 支援圖片間快速切換</li>
                    <li>💾 <strong>自動保存</strong> - 修正後自動重命名檔案</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # 確保索引有效
    if st.session_state.current_index >= len(st.session_state.folder_images):
        st.session_state.current_index = 0
    
    # 使用容器確保正確的高度
    work_container = st.container()
    
    with work_container:
        # 三欄布局 - 使用固定比例
        col1, col2, col3 = st.columns([1, 2, 1], gap="medium")
        
        # 左側：圖片列表面板
        with col1:
            with st.container():
                # 標題 - 左側面板，🎨 綠色主題（可在這裡改顏色）
                st.markdown('''
                <div style="
                    background: linear-gradient(135deg, #27ae60, #2ecc71);  /* 🎨 左側標題背景漸變 */
                    color: white;  /* 🎨 左側標題文字顏色 */
                    text-align: center;
                    padding: 14px 18px;
                    border-radius: 10px;
                    margin-bottom: 18px;
                    box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);  /* 🎨 左側標題陰影顏色 */
                    border: 2px solid #27ae60;  /* 🎨 左側標題邊框顏色 */
                ">
                    <h2 style="
                        font-size: 1.4rem;
                        font-weight: bold;
                        margin: 0;
                        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    ">📋 圖片列表</h2>
                    <p style="
                        font-size: 0.8rem;
                        margin: 4px 0 0 0;
                        opacity: 0.9;
                    ">IMAGE LIST</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # 載入統計和分頁信息
                total_count = len(st.session_state.folder_images)
                ai_count = len(st.session_state.ai_predictions)
                
                # 分頁設置
                IMAGES_PER_PAGE = 20
                total_pages = (total_count + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE if total_count > 0 else 1
                current_page = st.session_state.get('list_page', 0)
                
                # 確保頁碼有效
                if current_page >= total_pages:
                    current_page = max(0, total_pages - 1)
                    st.session_state.list_page = current_page
                
                # 顯示統計和分頁信息
                st.caption(f"總數: {total_count} | AI識別: {ai_count} | 第 {current_page + 1}/{total_pages} 頁")
                
                # 分頁按鈕
                page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
                with page_col1:
                    if st.button("⬅️ 上一頁", disabled=current_page <= 0, key="list_prev_page", use_container_width=True):
                        st.session_state.list_page = max(0, current_page - 1)
                        safe_rerun()
                with page_col2:
                    st.markdown(f'<div style="text-align: center; padding: 6px; color: #bdc3c7; font-size: 0.8rem;">頁 {current_page + 1} / {total_pages}</div>', unsafe_allow_html=True)
                with page_col3:
                    if st.button("下一頁 ➡️", disabled=current_page >= total_pages - 1, key="list_next_page", use_container_width=True):
                        st.session_state.list_page = min(total_pages - 1, current_page + 1)
                        safe_rerun()
                
                # 計算當前頁的圖片範圍
                start_idx = current_page * IMAGES_PER_PAGE
                end_idx = min(start_idx + IMAGES_PER_PAGE, total_count)
                page_images = list(range(start_idx, end_idx))
                
                # 創建滾動容器
                list_container = st.container()
                with list_container:
                    # 圖片列表 - 兩列顯示，每頁20張
                    if page_images:
                        # 計算需要多少行（每行2個）
                        rows = (len(page_images) + 1) // 2
                        
                        for row in range(rows):
                            # 創建兩列
                            img_col1, img_col2 = st.columns(2, gap="small")
                            
                            # 左列圖片
                            left_idx_in_page = row * 2
                            if left_idx_in_page < len(page_images):
                                left_idx = page_images[left_idx_in_page]
                                with img_col1:
                                    img_info = st.session_state.folder_images[left_idx]
                                    ai_pred = st.session_state.ai_predictions.get(left_idx, {})
                                    original_label = img_info.get('original_label', '')
                                    is_current = left_idx == st.session_state.current_index
                                    
                                    # 緊湊的顯示格式
                                    original_display = original_label if original_label else "----"
                                    ai_display = ai_pred.get('text', '----') if ai_pred else '----'
                                    
                                    # 顯示置信度
                                    confidence = ai_pred.get('confidence', 0) if ai_pred else 0
                                    conf_indicator = f"({confidence:.0%})" if confidence > 0 else ""
                                    
                                    display_text = f"{original_display}|{ai_display}{conf_indicator}"
                                    
                                    button_type = "primary" if is_current else "secondary"
                                    
                                    if st.button(
                                        display_text,
                                        key=f"img_btn_{left_idx}_{current_page}",
                                        help=f"#{left_idx+1}: {img_info['name']}\n原始: {original_label or '無'}\nAI: {ai_pred.get('text', '未識別')}",
                                        type=button_type,
                                        use_container_width=True
                                    ):
                                        navigate_to_image(left_idx)
                                        safe_rerun()
                            
                            # 右列圖片
                            right_idx_in_page = row * 2 + 1
                            if right_idx_in_page < len(page_images):
                                right_idx = page_images[right_idx_in_page]
                                with img_col2:
                                    img_info = st.session_state.folder_images[right_idx]
                                    ai_pred = st.session_state.ai_predictions.get(right_idx, {})
                                    original_label = img_info.get('original_label', '')
                                    is_current = right_idx == st.session_state.current_index
                                    
                                    # 緊湊的顯示格式
                                    original_display = original_label if original_label else "----"
                                    ai_display = ai_pred.get('text', '----') if ai_pred else '----'
                                    
                                    # 顯示置信度
                                    confidence = ai_pred.get('confidence', 0) if ai_pred else 0
                                    conf_indicator = f"({confidence:.0%})" if confidence > 0 else ""
                                    
                                    display_text = f"{original_display}|{ai_display}{conf_indicator}"
                                    
                                    button_type = "primary" if is_current else "secondary"
                                    
                                    if st.button(
                                        display_text,
                                        key=f"img_btn_{right_idx}_{current_page}",
                                        help=f"#{right_idx+1}: {img_info['name']}\n原始: {original_label or '無'}\nAI: {ai_pred.get('text', '未識別')}",
                                        type=button_type,
                                        use_container_width=True
                                    ):
                                        navigate_to_image(right_idx)
                                        safe_rerun()
                    else:
                        st.info("此頁沒有圖片")
        
        # 中央：圖片預覽面板
        with col2:
            with st.container():
                # 標題 - 中央面板，🎨 藍色主題（可在這裡改顏色）
                st.markdown('''
                <div style="
                    background: linear-gradient(135deg, #3498db, #2980b9);  /* 🎨 中央標題背景漸變 */
                    color: white;  /* 🎨 中央標題文字顏色 */
                    text-align: center;
                    padding: 16px 22px;
                    border-radius: 12px;
                    margin-bottom: 22px;
                    box-shadow: 0 5px 18px rgba(52, 152, 219, 0.4);  /* 🎨 中央標題陰影顏色 */
                    border: 3px solid #2980b9;  /* 🎨 中央標題邊框顏色 */
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        position: absolute;
                        top: -50%;
                        right: -20px;
                        width: 80px;
                        height: 80px;
                        background: rgba(255,255,255,0.1);  /* 🎨 中央標題裝飾圓形顏色 */
                        border-radius: 50%;
                    "></div>
                    <h2 style="
                        font-size: 1.6rem;
                        font-weight: bold;
                        margin: 0;
                        text-shadow: 0 3px 6px rgba(0,0,0,0.4);
                        position: relative;
                        z-index: 2;
                    ">🖼️ 驗證碼預覽</h2>
                    <p style="
                        font-size: 0.85rem;
                        margin: 6px 0 0 0;
                        opacity: 0.9;
                        position: relative;
                        z-index: 2;
                        letter-spacing: 1px;
                    ">CAPTCHA PREVIEW</p>
                </div>
                ''', unsafe_allow_html=True)
                
                if st.session_state.folder_images:
                    current_img = st.session_state.folder_images[st.session_state.current_index]
                    
                    try:
                        image = Image.open(current_img['path'])
                        
                        # 圖片信息
                        st.caption(f"檔案: {current_img['name']}")
                        
                        # 圖片顯示
                        st.image(image, use_container_width=True)
                        
                        # 快速信息
                        current_idx = st.session_state.current_index
                        original_label = current_img.get('original_label', '')
                        
                        info_col1, info_col2, info_col3 = st.columns(3)
                        with info_col1:
                            st.metric("序號", f"#{current_idx + 1}")
                        with info_col2:
                            st.metric("原始標籤", original_label or "無")
                        with info_col3:
                            if current_idx in st.session_state.ai_predictions:
                                ai_pred = st.session_state.ai_predictions[current_idx]
                                st.metric("AI識別", f"{ai_pred['text']} ({ai_pred['confidence']:.0%})")
                            else:
                                st.metric("AI識別", "等待中")
                        
                    except Exception as e:
                        st.error(f"❌ 無法載入圖片: {str(e)}")
        
        # 右側：控制面板
        with col3:
            with st.container():
                # 標題 - 右側面板，🎨 綠色主題（替換原紅色）
                st.markdown('''
                <div style="
                    background: linear-gradient(135deg, #2ecc71, #27ae60);  /* 🎨 改為綠色漸變背景 */
                    color: white;  /* 🎨 右側標題文字顏色 */
                    text-align: center;
                    padding: 14px 18px;
                    border-radius: 10px;
                    margin-bottom: 18px;
                    box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);  /* 🎨 改為綠色陰影 */
                    border: 2px solid #1e8449;  /* 🎨 改為深綠色邊框 */
                    position: relative;
                ">
                    <div style="
                        position: absolute;
                        top: 5px;
                        right: 5px;
                        width: 8px;
                        height: 8px;
                        background: #3498db;  /* 🎨 改為藍色指示燈形成對比 */
                        border-radius: 50%;
                        animation: pulse 2s infinite;
                    "></div>
                    <h2 style="
                        font-size: 1.4rem;
                        font-weight: bold;
                        margin: 0;
                        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    ">⚙️ 控制面板</h2>
                    <p style="
                        font-size: 0.8rem;
                        margin: 4px 0 0 0;
                        opacity: 0.9;
                    ">CONTROL PANEL</p>
                </div>
                
                <style>
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                </style>
                ''', unsafe_allow_html=True)
                
                if st.session_state.folder_images:
                    current_idx = st.session_state.current_index
                    current_img = st.session_state.folder_images[current_idx]
                    
                    # AI識別結果
                    st.markdown("#### 🤖 AI識別")
                    
                    if current_idx in st.session_state.ai_predictions:
                        ai_pred = st.session_state.ai_predictions[current_idx]
                        confidence = ai_pred['confidence']
                        
                        st.info(f"AI結果: **{ai_pred['text']}**")
                        st.progress(confidence, text=f"置信度: {confidence:.1%}")
                        
                        if st.button("🎯 使用AI結果", key=f"ctrl_use_ai_{current_idx}", use_container_width=True):
                            if SimpleCaptchaCorrector.validate_label(ai_pred['text']):
                                # 設置AI結果到temp_label
                                st.session_state.temp_label = ai_pred['text']
                                # 增加觸發器來強制輸入框更新
                                trigger_key = f'update_input_{current_idx}'
                                st.session_state[trigger_key] = st.session_state.get(trigger_key, 0) + 1
                                # 顯示成功訊息並重新運行
                                st.success(f"✅ 已填入: {ai_pred['text']}")
                                safe_rerun()
                            else:
                                st.warning("⚠️ AI結果格式無效")
                    else:
                        st.info("等待AI識別...")
                    
                    # 標籤編輯
                    st.markdown("#### ✏️ 標籤編輯")
                    
                    # 初始化或更新temp_label
                    if not hasattr(st.session_state, 'temp_label') or not st.session_state.temp_label:
                        st.session_state.temp_label = get_default_label_for_current_image()
                    
                    # 創建一個強制更新的觸發器
                    update_trigger = st.session_state.get(f'update_input_{current_idx}', 0)
                    
                    # 輸入框 - 使用觸發器來強制更新
                    new_label = st.text_input(
                        "新標籤 (4位大寫字母)",
                        value=st.session_state.temp_label,
                        max_chars=4,
                        placeholder="ABCD",
                        key=f"ctrl_label_input_{current_idx}_v{update_trigger}",
                        help="輸入4個大寫英文字母作為驗證碼標籤"
                    ).upper()
                    
                    # 即時更新temp_label
                    st.session_state.temp_label = new_label
                    
                    st.session_state.temp_label = new_label
                    is_valid = SimpleCaptchaCorrector.validate_label(new_label)
                    
                    # 驗證狀態
                    if new_label:
                        if is_valid:
                            st.success("✅ 格式正確")
                        else:
                            st.error("❌ 需要4個大寫字母")
                    
                    # 保存按鈕
                    if st.button("💾 保存修改", disabled=not is_valid, type="primary", key=f"ctrl_save_{current_idx}", use_container_width=True):
                        if save_current_file(new_label):
                            if current_idx < len(st.session_state.folder_images) - 1:
                                new_idx = current_idx + 1
                                navigate_to_image(new_idx)
                                st.balloons()
                                safe_rerun()
                            else:
                                st.success("🎉 全部完成！")
                                st.balloons()
                    
                    # 導航
                    st.markdown("#### 🧭 導航")
                    
                    nav_col1, nav_col2 = st.columns(2)
                    with nav_col1:
                        if st.button("⬅️ 上一張", disabled=current_idx == 0, key=f"prev_{current_idx}", use_container_width=True):
                            new_idx = current_idx - 1
                            navigate_to_image(new_idx)
                            safe_rerun()
                    with nav_col2:
                        last_idx = len(st.session_state.folder_images) - 1
                        if st.button("下一張 ➡️", disabled=current_idx >= last_idx, key=f"next_{current_idx}", use_container_width=True):
                            new_idx = current_idx + 1
                            navigate_to_image(new_idx)
                            safe_rerun()
                    
                    # 導航區塊
                    st.markdown("#### 🧭 導航")
                    
                    nav_col1, nav_col2 = st.columns(2)
                    with nav_col1:
                        if st.button("⬅️ 上一張", disabled=current_idx == 0, key=f"nav_prev_{current_idx}", use_container_width=True):
                            new_idx = current_idx - 1
                            navigate_to_image(new_idx)
                            safe_rerun()
                    with nav_col2:
                        last_idx = len(st.session_state.folder_images) - 1
                        if st.button("下一張 ➡️", disabled=current_idx >= last_idx, key=f"nav_next_{current_idx}", use_container_width=True):
                            new_idx = current_idx + 1
                            navigate_to_image(new_idx)
                            safe_rerun()
                    
                    # 進度顯示
                    progress = (current_idx + 1) / len(st.session_state.folder_images)
                    st.progress(progress, text=f"進度: {current_idx + 1}/{len(st.session_state.folder_images)}")
                    
                    # 統計區塊
                    st.markdown("#### 📊 統計")
                    
                    # 使用簡潔的metric顯示
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("總檔案", len(st.session_state.folder_images))
                    with col_stat2:
                        st.metric("已修正", st.session_state.modified_count)
                    
                    # AI準確率單獨顯示
                    if st.session_state.modified_count > 0:
                        ai_acc = (st.session_state.ai_accurate_count / st.session_state.modified_count) * 100
                        st.metric("AI準確率", f"{ai_acc:.0f}%")
                    else:
                        st.metric("AI準確率", "0%")
                    
                    # 快速跳轉
                    if len(st.session_state.folder_images) > 10:
                        st.markdown("#### ⚡ 快速跳轉")
                        
                        jump_col1, jump_col2 = st.columns(2)
                        with jump_col1:
                            if st.button("🏠 首張", disabled=current_idx == 0, key=f"nav_jump_first_{current_idx}", use_container_width=True):
                                navigate_to_image(0)
                                safe_rerun()
                        with jump_col2:
                            last_idx = len(st.session_state.folder_images) - 1
                            if st.button("🏁 末張", disabled=current_idx == last_idx, key=f"nav_jump_last_{current_idx}", use_container_width=True):
                                navigate_to_image(last_idx)
                                safe_rerun()

def main():
    if 'initialized' not in st.session_state:
        init_session_state()
    
    # 載入模型
    predictor = load_crnn_model()
    
    # 緊湊的頂部區域
    render_compact_header(predictor)
    
    # 添加分隔線
    st.markdown("---")
    
    # 最大化的工作區域
    render_maximized_work_area(predictor)
    
    # 調試信息（可選，幫助診斷問題）
    if st.checkbox("顯示調試信息", key="debug_info"):
        st.write("調試信息:")
        st.write(f"folder_images 長度: {len(st.session_state.folder_images)}")
        st.write(f"current_index: {st.session_state.current_index}")
        st.write(f"ai_predictions 長度: {len(st.session_state.ai_predictions)}")
        st.write(f"folder_path: {st.session_state.folder_path}")

if __name__ == "__main__":
    main()