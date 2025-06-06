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
    page_title="AI驗證碼識別工具",  # 移除emoji，只用於瀏覽器標籤
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
    
    /* 全局樣式 - 深藍色主題，最小化間距 */
    .main .block-container {
        padding: 0.5rem !important;
        max-width: 100% !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
        color: #ecf0f1;
    }
    
    /* 緊湊的頂部區域 */
    .compact-header {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0 16px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        border: 1px solid #34495e;
    }
    
    .compact-title {
        font-size: 1.6rem;
        font-weight: bold;
        color: #ecf0f1;
        margin: 0 0 6px 0;
        text-align: center;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .compact-subtitle {
        font-size: 0.9rem;
        color: #3498db;
        text-align: center;
        margin: 0;
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
    
    /* 按鈕樣式 - 增強互動 */
    .stButton > button {
        background: linear-gradient(135deg, #34495e, #2c3e50) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        min-height: 36px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 6px rgba(52, 152, 219, 0.4) !important;
    }
    
    /* 按鈕水波紋效果 */
    .stButton > button:before {
        content: '' !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        width: 0 !important;
        height: 0 !important;
        background: rgba(255, 255, 255, 0.2) !important;
        border-radius: 50% !important;
        transform: translate(-50%, -50%) !important;
        transition: width 0.2s ease, height 0.2s ease !important;
    }
    
    .stButton > button:active:before {
        width: 80px !important;
        height: 80px !important;
    }
    
    /* 主要按鈕 - 增強互動效果 */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #e94560, #c0392b) !important;
        font-weight: bold !important;
        font-size: 1rem !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        border: none !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.3) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #c0392b, #a93226) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.4) !important;
    }
    
    div[data-testid="stButton"] button[kind="primary"]:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(233, 69, 96, 0.5) !important;
    }
    
    /* 按鈕點擊動畫效果 */
    div[data-testid="stButton"] button[kind="primary"]:before {
        content: '' !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        width: 0 !important;
        height: 0 !important;
        background: rgba(255, 255, 255, 0.3) !important;
        border-radius: 50% !important;
        transform: translate(-50%, -50%) !important;
        transition: width 0.3s ease, height 0.3s ease !important;
    }
    
    div[data-testid="stButton"] button[kind="primary"]:active:before {
        width: 100px !important;
        height: 100px !important;
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
    
    /* 保存按鈕成功狀態 */
    .stButton > button.success-pulse {
        animation: successPulse 0.6s ease !important;
    }
    
    @keyframes successPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); box-shadow: 0 0 20px rgba(39, 174, 96, 0.6); }
        100% { transform: scale(1); }
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
    
    # 標題 - 放在框內
    st.markdown('''
    <div style="text-align: center; margin-bottom: 15px;">
        <div class="compact-title">🎯 AI驗證碼識別工具</div>
        <div class="compact-subtitle">CRNN模型 | 4位大寫英文字母識別</div>
    </div>
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
                st.markdown("### 📋 圖片列表")
                
                # 載入統計
                total_count = len(st.session_state.folder_images)
                ai_count = len(st.session_state.ai_predictions)
                st.caption(f"總數: {total_count} | AI識別: {ai_count}")
                
                # 創建滾動容器
                list_container = st.container()
                with list_container:
                    # 圖片列表
                    display_count = min(50, total_count)
                    
                    for i in range(display_count):
                        img_info = st.session_state.folder_images[i]
                        ai_pred = st.session_state.ai_predictions.get(i, {})
                        original_label = img_info.get('original_label', '')
                        is_current = i == st.session_state.current_index
                        
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
                            key=f"img_btn_{i}_{total_count}",
                            help=f"#{i+1}: {img_info['name']}\n原始: {original_label or '無'}\nAI: {ai_pred.get('text', '未識別')}",
                            type=button_type,
                            use_container_width=True
                        ):
                            st.session_state.current_index = i
                            st.session_state.temp_label = get_default_label_for_current_image()
                            safe_rerun()
        
        # 中央：圖片預覽面板
        with col2:
            with st.container():
                st.markdown("### 🖼️ 驗證碼預覽")
                
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
                st.markdown("### ⚙️ 控制面板")
                
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
                        
                        if st.button("🎯 使用AI結果", key=f"use_ai_{current_idx}", use_container_width=True):
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
                        key=f"label_input_{current_idx}_v{update_trigger}",
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
                    if st.button("💾 保存修改", disabled=not is_valid, type="primary", key=f"save_{current_idx}", use_container_width=True):
                        if save_current_file(new_label):
                            if current_idx < len(st.session_state.folder_images) - 1:
                                st.session_state.current_index = current_idx + 1
                                st.session_state.temp_label = get_default_label_for_current_image()
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
                            st.session_state.current_index = current_idx - 1
                            st.session_state.temp_label = get_default_label_for_current_image()
                            safe_rerun()
                    with nav_col2:
                        last_idx = len(st.session_state.folder_images) - 1
                        if st.button("下一張 ➡️", disabled=current_idx >= last_idx, key=f"next_{current_idx}", use_container_width=True):
                            st.session_state.current_index = current_idx + 1
                            st.session_state.temp_label = get_default_label_for_current_image()
                            safe_rerun()
                    
                    # 進度
                    progress = (current_idx + 1) / len(st.session_state.folder_images)
                    st.progress(progress, text=f"進度: {current_idx + 1}/{len(st.session_state.folder_images)}")
                    
                    # 統計
                    st.markdown("#### 📊 統計")
                    
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.metric("總檔案", len(st.session_state.folder_images))
                        st.metric("已修正", st.session_state.modified_count)
                    with stat_col2:
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
                            if st.button("🏠 首張", disabled=current_idx == 0, key=f"jump_first_{current_idx}", use_container_width=True):
                                st.session_state.current_index = 0
                                st.session_state.temp_label = get_default_label_for_current_image()
                                safe_rerun()
                        with jump_col2:
                            last_idx = len(st.session_state.folder_images) - 1
                            if st.button("🏁 末張", disabled=current_idx == last_idx, key=f"jump_last_{current_idx}", use_container_width=True):
                                st.session_state.current_index = last_idx
                                st.session_state.temp_label = get_default_label_for_current_image()
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