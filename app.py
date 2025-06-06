#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit + CRNN模型整合 - 專業驗證碼識別工具"""

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
    page_title="🎯 AI驗證碼識別工具",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

# 專業界面CSS樣式
st.markdown("""
<style>
    .main .block-container {
        padding: 0;
        margin: 0;
        max-width: 100%;
        background: #1e2347;
    }
    
    .stApp {
        background: #1e2347;
    }
    
    /* 隱藏 Streamlit 默認元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    .css-1d391kg {padding: 0;}
    
    /* 主要容器 */
    .main-interface {
        display: grid;
        grid-template-columns: 1fr 2fr 1fr;
        height: 100vh;
        gap: 0;
        background: #1e2347;
    }
    
    /* 左側圖片列表面板 */
    .image-list-panel {
        background: #2c3e50;
        border-radius: 0 0 0 15px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }
    
    .panel-header {
        background: linear-gradient(135deg, #3b4a6b, #2c3e50);
        color: white;
        padding: 15px 20px;
        font-weight: bold;
        font-size: 1.1rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .image-list {
        flex: 1;
        overflow-y: auto;
        padding: 10px;
        background: #34495e;
    }
    
    .image-item {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin: 2px 0;
        background: #34495e;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
        color: white;
        font-size: 0.85rem;
    }
    
    .image-item:hover {
        background: #4a90e2;
        transform: translateX(3px);
    }
    
    .image-item.active {
        background: #e74c3c;
        border-color: #c0392b;
        box-shadow: 0 2px 8px rgba(231, 76, 60, 0.4);
    }
    
    .image-item .index {
        color: #bdc3c7;
        font-weight: bold;
        margin-right: 8px;
        min-width: 25px;
        font-size: 0.8rem;
    }
    
    .image-item .filename {
        flex: 1;
        font-family: 'Consolas', monospace;
        font-size: 0.75rem;
        color: white;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .image-item .original-label {
        background: #f39c12;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
        margin: 0 3px;
    }
    
    .image-item .ai-label {
        background: #9b59b6;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    
    /* 中央圖片預覽面板 */
    .image-preview-panel {
        background: #34495e;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: none;
    }
    
    .preview-container {
        background: #2c3e50;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        text-align: center;
    }
    
    .captcha-display {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .captcha-display img {
        max-width: 100%;
        max-height: 150px;
        image-rendering: pixelated;
    }
    
    /* 右側控制面板 */
    .control-panel {
        background: #2c3e50;
        border-radius: 0 0 15px 0;
        display: flex;
        flex-direction: column;
    }
    
    .control-section {
        padding: 15px 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .control-section:last-child {
        border-bottom: none;
        flex: 1;
    }
    
    .section-title {
        color: #ecf0f1;
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .info-display {
        background: #34495e;
        padding: 10px;
        border-radius: 6px;
        margin: 8px 0;
        font-family: 'Consolas', monospace;
        font-size: 0.9rem;
        color: #bdc3c7;
    }
    
    .original-label-display {
        background: #e74c3c;
        color: white;
        text-align: center;
        padding: 12px;
        border-radius: 8px;
        font-size: 1.5rem;
        font-weight: bold;
        letter-spacing: 3px;
        margin: 10px 0;
    }
    
    .ai-result-display {
        background: linear-gradient(135deg, #9b59b6, #8e44ad);
        color: white;
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        letter-spacing: 2px;
        margin: 10px 0;
        box-shadow: 0 4px 12px rgba(155, 89, 182, 0.3);
    }
    
    .confidence-bar {
        height: 8px;
        background: #34495e;
        border-radius: 4px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #e74c3c, #f39c12, #27ae60);
        transition: width 0.3s ease;
        border-radius: 4px;
    }
    
    .confidence-text {
        text-align: center;
        color: #bdc3c7;
        font-size: 0.85rem;
        margin-top: 5px;
    }
    
    .use-ai-btn {
        background: linear-gradient(135deg, #9b59b6, #8e44ad);
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 6px;
        cursor: pointer;
        width: 100%;
        font-weight: bold;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    
    .use-ai-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(155, 89, 182, 0.4);
    }
    
    .label-input {
        width: 100%;
        padding: 15px;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        border: 3px solid #34495e;
        border-radius: 8px;
        background: #27ae60;
        color: white;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin: 10px 0;
        font-family: 'Consolas', monospace;
    }
    
    .label-input:focus {
        outline: none;
        border-color: #27ae60;
        background: #2ecc71;
        box-shadow: 0 0 10px rgba(39, 174, 96, 0.4);
    }
    
    .save-btn {
        background: linear-gradient(135deg, #27ae60, #229954);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 8px;
        cursor: pointer;
        width: 100%;
        font-weight: bold;
        font-size: 1rem;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .save-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(39, 174, 96, 0.4);
    }
    
    .save-btn:disabled {
        background: #7f8c8d;
        cursor: not-allowed;
        transform: none;
        opacity: 0.6;
    }
    
    .nav-section {
        padding: 15px 20px;
    }
    
    .nav-buttons {
        display: flex;
        gap: 8px;
        margin: 10px 0;
    }
    
    .nav-btn {
        flex: 1;
        padding: 12px 8px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-weight: bold;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        color: white;
    }
    
    .nav-btn.prev {
        background: linear-gradient(135deg, #3498db, #2980b9);
    }
    
    .nav-btn.next {
        background: linear-gradient(135deg, #f39c12, #e67e22);
    }
    
    .nav-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .nav-btn:disabled {
        background: #7f8c8d;
        cursor: not-allowed;
        transform: none;
        opacity: 0.6;
    }
    
    .progress-display {
        text-align: center;
        color: #e74c3c;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    /* 滾動條樣式 */
    .image-list::-webkit-scrollbar {
        width: 8px;
    }
    
    .image-list::-webkit-scrollbar-track {
        background: #2c3e50;
    }
    
    .image-list::-webkit-scrollbar-thumb {
        background: #4a90e2;
        border-radius: 4px;
    }
    
    .image-list::-webkit-scrollbar-thumb:hover {
        background: #357abd;
    }
    
    /* 響應式設計 */
    @media (max-width: 1024px) {
        .main-interface {
            grid-template-columns: 1fr;
            grid-template-rows: auto auto auto;
        }
        
        .image-list {
            max-height: 200px;
        }
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
        'folder_path': "massive_real_captchas",
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
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(st.session_state.folder_images)
    batch_predictions = {}
    
    for i, img_info in enumerate(st.session_state.folder_images):
        status_text.text(f"🤖 AI識別中 ({i+1}/{total_files}): {img_info['name']}")
        
        try:
            image = Image.open(img_info['path'])
            predicted_text, confidence = predictor.predict(image)
            
            batch_predictions[i] = {
                'text': predicted_text,
                'confidence': confidence
            }
            
        except Exception as e:
            batch_predictions[i] = {'text': "ERROR", 'confidence': 0}
        
        progress_bar.progress((i + 1) / total_files)
    
    st.session_state.ai_predictions = batch_predictions
    status_text.success("🎯 AI批量識別完成！")
    progress_bar.empty()

def get_default_label_for_current_image():
    if not st.session_state.folder_images:
        return ""
    
    current_idx = st.session_state.current_index
    current_img = st.session_state.folder_images[current_idx]
    
    if current_idx in st.session_state.ai_predictions:
        ai_pred = st.session_state.ai_predictions[current_idx]
        if (ai_pred['confidence'] > 0.7 and 
            SimpleCaptchaCorrector.validate_label(ai_pred['text'])):
            return ai_pred['text']
    
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
        
        # 生成新檔名，處理重複情況
        new_filename = generate_unique_filename(old_path.parent, new_label)
        new_path = old_path.parent / new_filename
        
        # 如果路徑完全相同，表示沒有變更
        if old_path.resolve() == new_path.resolve():
            st.info(f"ℹ️ 檔名未變更: {new_filename}")
            return True
        
        # 執行重命名
        old_path.replace(new_path)
        
        # 更新統計
        original_label = current_file['original_label']
        if (st.session_state.ai_predictions.get(current_idx) and 
            st.session_state.ai_predictions[current_idx]['text'] == new_label and 
            original_label != new_label):
            st.session_state.ai_accurate_count += 1
        
        # 更新檔案記錄
        st.session_state.folder_images[current_idx] = {
            'name': new_filename,
            'path': str(new_path),
            'original_label': new_label
        }
        
        if current_idx not in st.session_state.modified_files:
            st.session_state.modified_count += 1
            st.session_state.modified_files.add(current_idx)
        
        # 顯示成功消息
        if "_" in new_filename:
            st.success(f"✅ 檔案已改名為: {new_filename} (自動避免重複)")
        else:
            st.success(f"✅ 檔案已改名為: {new_filename}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ 保存失敗: {e}")
        return False

def generate_unique_filename(directory: Path, label: str) -> str:
    """
    生成唯一的檔名，如果檔案已存在則加上 _001, _002 等後綴
    
    Args:
        directory: 目標目錄
        label: 4位大寫字母標籤
    
    Returns:
        唯一的檔名
    """
    base_filename = f"{label}.png"
    target_path = directory / base_filename
    
    # 如果檔案不存在，直接使用原始檔名
    if not target_path.exists():
        return base_filename
    
    # 檔案已存在，尋找可用的後綴
    counter = 1
    while counter <= 999:  # 最多支援到 _999
        suffix_filename = f"{label}_{counter:03d}.png"
        suffix_path = directory / suffix_filename
        
        if not suffix_path.exists():
            return suffix_filename
        
        counter += 1
    
    # 如果連 _999 都存在，則使用時間戳
    import time
    timestamp = int(time.time() * 1000) % 100000
    return f"{label}_{timestamp}.png"

def navigate_to_image(new_index: int):
    if not st.session_state.folder_images:
        return
    
    if 0 <= new_index < len(st.session_state.folder_images):
        st.session_state.current_index = new_index
        st.session_state.temp_label = get_default_label_for_current_image()

def main():
    if 'initialized' not in st.session_state:
        init_session_state()
    
    # 載入模型
    predictor = load_crnn_model()
    
    # 頂部標題區域
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50, #34495e); padding: 20px; text-align: center; margin-bottom: 20px; border-radius: 10px;">
        <h1 style="color: #e74c3c; font-size: 2rem; margin: 0; font-weight: bold;">
            🎯 AI驗證碼識別工具 - CRNN自動識別版
        </h1>
        <p style="color: #ecf0f1; margin: 5px 0 0 0; font-size: 1rem;">
            使用最新訓練的CRNN模型，專門識別4位大寫英文字母驗證碼
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI模型狀態顯示
    if predictor is not None:
        accuracy = predictor.model_info.get('best_val_captcha_acc', 0) * 100
        st.markdown(f"""
        <div style="background: #27ae60; color: white; padding: 10px 20px; border-radius: 8px; text-align: center; margin-bottom: 15px; font-weight: bold;">
            🤖 CRNN模型已就緒！準確率: {accuracy:.2f}%
        </div>
        """, unsafe_allow_html=True)
        
        # 模型詳細信息
        epoch = predictor.model_info.get('epoch', 'unknown')
        st.markdown(f"""
        <div style="background: rgba(155, 89, 182, 0.2); border: 1px solid #9b59b6; color: #bb8fce; padding: 8px 15px; border-radius: 6px; margin-bottom: 15px; font-size: 0.9rem;">
            📋 模型訓練輪數: {epoch} | 驗證準確率: {accuracy:.2f}% | 支援字符: {CHARACTERS} | 序列長度: {CAPTCHA_LENGTH_EXPECTED}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #e74c3c; color: white; padding: 10px 20px; border-radius: 8px; text-align: center; margin-bottom: 15px; font-weight: bold;">
            ❌ CRNN模型載入失敗，請檢查模型檔案路徑
        </div>
        """, unsafe_allow_html=True)
    
    # 資料夾選擇區域
    st.markdown("""
    <div style="background: #34495e; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #ecf0f1; margin-bottom: 15px;">📁 資料夾路徑設定:</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 預設路徑按鈕
    st.markdown("**快速選擇路徑:**")
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    
    with preset_col1:
        if st.button("🖥️ 桌面", use_container_width=True):
            st.session_state.folder_path = r"C:\Users\User\Desktop"
    
    with preset_col2:
        if st.button("📥 下載", use_container_width=True):
            st.session_state.folder_path = r"C:\Users\User\Downloads"
    
    with preset_col3:
        if st.button("🎯 預設偵錯", use_container_width=True):
            st.session_state.folder_path = r"C:\Users\User\Desktop\Python3.8\02_emnist\debug_captchas_adaptive_captcha_paper"
    
    with preset_col4:
        if st.button("🧪 測試數據", use_container_width=True):
            st.session_state.folder_path = r"C:\Users\User\Desktop\Python3.8\02_emnist\debug_captchas_augmented_all_split\test"
    
    # 路徑輸入和載入按鈕
    path_col1, path_col2, path_col3 = st.columns([3, 1, 1])
    
    with path_col1:
        folder_path = st.text_input(
            "資料夾路徑", 
            value=st.session_state.folder_path, 
            key="folder_input",
            placeholder="請輸入PNG圖片資料夾的絕對路徑",
            help="支援拖拽資料夾到此處"
        )
        st.session_state.folder_path = folder_path
    
    with path_col2:
        if st.button("🚀 載入圖片", type="primary", use_container_width=True):
            if folder_path.strip():
                if load_images_from_folder(folder_path.strip()):
                    st.rerun()
            else:
                st.error("❌ 請輸入資料夾路徑")
    
    with path_col3:
        if st.button("🤖 批量識別", 
                    disabled=not st.session_state.folder_images or not predictor, 
                    use_container_width=True,
                    type="secondary"):
            if st.session_state.folder_images and predictor:
                perform_batch_ai_prediction(predictor)
                st.rerun()
    
    # 路徑提示信息
    if st.session_state.folder_images:
        total_files = len(st.session_state.folder_images)
        st.markdown(f"""
        <div style="background: rgba(52, 152, 219, 0.1); border: 1px solid #3498db; color: #85c1e9; padding: 8px 15px; border-radius: 6px; margin: 10px 0; font-size: 0.9rem;">
            💡 <strong>AI功能:</strong> 自動使用CRNN模型識別4位大寫英文字母 (A-Z)<br>
            💡 <strong>保存規則:</strong> 新檔名將是修正後的4位大寫英文字母 + ".png"<br>
            💡 <strong>已載入:</strong> {total_files} 張PNG圖片，若目標檔名已存在則會直接覆寫
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(52, 152, 219, 0.1); border: 1px solid #3498db; color: #85c1e9; padding: 8px 15px; border-radius: 6px; margin: 10px 0; font-size: 0.9rem;">
            💡 <strong>使用說明:</strong><br>
            1. 選擇包含PNG驗證碼圖片的資料夾<br>
            2. 點擊"載入圖片"掃描所有PNG檔案<br>
            3. 點擊"批量識別"使用AI自動識別所有圖片<br>
            4. 在下方界面中瀏覽和修正識別結果
        </div>
        """, unsafe_allow_html=True)
    
    # 主要界面
    if st.session_state.folder_images:
        render_main_interface(predictor)
    else:
        # 空狀態顯示
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #7f8c8d;">
            <h2 style="color: #95a5a6;">📂 請先載入圖片資料夾</h2>
            <p style="font-size: 1.1rem; margin-top: 20px;">選擇包含PNG驗證碼圖片的資料夾，然後點擊"載入圖片"開始處理</p>
        </div>
        """, unsafe_allow_html=True)

def render_main_interface(predictor):
    # 創建三列佈局
    col_list, col_preview, col_control = st.columns([1, 2, 1])
    
    # 左側：圖片列表面板
    with col_list:
        st.markdown("""
        <div class="panel-header">
            📋 圖片列表 (AI識別結果)
        </div>
        """, unsafe_allow_html=True)
        
        # 創建圖片列表容器
        with st.container():
            list_container = st.container()
            with list_container:
                for i, img_info in enumerate(st.session_state.folder_images[:50]):  # 限制顯示前50個避免太慢
                    # 檢查是否為當前圖片
                    is_active = i == st.session_state.current_index
                    is_modified = i in st.session_state.modified_files
                    
                    # 獲取AI預測結果
                    ai_pred = st.session_state.ai_predictions.get(i, {})
                    original_label = img_info.get('original_label', '')
                    
                    # 構建顯示文字
                    display_parts = []
                    display_parts.append(f"{i+1}.")
                    display_parts.append(img_info['name'][:15] + "..." if len(img_info['name']) > 15 else img_info['name'])
                    
                    if original_label:
                        display_parts.append(f"[{original_label}]")
                    
                    if ai_pred.get('text'):
                        display_parts.append(f"AI:{ai_pred['text']}")
                    
                    display_text = " ".join(display_parts)
                    
                    # 按鈕樣式
                    button_type = "primary" if is_active else "secondary"
                    
                    if st.button(
                        display_text,
                        key=f"img_btn_{i}",
                        help=f"點擊查看: {img_info['name']}\n原始標籤: {original_label}\nAI識別: {ai_pred.get('text', 'N/A')}",
                        type=button_type,
                        use_container_width=True
                    ):
                        navigate_to_image(i)
                        st.rerun()
                    
                    # 狀態指示
                    if is_active:
                        st.markdown("👆 **當前選中**")
                    elif is_modified:
                        st.markdown("✅ **已修正**")
    
    # 中央：圖片預覽面板
    with col_preview:
        st.markdown("""
        <div class="panel-header">
            🖼️ 驗證碼圖片預覽
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.current_index < len(st.session_state.folder_images):
            current_img = st.session_state.folder_images[st.session_state.current_index]
            
            # 創建圖片預覽容器
            st.markdown("""
            <div class="preview-container">
                <div class="captcha-display">
            """, unsafe_allow_html=True)
            
            try:
                image = Image.open(current_img['path'])
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"❌ 無法載入圖片: {e}")
            
            st.markdown("""
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 圖片信息
            st.markdown(f"**檔案名稱:** `{current_img['name']}`")
            
        else:
            st.markdown("""
            <div class="preview-container">
                <p style="font-size: 1.5rem; color: #95a5a6;">請選擇要查看的圖片 🎨</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 右側：控制面板
    with col_control:
        st.markdown("""
        <div class="panel-header">
            ⚙️ 控制面板
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.current_index < len(st.session_state.folder_images):
            current_img = st.session_state.folder_images[st.session_state.current_index]
            current_idx = st.session_state.current_index
            
            # 檔案信息區域
            st.markdown("##### 📄 檔案信息")
            st.markdown(f"**檔名:** `{current_img['name']}`")
            
            original_label = current_img.get('original_label', '')
            if original_label:
                st.markdown(f"""
                <div class="original-label-display">
                    {original_label}
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**原始標籤** (從檔名提取)")
            else:
                st.markdown("**原始標籤:** 無法提取")
            
            # AI識別結果區域
            st.markdown("##### 🤖 AI識別結果")
            
            if current_idx in st.session_state.ai_predictions:
                ai_pred = st.session_state.ai_predictions[current_idx]
                
                st.markdown(f"""
                <div class="ai-result-display">
                    {ai_pred['text']}
                </div>
                """, unsafe_allow_html=True)
                
                # 置信度顯示
                confidence = ai_pred['confidence']
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence * 100}%"></div>
                </div>
                <div class="confidence-text">置信度: {confidence:.1%}</div>
                """, unsafe_allow_html=True)
                
                # 使用AI結果按鈕
                if st.button("🎯 使用AI識別結果", 
                           key=f"use_ai_{current_idx}",
                           use_container_width=True):
                    if SimpleCaptchaCorrector.validate_label(ai_pred['text']):
                        st.session_state.temp_label = ai_pred['text']
                        st.rerun()
                    else:
                        st.warning("⚠️ AI識別結果格式無效")
            else:
                st.markdown("""
                <div class="ai-result-display">
                    等待AI識別...
                </div>
                """, unsafe_allow_html=True)
                st.info("💡 請先執行 AI 批量識別")
            
            # 標籤修正區域
            st.markdown("##### ✏️ 標籤修正 (4位大寫字母)")
            
            # 獲取預設標籤
            if not st.session_state.temp_label:
                st.session_state.temp_label = get_default_label_for_current_image()
            
            # 標籤輸入框
            new_label = st.text_input(
                "輸入修正後的標籤",
                value=st.session_state.temp_label,
                max_chars=4,
                key=f"label_input_{current_idx}",
                help="只能輸入A-Z的大寫字母",
                placeholder="例如: ABCD"
            ).upper()
            
            st.session_state.temp_label = new_label
            is_valid = SimpleCaptchaCorrector.validate_label(new_label)
            
            # 驗證提示
            if new_label:
                if is_valid:
                    st.success("✅ 格式正確")
                else:
                    st.error("❌ 請輸入4個大寫英文字母")
            
            # 保存按鈕
            if st.button("💾 保存修改", 
                        disabled=not is_valid, 
                        use_container_width=True, 
                        type="primary",
                        key=f"save_btn_{current_idx}"):
                if save_current_file(new_label):
                    if current_idx < len(st.session_state.folder_images) - 1:
                        navigate_to_image(current_idx + 1)
                        st.balloons()
                        st.rerun()
                    else:
                        st.success("🎉 全部處理完成！")
                        st.balloons()
            
            # 導航區域
            st.markdown("##### 🧭 導航")
            
            nav_col1, nav_col2 = st.columns(2)
            
            with nav_col1:
                if st.button("⬅️ 上一張", 
                           disabled=current_idx == 0, 
                           use_container_width=True, 
                           key=f"prev_btn_{current_idx}"):
                    navigate_to_image(current_idx - 1)
                    st.rerun()
            
            with nav_col2:
                if st.button("下一張 ➡️", 
                           disabled=current_idx >= len(st.session_state.folder_images) - 1, 
                           use_container_width=True, 
                           key=f"next_btn_{current_idx}"):
                    navigate_to_image(current_idx + 1)
                    st.rerun()
            
            # 進度顯示
            st.markdown(f"""
            <div class="progress-display">
                {current_idx + 1} / {len(st.session_state.folder_images)}
            </div>
            """, unsafe_allow_html=True)
            
            progress_pct = (current_idx + 1) / len(st.session_state.folder_images)
            st.progress(progress_pct, text=f"進度: {progress_pct:.1%}")
            
            # 統計信息
            st.markdown("##### 📊 處理統計")
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("總檔案", len(st.session_state.folder_images))
                st.metric("已修正", st.session_state.modified_count)
            
            with col_stat2:
                ai_acc = (st.session_state.ai_accurate_count / max(st.session_state.modified_count, 1)) * 100
                st.metric("AI準確率", f"{ai_acc:.0f}%")
                
                overall_progress = (st.session_state.modified_count / len(st.session_state.folder_images)) * 100
                st.metric("完成進度", f"{overall_progress:.0f}%")
            
            # 快速跳轉
            if len(st.session_state.folder_images) > 10:
                st.markdown("##### ⚡ 快速跳轉")
                
                jump_col1, jump_col2 = st.columns(2)
                
                with jump_col1:
                    if st.button("🏠 回到開頭", 
                               key=f"jump_start_{current_idx}",
                               disabled=current_idx == 0,
                               use_container_width=True):
                        navigate_to_image(0)
                        st.rerun()
                
                with jump_col2:
                    last_idx = len(st.session_state.folder_images) - 1
                    if st.button("🏁 跳到最後", 
                               key=f"jump_end_{current_idx}",
                               disabled=current_idx == last_idx,
                               use_container_width=True):
                        navigate_to_image(last_idx)
                        st.rerun()
                
                # 中間位置跳轉
                mid_idx = len(st.session_state.folder_images) // 2
                if st.button("📍 跳到中間", 
                           key=f"jump_mid_{current_idx}",
                           disabled=current_idx == mid_idx,
                           use_container_width=True):
                    navigate_to_image(mid_idx)
                    st.rerun()

if __name__ == "__main__":
    main()