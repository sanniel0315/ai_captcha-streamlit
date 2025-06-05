#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit + CRNN模型整合 - 自動驗證碼識別工具 (參照Flask版本功能)"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import re
import string
import time
from pathlib import Path
import warnings
from typing import List, Tuple, Dict, Optional
import sys
import subprocess

# 抑制警告
warnings.filterwarnings('ignore')

# 檢查是否在正確的Streamlit環境中運行
def check_streamlit_context():
    """檢查是否在正確的Streamlit環境中運行"""
    try:
        # 檢查是否存在streamlit的運行上下文
        import streamlit.runtime.scriptrunner.script_run_context as script_run_context
        ctx = script_run_context.get_script_run_ctx()
        return ctx is not None
    except:
        return False

# 如果不在Streamlit上下文中運行，提供友好的錯誤信息並嘗試自動啟動
if not check_streamlit_context():
    print("\n" + "="*60)
    print("🚨 請使用正確的方式運行此Streamlit應用！")
    print("="*60)
    print("\n正確的運行方式：")
    print("1. 開啟命令提示符或PowerShell")
    print(f"2. 切換到應用目錄：")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"   cd {current_dir}")
    print("3. 運行以下命令：")
    print("   streamlit run app.py")
    print(f"\n或者直接運行：")
    print(f"   streamlit run {os.path.abspath(__file__)}")
    print("\n" + "="*60)
    print("💡 提示：不要直接用python執行此文件！")
    print("="*60)
    
    # 嘗試自動啟動streamlit
    try:
        current_file = os.path.abspath(__file__)
        print("🚀 正在嘗試自動啟動Streamlit...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", current_file])
    except Exception as e:
        print(f"❌ 自動啟動失敗：{e}")
        print("請手動使用上述命令運行。")
    
    sys.exit(1)

# 頁面配置
st.set_page_config(
    page_title="🎯 AI驗證碼識別工具",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 模型配置 - 參照Flask版本，根據項目結構調整
MODEL_PATHS = [
    "best_crnn_captcha_model.pth",  # 主目錄中的模型檔案
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

# 高級自定義CSS樣式
st.markdown("""
<style>
    /* 主體背景 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
    }
    
    /* 標題樣式 */
    .main-title {
        background: linear-gradient(135deg, #0f3460, #16213e);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        color: white;
    }
    
    /* AI狀態卡片 */
    .ai-status-card {
        background: linear-gradient(135deg, #27ae60, #229954);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .ai-status-error {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
    }
    
    /* AI結果顯示 */
    .ai-result {
        background: linear-gradient(135deg, #8e44ad, #9b59b6);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
    }
    
    /* 成功結果 */
    .success-result {
        background: linear-gradient(135deg, #27ae60, #229954);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* 統計卡片 */
    .metric-card {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    /* 隱藏Streamlit默認元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    header {visibility: hidden;}
    
    /* 圖片容器樣式 */
    .image-container {
        text-align: center;
        padding: 1rem;
        background: #2c3e50;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* 列表項目樣式 */
    .image-item {
        background: #34495e;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        border-left: 4px solid #3498db;
    }
    
    .image-item.modified {
        border-left-color: #27ae60;
        background: #1e3a2e;
    }
    
    .image-item.current {
        border-left-color: #e74c3c;
        background: #3a1e1e;
    }
</style>
""", unsafe_allow_html=True)

# 工具類 - 參照Flask版本
class SimpleCaptchaCorrector:
    @staticmethod
    def extract_label_from_filename(filename: str) -> str:
        """從PNG檔名擷取第一組4個大寫英文字母"""
        name_without_ext, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            return ""
        match = re.search(r'([A-Z]{4})', name_without_ext)
        return match.group(1).upper() if match else ""

    @staticmethod
    def validate_label(label: str) -> bool:
        """驗證是否為4位大寫英文字母"""
        return bool(re.fullmatch(r'[A-Z]{4}', label))

    @staticmethod
    def generate_new_filename(new_label: str) -> str:
        """依新標籤產生檔名"""
        return f"{new_label}.png"

# CRNN模型 - 參照Flask版本
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

# 預測器類 - 參照Flask版本
class CRNNPredictor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.config = None
        self.is_loaded = False
        self.model_info = {}

    def load_model(self, model_path: str):
        """載入CRNN模型"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                return False

            print(f"🔄 正在載入模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                self.config = DEFAULT_CONFIG.copy()

            for key, val in DEFAULT_CONFIG.items():
                self.config.setdefault(key, val)

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
                print("❌ 找不到 model_state_dict 或 state_dict")
                return False

            self.model.load_state_dict(checkpoint[sd_key])
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Grayscale(self.config['INPUT_CHANNELS']),
                transforms.Resize((self.config['IMAGE_HEIGHT'], self.config['IMAGE_WIDTH'])),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * self.config['INPUT_CHANNELS'], [0.5] * self.config['INPUT_CHANNELS'])
            ])

            self.is_loaded = True
            self.model_info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_val_captcha_acc': checkpoint.get('best_val_captcha_acc', 0),
                'idx_to_char': checkpoint.get('idx_to_char', IDX_TO_CHAR)
            }

            print(f"✅ CRNN模型載入成功 (epoch={self.model_info['epoch']}, acc={self.model_info['best_val_captcha_acc']:.4f})")
            return True

        except Exception as e:
            print(f"❌ 模型載入失敗: {e}")
            return False

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """對單張圖片做預測"""
        if not self.is_loaded:
            return "", 0.0

        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)

            _, width_cnn_output, _ = outputs.shape
            seq_len = self.config['SEQUENCE_LENGTH']

            if width_cnn_output >= seq_len:
                start = (width_cnn_output - seq_len) // 2
                focused = outputs[:, start:start + seq_len, :]
            else:
                pad = seq_len - width_cnn_output
                focused = torch.cat([outputs, outputs[:, -1:, :].repeat(1, pad, 1)], dim=1)

            pred_indices = torch.argmax(focused, dim=2)[0]
            idx_to_char_map = self.model_info.get('idx_to_char', IDX_TO_CHAR)
            
            if isinstance(next(iter(idx_to_char_map.keys())), str):
                idx_to_char_map = {int(k): v for k, v in idx_to_char_map.items()}

            text = ''.join(idx_to_char_map.get(idx.item(), '?') for idx in pred_indices).upper()

            probs = torch.softmax(focused, dim=2)
            max_probs = torch.max(probs, dim=2)[0]
            confidence = float(torch.mean(max_probs).item())

            return text, confidence

        except Exception as e:
            print(f"❌ 預測失敗: {e}")
            return "", 0.0

def check_model_file(model_path):
    """檢查模型檔案的詳細信息"""
    try:
        if not os.path.exists(model_path):
            return {"status": "error", "message": f"檔案不存在: {model_path}"}
        
        file_size = os.path.getsize(model_path) / (1024*1024)
        
        # 嘗試載入並檢查checkpoint內容
        checkpoint = torch.load(model_path, map_location='cpu')
        
        info = {
            "status": "success",
            "file_size_mb": file_size,
            "checkpoint_keys": list(checkpoint.keys()),
            "has_config": "config" in checkpoint,
            "has_model_state": "model_state_dict" in checkpoint or "state_dict" in checkpoint,
            "epoch": checkpoint.get('epoch', 'unknown'),
            "accuracy": checkpoint.get('best_val_captcha_acc', 0)
        }
        
        if 'config' in checkpoint:
            info['config'] = checkpoint['config']
        
        return info
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    """檢查項目中的重要檔案"""
    current_dir = Path(".")
    
    # 檢查模型檔案
    model_files = []
    for model_path in MODEL_PATHS:
        if Path(model_path).exists():
            model_files.append(model_path)
    
    # 檢查圖片資料夾
    image_folders = []
    for item in current_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            png_count = len(list(item.glob('*.png')))
            if png_count > 0:
                image_folders.append(f"{item.name} ({png_count} PNG檔案)")
    
    return model_files, image_folders
def init_session_state():
    """初始化session state變量"""
    defaults = {
        'folder_images': [],
        'current_index': 0,
        'ai_predictions': {},
        'modified_labels': {},
        'modified_count': 0,
        'modified_files': set(),
        'ai_accurate_count': 0,
        'folder_path': "massive_real_captchas"  # 根據您的項目結構調整預設路徑
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# 載入模型（使用緩存）
@st.cache_resource
def load_crnn_model():
    """載入並緩存CRNN模型"""
    predictor = CRNNPredictor()
    
    model_files = ['best_crnn_captcha_model.pth', 'model.pth', 'crnn_model.pth']
    model_path = None
    
    for file in model_files:
        if os.path.exists(file):
            model_path = file
            break
    
    if model_path is None:
        return None
    
    if predictor.load_model(model_path):
        return predictor
    else:
        return None

def load_images_from_folder(folder_path: str):
    """從資料夾載入圖片"""
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
        st.session_state.modified_labels = {}
        st.session_state.modified_count = 0
        st.session_state.modified_files = set()
        st.session_state.ai_accurate_count = 0
        
        st.success(f"✅ 成功載入 {len(image_files_list)} 張PNG圖片")
        return True
        
    except Exception as e:
        st.error(f"❌ 載入圖片時異常: {e}")
        return False

def perform_batch_ai_prediction(predictor):
    """執行批量AI預測"""
    if not st.session_state.folder_images or not predictor:
        return
    
    # 顯示進度
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(st.session_state.folder_images)
    
    for i, img_info in enumerate(st.session_state.folder_images):
        status_text.text(f"🤖 AI識別中 ({i+1}/{total_files}): {img_info['name']}")
        
        try:
            image = Image.open(img_info['path'])
            predicted_text, confidence = predictor.predict(image)
            
            st.session_state.ai_predictions[i] = {
                'text': predicted_text,
                'confidence': confidence
            }
            
        except Exception as e:
            st.error(f"❌ AI預測失敗 {img_info['name']}: {e}")
            st.session_state.ai_predictions[i] = {'text': "ERROR", 'confidence': 0}
        
        progress_bar.progress((i + 1) / total_files)
    
    status_text.success("🎯 AI批量識別完成！")
    progress_bar.empty()

def save_current_file(new_label: str):
    """保存當前文件的修改"""
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
        
        # 檢查是否需要改名
        if old_path.resolve() == new_path.resolve():
            st.info(f"ℹ️ 檔名未變更: {new_filename}")
            return True
        
        # 如果目標檔案存在，會被覆蓋
        if new_path.exists():
            st.warning(f"⚠️ 目標檔案 {new_filename} 已存在，將被覆蓋")
        
        # 執行改名
        old_path.replace(new_path)
        
        # 更新記錄
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

def main():
    """主應用程序"""
    try:
        # 初始化
        init_session_state()
        
        # 主標題
        st.markdown("""
        <div class="main-title">
            <h1>🎯 AI驗證碼識別工具 - CRNN自動識別版</h1>
            <p>使用CRNN模型自動識別4位大寫英文字母驗證碼</p>
            <p style="font-size: 0.9rem; opacity: 0.8;">當前項目: ai_captcha-streamlit</p>
        </div>
        """, unsafe_allow_html=True)

        # 載入模型
        with st.spinner("🔄 正在載入CRNN模型..."):
            predictor = load_crnn_model()
        
        # 檢查項目檔案
        model_files, image_folders = check_project_files()
        
        # 側邊欄
        with st.sidebar:
            st.markdown("### ⚙️ 控制面板")
            
            # 項目檔案狀態
            st.markdown("### 📋 項目檔案狀態")
            
            # 模型檔案狀態
            if model_files:
                st.success(f"✅ 找到 {len(model_files)} 個模型檔案")
                for model_file in model_files:
                    file_size = os.path.getsize(model_file) / (1024*1024)
                    st.text(f"📦 {model_file} ({file_size:.2f} MB)")
            else:
                st.error("❌ 未找到模型檔案")
            
            # 圖片資料夾狀態
            if image_folders:
                st.success(f"✅ 找到 {len(image_folders)} 個圖片資料夾")
                for folder in image_folders:
                    st.text(f"📁 {folder}")
            else:
                st.warning("⚠️ 未找到包含PNG檔案的資料夾")
            
            # 模型狀態
            if predictor is not None:
                st.markdown("""
                <div class="ai-status-card">
                    🤖 CRNN模型已就緒<br>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📊 模型詳情")
                if predictor.model_info:
                    epoch = predictor.model_info.get('epoch', 'unknown')
                    accuracy = predictor.model_info.get('best_val_captcha_acc', 0)
                    st.info(f"📈 訓練輪數: {epoch}")
                    st.info(f"📊 驗證準確率: {accuracy:.4f}")
                    st.info(f"🔤 支援字符: {CHARACTERS}")
                    st.info(f"📏 序列長度: {CAPTCHA_LENGTH_EXPECTED}")
            else:
                st.markdown("""
                <div class="ai-status-card ai-status-error">
                    ❌ 模型載入失敗<br>
                    請檢查模型文件
                </div>
                """, unsafe_allow_html=True)
                st.error("找不到模型文件。請確保以下任一文件存在：")
                for path in MODEL_PATHS:
                    st.error(f"• {path}")
                st.stop()
            
            st.markdown("### 🎯 功能選擇")
            page_mode = st.radio(
                "選擇操作模式",
                ["📁 資料夾批量處理", "📷 單張識別", "📊 統計分析"],
                index=0
            )

        # 主要內容區域
        if page_mode == "📁 資料夾批量處理":
            folder_batch_processing(predictor)
        elif page_mode == "📷 單張識別":
            single_image_recognition(predictor)
        else:
            statistics_analysis(predictor)
            
    except Exception as e:
        st.error(f"❌ 應用程序發生錯誤: {e}")
        st.error("請重新載入頁面或聯繫支援")

def folder_batch_processing(predictor):
    """資料夾批量處理功能"""
    st.markdown("## 📁 資料夾批量處理")
    
    # 路徑設定區域 - 基於實際存在的資料夾
    st.markdown("### 📂 資料夾路徑設定")
    
    # 顯示可用的圖片資料夾
    if image_folders:
        st.markdown("#### 🎯 專案中可用的圖片資料夾:")
        cols = st.columns(min(len(image_folders), 4))
        for i, folder_info in enumerate(image_folders):
            folder_name = folder_info.split(' (')[0]  # 取得資料夾名稱
            with cols[i % 4]:
                if st.button(f"📁 {folder_name}", help=f"選擇: {folder_info}", key=f"proj_folder_{i}"):
                    st.session_state.folder_path = folder_name
    
    # 其他常用路徑
    st.markdown("#### 🔗 其他常用路徑:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📁 massive_real_captchas", help="設定為項目中的massive_real_captchas資料夾"):
            st.session_state.folder_path = "massive_real_captchas"
    
    with col2:
        if st.button("🖥️ 桌面", help="設定為桌面路徑"):
            st.session_state.folder_path = r"C:\Users\User\Desktop"
    
    with col3:
        if st.button("📥 下載", help="設定為下載資料夾"):
            st.session_state.folder_path = r"C:\Users\User\Downloads"
    
    with col4:
        if st.button("🧪 測試數據", help="設定為測試數據路徑"):
            st.session_state.folder_path = r"C:\Users\User\Desktop\Python3.8\02_emnist\debug_captchas_augmented_all_split\test"
    
    # 路徑輸入
    folder_path = st.text_input(
        "📁 資料夾路徑",
        value=st.session_state.folder_path,
        help="請輸入包含PNG圖片的資料夾絕對路徑"
    )
    st.session_state.folder_path = folder_path
    
    # 載入按鈕
    col_load, col_predict = st.columns(2)
    
    with col_load:
        if st.button("🚀 載入圖片", type="primary"):
            if folder_path.strip():
                if load_images_from_folder(folder_path.strip()):
                    st.rerun()
            else:
                st.error("❌ 請輸入資料夾路徑")
    
    with col_predict:
        if st.button("🤖 開始AI批量識別", type="secondary", disabled=not st.session_state.folder_images):
            if st.session_state.folder_images:
                perform_batch_ai_prediction(predictor)
                st.rerun()

    # 提示信息
    st.info("💡 **AI功能**: 自動使用CRNN模型識別4位大寫英文字母 (A-Z)")
    st.info("💡 **保存規則**: 新檔名將是修正後的4位大寫英文字母 + \".png\"")
    st.warning("⚠️ **注意**: 若目標檔名已存在，則會直接覆寫")
    
    # 如果有載入的圖片，顯示處理界面
    if st.session_state.folder_images:
        display_image_processing_interface(predictor)

def display_image_processing_interface(predictor):
    """顯示圖片處理界面"""
    st.markdown("---")
    st.markdown("## 🖼️ 圖片處理界面")
    
    # 統計信息
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("📋 總檔案數", len(st.session_state.folder_images))
    
    with col_stats2:
        st.metric("✅ 已修正", st.session_state.modified_count)
    
    with col_stats3:
        ai_acc = (st.session_state.ai_accurate_count / max(st.session_state.modified_count, 1)) * 100
        st.metric("🤖 AI準確率", f"{ai_acc:.0f}%")
    
    with col_stats4:
        progress = (st.session_state.modified_count / len(st.session_state.folder_images)) * 100
        st.metric("📈 完成進度", f"{progress:.0f}%")
    
    # 主要處理區域
    col_list, col_main = st.columns([1, 2])
    
    with col_list:
        st.markdown("#### 📋 圖片列表")
        
        # 圖片列表
        list_container = st.container()
        with list_container:
            for i, img_info in enumerate(st.session_state.folder_images):
                # 構建顯示文本
                display_text = f"{i+1}. {img_info['name'][:20]}..."
                
                # 添加AI預測結果
                if i in st.session_state.ai_predictions:
                    ai_pred = st.session_state.ai_predictions[i]
                    display_text += f" | AI: {ai_pred['text']}"
                
                # 樣式類別
                style_class = "image-item"
                if i in st.session_state.modified_files:
                    style_class += " modified"
                if i == st.session_state.current_index:
                    style_class += " current"
                
                # 顯示項目
                if st.button(
                    display_text,
                    key=f"img_btn_{i}",
                    help=f"點擊查看: {img_info['name']}",
                    use_container_width=True
                ):
                    st.session_state.current_index = i
                    st.rerun()
                
                # 顯示樣式標記
                if i == st.session_state.current_index:
                    st.markdown("👆 **當前選中**")
                elif i in st.session_state.modified_files:
                    st.markdown("✅ 已修正")
    
    with col_main:
        st.markdown("#### 🖼️ 當前圖片處理")
        
        if st.session_state.current_index < len(st.session_state.folder_images):
            current_img = st.session_state.folder_images[st.session_state.current_index]
            
            # 圖片顯示
            col_img, col_control = st.columns([2, 1])
            
            with col_img:
                try:
                    image = Image.open(current_img['path'])
                    st.image(
                        image,
                        caption=f"檔案: {current_img['name']}",
                        use_column_width=True
                    )
                except Exception as e:
                    st.error(f"❌ 無法載入圖片: {e}")
            
            with col_control:
                st.markdown("##### 📄 檔案信息")
                st.text(f"檔名: {current_img['name']}")
                st.text(f"原始標籤: {current_img['original_label'] or '無法提取'}")
                
                # AI識別結果
                current_idx = st.session_state.current_index
                if current_idx in st.session_state.ai_predictions:
                    ai_pred = st.session_state.ai_predictions[current_idx]
                    
                    st.markdown("##### 🤖 AI識別結果")
                    st.markdown(f"""
                    <div class="ai-result">
                        {ai_pred['text']}<br>
                        置信度: {ai_pred['confidence']:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(ai_pred['confidence'])
                    
                    # 使用AI結果按鈕
                    if st.button("🎯 使用AI識別結果", use_container_width=True):
                        if SimpleCaptchaCorrector.validate_label(ai_pred['text']):
                            st.session_state.temp_label = ai_pred['text']
                            st.success(f"✅ 已填入AI結果: {ai_pred['text']}")
                        else:
                            st.warning("⚠️ AI預測結果格式無效")
                
                # 標籤修正
                st.markdown("##### ✏️ 標籤修正")
                
                # 預設值邏輯
                default_value = ""
                if current_idx in st.session_state.ai_predictions:
                    ai_pred = st.session_state.ai_predictions[current_idx]
                    if (ai_pred['confidence'] > 0.7 and 
                        SimpleCaptchaCorrector.validate_label(ai_pred['text'])):
                        default_value = ai_pred['text']
                
                if not default_value and current_img['original_label']:
                    default_value = current_img['original_label']
                
                # 使用臨時變量來處理輸入
                if 'temp_label' not in st.session_state:
                    st.session_state.temp_label = default_value
                
                new_label = st.text_input(
                    "輸入4個大寫字母",
                    value=st.session_state.temp_label,
                    max_chars=4,
                    key=f"label_input_{current_idx}",
                    help="只能輸入A-Z的大寫字母"
                ).upper()
                
                # 更新臨時變量
                st.session_state.temp_label = new_label
                
                # 驗證輸入
                is_valid = SimpleCaptchaCorrector.validate_label(new_label)
                
                if new_label:
                    if is_valid:
                        st.success(f"✅ 格式正確: {new_label}")
                    else:
                        st.error("❌ 請輸入4個大寫英文字母")
                
                # 保存按鈕
                if st.button(
                    "💾 保存修改",
                    disabled=not is_valid,
                    use_container_width=True,
                    type="primary"
                ):
                    if save_current_file(new_label):
                        # 保存成功後自動跳到下一張
                        if current_idx < len(st.session_state.folder_images) - 1:
                            st.session_state.current_index += 1
                            # 重置臨時標籤
                            next_img = st.session_state.folder_images[st.session_state.current_index]
                            next_default = ""
                            if st.session_state.current_index in st.session_state.ai_predictions:
                                next_ai = st.session_state.ai_predictions[st.session_state.current_index]
                                if (next_ai['confidence'] > 0.7 and 
                                    SimpleCaptchaCorrector.validate_label(next_ai['text'])):
                                    next_default = next_ai['text']
                            if not next_default and next_img['original_label']:
                                next_default = next_img['original_label']
                            st.session_state.temp_label = next_default
                        else:
                            st.balloons()
                            st.success("🎉 全部處理完成！")
                        st.rerun()
                
                # 導航按鈕
                st.markdown("##### 🧭 導航")
                nav_col1, nav_col2 = st.columns(2)
                
                with nav_col1:
                    if st.button(
                        "⬅️ 上一張",
                        disabled=current_idx == 0,
                        use_container_width=True
                    ):
                        st.session_state.current_index -= 1
                        # 更新臨時標籤
                        prev_img = st.session_state.folder_images[st.session_state.current_index]
                        prev_default = ""
                        if st.session_state.current_index in st.session_state.ai_predictions:
                            prev_ai = st.session_state.ai_predictions[st.session_state.current_index]
                            if (prev_ai['confidence'] > 0.7 and 
                                SimpleCaptchaCorrector.validate_label(prev_ai['text'])):
                                prev_default = prev_ai['text']
                        if not prev_default and prev_img['original_label']:
                            prev_default = prev_img['original_label']
                        st.session_state.temp_label = prev_default
                        st.rerun()
                
                with nav_col2:
                    if st.button(
                        "下一張 ➡️",
                        disabled=current_idx >= len(st.session_state.folder_images) - 1,
                        use_container_width=True
                    ):
                        st.session_state.current_index += 1
                        # 更新臨時標籤
                        next_img = st.session_state.folder_images[st.session_state.current_index]
                        next_default = ""
                        if st.session_state.current_index in st.session_state.ai_predictions:
                            next_ai = st.session_state.ai_predictions[st.session_state.current_index]
                            if (next_ai['confidence'] > 0.7 and 
                                SimpleCaptchaCorrector.validate_label(next_ai['text'])):
                                next_default = next_ai['text']
                        if not next_default and next_img['original_label']:
                            next_default = next_img['original_label']
                        st.session_state.temp_label = next_default
                        st.rerun()
                
                # 進度指示器
                st.markdown(f"**📍 進度**: {current_idx + 1} / {len(st.session_state.folder_images)}")
                progress_pct = (current_idx + 1) / len(st.session_state.folder_images)
                st.progress(progress_pct)

def single_image_recognition(predictor):
    """單張圖片識別"""
    st.markdown("## 📷 單張圖片識別")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🖼️ 上傳圖片")
        
        uploaded_file = st.file_uploader(
            "選擇驗證碼圖片",
            type=['png', 'jpg', 'jpeg'],
            help="支援PNG、JPG、JPEG格式"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="上傳的驗證碼", use_column_width=True)
            
            original_label = SimpleCaptchaCorrector.extract_label_from_filename(uploaded_file.name)
            if original_label:
                st.info(f"📝 檔名中的標籤: **{original_label}**")
    
    with col2:
        st.markdown("#### 🎯 識別結果")
        
        if uploaded_file is not None:
            if st.button("🚀 開始AI識別", type="primary", use_container_width=True):
                with st.spinner("🤖 AI正在識別中..."):
                    predicted_text, confidence = predictor.predict(image)
                
                if predicted_text:
                    st.markdown(f"""
                    <div class="ai-result">
                        🤖 AI識別結果: <strong>{predicted_text}</strong><br>
                        📊 置信度: {confidence:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(confidence)
                    
                    # 置信度評估
                    if confidence > 0.9:
                        st.success("🟢 高置信度 - 結果可信")
                    elif confidence > 0.7:
                        st.warning("🟡 中等置信度 - 建議檢查")
                    else:
                        st.warning("🟠 低置信度 - 需要驗證")
                    
                    # 結果修正
                    st.markdown("#### ✏️ 結果修正")
                    corrected_text = st.text_input(
                        "修正結果:",
                        value=predicted_text,
                        max_chars=4,
                        help="可以修正AI識別結果"
                    ).upper()
                    
                    is_valid = SimpleCaptchaCorrector.validate_label(corrected_text)
                    
                    if corrected_text and is_valid:
                        st.success(f"✅ 格式正確: {corrected_text}")
                        
                        if st.button("💾 確認結果", use_container_width=True):
                            st.markdown(f"""
                            <div class="success-result">
                                ✅ 已確認結果: <strong>{corrected_text}</strong><br>
                                建議檔名: {SimpleCaptchaCorrector.generate_new_filename(corrected_text)}
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                    elif corrected_text:
                        st.error("❌ 請輸入4個大寫英文字母")
                else:
                    st.error("❌ AI識別失敗，請嘗試其他圖片")

def statistics_analysis(predictor):
    """統計分析"""
    st.markdown("## 📊 統計分析")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🔧 技術規格")
        
        # 使用表格顯示技術規格
        specs_data = {
            "項目": [
                "模型架構",
                "支援字符",
                "字符數量", 
                "序列長度",
                "計算設備",
                "輸入尺寸",
                "隱藏層大小",
                "LSTM層數"
            ],
            "規格": [
                "CRNN (CNN + LSTM)",
                CHARACTERS,
                len(CHARACTERS),
                CAPTCHA_LENGTH_EXPECTED,
                "CPU" if not torch.cuda.is_available() else "CUDA",
                f"{DEFAULT_CONFIG['IMAGE_HEIGHT']}×{DEFAULT_CONFIG['IMAGE_WIDTH']}",
                DEFAULT_CONFIG['HIDDEN_SIZE'],
                DEFAULT_CONFIG['NUM_LAYERS']
            ]
        }
        
        import pandas as pd
        specs_df = pd.DataFrame(specs_data)
        st.dataframe(specs_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### 📈 性能指標")
        
        if predictor and predictor.model_info:
            # 使用指標卡片顯示
            st.metric(
                "訓練輪數",
                predictor.model_info.get('epoch', 'unknown')
            )
            
            accuracy = predictor.model_info.get('best_val_captcha_acc', 0)
            st.metric(
                "驗證準確率",
                f"{accuracy:.4f}",
                f"{accuracy*100:.2f}%"
            )
            
            st.metric(
                "推理速度",
                "~100ms/圖片",
                help="平均單張圖片處理時間"
            )
            
            st.metric(
                "支援格式",
                "PNG, JPG, JPEG"
            )
        else:
            st.warning("⚠️ 模型未正確載入，無法顯示性能指標")
    
    # 當前處理統計
    if st.session_state.folder_images:
        st.markdown("---")
        st.markdown("#### 📋 當前批次統計")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric(
                "📋 總檔案數",
                len(st.session_state.folder_images)
            )
        
        with col_stat2:
            st.metric(
                "✅ 已修正檔案",
                st.session_state.modified_count,
                f"{(st.session_state.modified_count/len(st.session_state.folder_images)*100):.1f}%"
            )
        
        with col_stat3:
            ai_accuracy = 0
            if st.session_state.modified_count > 0:
                ai_accuracy = (st.session_state.ai_accurate_count / st.session_state.modified_count) * 100
            st.metric(
                "🤖 AI準確率",
                f"{ai_accuracy:.1f}%",
                help="AI預測與最終標籤的匹配率"
            )
        
        with col_stat4:
            remaining = len(st.session_state.folder_images) - st.session_state.modified_count
            st.metric(
                "⏳ 剩餘處理",
                remaining,
                f"{remaining} 張圖片"
            )
        
        # AI預測信心度分布
        if st.session_state.ai_predictions:
            st.markdown("#### 📊 AI預測信心度分布")
            
            confidences = [pred['confidence'] for pred in st.session_state.ai_predictions.values()]
            
            # 統計不同信心度區間
            high_conf = sum(1 for c in confidences if c > 0.9)
            med_conf = sum(1 for c in confidences if 0.7 <= c <= 0.9)
            low_conf = sum(1 for c in confidences if c < 0.7)
            
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            
            with conf_col1:
                st.metric(
                    "🟢 高信心度 (>90%)",
                    high_conf,
                    f"{(high_conf/len(confidences)*100):.1f}%"
                )
            
            with conf_col2:
                st.metric(
                    "🟡 中信心度 (70-90%)",
                    med_conf,
                    f"{(med_conf/len(confidences)*100):.1f}%"
                )
            
            with conf_col3:
                st.metric(
                    "🟠 低信心度 (<70%)",
                    low_conf,
                    f"{(low_conf/len(confidences)*100):.1f}%"
                )
            
            # 顯示平均信心度
            avg_confidence = sum(confidences) / len(confidences)
            st.metric(
                "📊 平均信心度",
                f"{avg_confidence:.3f}",
                f"{avg_confidence*100:.1f}%"
            )

if __name__ == "__main__":
    main()