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

# 環境設定 - 完全修正 PyTorch 與 Streamlit 兼容性問題
import os
import sys
import warnings

# 在導入任何模組之前設置環境變數
os.environ['TORCH_DISABLE_EXTENSIONS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_JIT'] = '0'
os.environ['STREAMLIT_WATCHDOG_DISABLE'] = '1'
warnings.filterwarnings('ignore')

# 修正 sys.modules 來完全避免 torch.classes 問題
import types

# 創建一個虛假的 torch.classes 模組來滿足 Streamlit 的路徑檢查
class FakeTorchClasses:
    def __init__(self):
        self.__path__ = []
        self._path = []
    
    def __getattr__(self, name):
        if name in ['__path__', '_path']:
            return []
        return None

# 在導入 torch 之前預先註冊
if 'torch' not in sys.modules:
    fake_torch = types.ModuleType('torch')
    fake_torch.classes = FakeTorchClasses()
    fake_torch._classes = FakeTorchClasses()
    sys.modules['torch.classes'] = FakeTorchClasses()
    sys.modules['torch._classes'] = FakeTorchClasses()

import streamlit as st
from PIL import Image
import re
import string
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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

def safe_image_display(image, caption=None):
    """安全的圖片顯示函數，兼容不同版本的Streamlit"""
    try:
        # 嘗試新版本參數 (Streamlit >= 1.18.0)
        st.image(image, use_container_width=True, caption=caption)
    except TypeError:
        # 回退到舊版本參數 (Streamlit < 1.18.0)
        try:
            st.image(image, use_column_width=True, caption=caption)
        except TypeError:
            # 最基本的顯示方式
            st.image(image, caption=caption)
    except Exception as e:
        st.error(f"圖片顯示錯誤: {e}")
        if caption:
            st.text(f"圖片: {caption}")

# 頁面配置
st.set_page_config(
    page_title="CRNN AI Tool",  # 瀏覽器標籤簡潔標題
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 完全禁用 Streamlit 文件監視器來避免 torch.classes 問題
try:
    from streamlit.watcher import local_sources_watcher
    # 修補 get_module_paths 函數
    original_get_module_paths = local_sources_watcher.LocalSourcesWatcher._get_module_paths
    
    def safe_get_module_paths(self, module):
        """安全的模組路徑獲取，跳過 torch.classes"""
        try:
            if hasattr(module, '__name__') and 'torch' in str(module.__name__):
                return []
            return original_get_module_paths(self, module)
        except Exception:
            return []
    
    local_sources_watcher.LocalSourcesWatcher._get_module_paths = safe_get_module_paths
    
except Exception:
    pass

# 另一種方法：直接修補 extract_paths 函數
try:
    from streamlit.watcher.local_sources_watcher import extract_paths
    import importlib
    
    def safe_extract_paths(module):
        """安全的路徑提取，避免 torch.classes 問題"""
        try:
            if hasattr(module, '__name__') and 'torch' in str(module.__name__):
                return []
            if hasattr(module, '__path__'):
                if hasattr(module.__path__, '_path'):
                    return list(module.__path__._path)
                else:
                    return list(module.__path__)
            return []
        except Exception:
            return []
    
    # 替換原始函數
    import streamlit.watcher.local_sources_watcher as lsw
    lsw.extract_paths = safe_extract_paths
    
except Exception:
    pass

# 延遲導入 PyTorch - 完全修正兼容性問題
@st.cache_resource
def import_torch_modules():
    """安全地導入 PyTorch 模組，完全避免與 Streamlit 衝突"""
    try:
        # 進一步設置環境變數
        os.environ.setdefault('TORCH_DISABLE_EXTENSIONS', '1')
        os.environ.setdefault('PYTORCH_JIT', '0')
        os.environ.setdefault('TORCH_SHOW_CPP_STACKTRACES', '0')
        
        # 禁用 Streamlit 的模組監視
        if hasattr(st, 'config'):
            try:
                st.config.set_option('server.fileWatcherType', 'none')
            except:
                pass
        
        import torch
        import torch.nn as nn
        import torchvision.transforms as transforms
        
        # 完全修正 torch.classes 問題
        if hasattr(torch, '_classes'):
            try:
                # 創建一個安全的 __path__ 屬性
                class SafePath:
                    def __init__(self):
                        self._path = []
                    
                    def __iter__(self):
                        return iter([])
                    
                    def __getitem__(self, index):
                        raise IndexError("No paths available")
                    
                    def __len__(self):
                        return 0
                
                torch._classes.__path__ = SafePath()
                
            except Exception as e:
                pass
        
        # 同樣處理其他可能的問題模組
        if hasattr(torch, 'classes'):
            try:
                torch.classes.__path__ = []
            except:
                pass
                
        return torch, nn, transforms
        
    except ImportError as e:
        st.error(f"PyTorch 導入失敗: {e}")
        st.info("請安裝 PyTorch: pip install torch torchvision")
        return None, None, None
        
    except Exception as e:
        st.warning(f"PyTorch 配置警告 (可忽略): {str(e)[:100]}...")
        # 嘗試基本導入
        try:
            import torch
            import torch.nn as nn
            import torchvision.transforms as transforms
            return torch, nn, transforms
        except:
            st.info("運行於無 AI 模式")
            return None, None, None

# 模型配置 - 適配 Streamlit Cloud
MODEL_PATHS = [
    "./best_crnn_captcha_model.pth",  # 專案根目錄
    "./models/best_crnn_captcha_model.pth",  # models 子目錄
    "./trained_models/best_crnn_captcha_model.pth",  # trained_models 子目錄
    "best_crnn_captcha_model.pth",  # 當前目錄
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

# 優化版CSS - 修正檔案名稱顯示
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
    
    /* 全局背景顏色 */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
        color: #ecf0f1;
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
        color: #ecf0f1;
    }
    
    .status-compact.error {
        background: rgba(231, 76, 60, 0.15);
        border-color: #e74c3c;
        color: #e74c3c;
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
    
    .panel-content-maximized {
        flex: 1;
        overflow-y: auto;
        padding: 8px;
        min-height: 0;
    }
    
    /* 圖片列表按鈕 - 使用與"使用AI結果"相同的藍色hover效果 */
    .stButton > button {
        background: linear-gradient(135deg, #2c3e50, #34495e) !important;
        color: #ffffff !important;
        border: 2px solid #34495e !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        font-size: 1rem !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        min-height: 40px !important;
        line-height: 1.4 !important;
        outline: none !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.8) !important;
        letter-spacing: 1px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
        cursor: pointer !important;
        box-sizing: border-box !important;
    }
    
    .stButton > button:hover,
    .stButton > button:hover:not([kind="primary"]) {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;
        color: #ffffff !important;
        border: 2px solid #3498db !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3) !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.9) !important;
        outline: none !important;
    }
    
    .stButton > button:active {
        background: linear-gradient(135deg, #2980b9, #1f618d) !important;
        color: #ffffff !important;
        border: 2px solid #2980b9 !important;
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.4) !important;
        outline: none !important;
    }
    
    .stButton > button:focus,
    .stButton > button:focus-visible {
        background: linear-gradient(135deg, #3498db, #2980b9) !important;
        color: #ffffff !important;
        border: 2px solid #3498db !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2) !important;
    }
    
    /* 移除 Streamlit 默認的紅色 focus 樣式 */
    .stButton > button:focus:not(:focus-visible) {
        outline: none !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
    }
    
    /* 主要按鈕（選中狀態）- 移除紅色outline，優化點擊效果 */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #27ae60, #2ecc71) !important;
        color: white !important;
        font-weight: bold !important;
        border: 2px solid #2ecc71 !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        font-size: 1rem !important;
        min-height: 40px !important;
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3) !important;
        transition: all 0.3s ease !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.8) !important;
        letter-spacing: 1px !important;
        outline: none !important;
        box-sizing: border-box !important;
    }
    
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2ecc71, #1e8449) !important;
        color: white !important;
        border: 2px solid #1e8449 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(39, 174, 96, 0.4) !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.9) !important;
        outline: none !important;
    }
    
    div[data-testid="stButton"] button[kind="primary"]:active {
        background: linear-gradient(135deg, #1e8449, #145a32) !important;
        color: white !important;
        border: 2px solid #145a32 !important;
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(39, 174, 96, 0.5) !important;
        outline: none !important;
    }
    
    div[data-testid="stButton"] button[kind="primary"]:focus,
    div[data-testid="stButton"] button[kind="primary"]:focus-visible {
        background: linear-gradient(135deg, #2ecc71, #1e8449) !important;
        color: white !important;
        border: 2px solid #27ae60 !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(39, 174, 96, 0.3) !important;
    }
    
    /* 移除 primary 按鈕的默認紅色 focus 樣式 */
    div[data-testid="stButton"] button[kind="primary"]:focus:not(:focus-visible) {
        outline: none !important;
        box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3) !important;
    }
    
    /* 輸入框樣式 */
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
        border-color: #27ae60 !important;
        box-shadow: 0 0 12px rgba(39, 174, 96, 0.4) !important;
        transform: scale(1.02) !important;
    }
    
    .stTextInput > div > div > input:hover {
        border-color: #3498db !important;
        box-shadow: 0 0 8px rgba(52, 152, 219, 0.3) !important;
    }
    
    /* 檔案名稱顯示容器 - 修正為黑色文字 */
    .filename-display {
        text-align: center;
        margin-bottom: 12px;
        background: white !important;
        padding: 10px 12px !important;
        border-radius: 8px !important;
        border: 2px solid #ddd !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
    }
    
    .filename-text {
        color: #2c3e50 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        font-family: 'Consolas', 'Monaco', monospace !important;
        margin: 0 !important;
        letter-spacing: 0.5px !important;
    }
    
    /* 圖片顯示容器 */
    .image-display-container {
        text-align: center;
        background: white;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    
    /* 進度條顏色 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #27ae60, #f39c12, #2ecc71) !important;
    }
    
    /* 成功/錯誤訊息樣式 */
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
    
    /* 滾動條美化 */
    .panel-content-maximized::-webkit-scrollbar {
        width: 6px;
    }
    
    .panel-content-maximized::-webkit-scrollbar-track {
        background: #34495e;
        border-radius: 3px;
    }
    
    .panel-content-maximized::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3498db, #2980b9);
        border-radius: 3px;
    }
    
    /* 響應式調整 */
    @media (max-width: 1200px) {
        .work-area-maximized {
            height: calc(100vh - 200px);
            min-height: 400px;
        }
    }
    
    /* 全局按鈕樣式重置 - 移除所有 Streamlit 默認的紅色樣式 */
    .stButton > button, 
    .stButton > button:hover, 
    .stButton > button:active,
    .stButton > button:focus,
    .stButton > button:focus-visible,
    div[data-testid="stButton"] button,
    div[data-testid="stButton"] button:hover,
    div[data-testid="stButton"] button:active,
    div[data-testid="stButton"] button:focus,
    div[data-testid="stButton"] button:focus-visible,
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-secondary"]:hover,
    button[data-testid="baseButton-secondary"]:focus,
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-primary"]:focus {
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.8) !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        outline: none !important;
        box-shadow: none !important;
        border-color: transparent !important;
    }
    
    /* 重置所有按鈕的 focus 狀態 */
    .stButton > button:focus,
    .stButton > button:focus-visible,
    div[data-testid="stButton"] button:focus,
    div[data-testid="stButton"] button:focus-visible,
    button:focus,
    button:focus-visible {
        outline: none !important;
        border-color: inherit !important;
    }
    
    /* 移除 Streamlit 內建的紅色邊框 */
    .stButton > button:focus:not(.stButton > button[kind="primary"]) {
        border: 2px solid #3498db !important;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.3) !important;
    }
    
    /* 移除任何可能的紅色樣式 */
    *:focus {
        outline: none !important;
    }
    
    /* 特別針對按鈕區域的紅色移除 */
    .stButton,
    .stButton > button,
    div[data-testid="stButton"] {
        border: none !important;
        outline: none !important;
    }
    
    /* Progress 條文字顏色修正 */
    .stProgress .stProgress-text {
        color: #ffffff !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }
    
    /* 確保所有 info/warning/error 文字都是白色 */
    .stAlert {
        color: #ffffff !important;
    }
    
    .stAlert [data-testid="alertContent"] {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }
    .stMetric {
        background: rgba(52, 152, 219, 0.1) !important;
        border: 1px solid rgba(52, 152, 219, 0.3) !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
    
    .stMetric label,
    .stMetric [data-testid="metric-container"] > div {
        color: #ffffff !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
    }
    
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #3498db !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.7) !important;
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
        'folder_path': "./massive_real_captchas",  # Streamlit Cloud 預設路徑
        'temp_label': "",
        'list_page': 0,
        'initialized': True,
        'streamlit_version': st.__version__  # 記錄 Streamlit 版本
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

@st.cache_resource
def load_crnn_model():
    """載入 CRNN 模型，包含錯誤處理"""
    try:
        predictor = CRNNPredictor()
        
        model_path = None
        for file in MODEL_PATHS:
            try:
                if os.path.exists(file):
                    model_path = file
                    break
            except Exception:
                continue
        
        if model_path is None:
            st.warning("⚠️ 未找到模型檔案，AI識別功能將不可用")
            st.info("請確認以下路徑之一存在模型檔案：")
            for path in MODEL_PATHS:
                st.code(path)
            return None
        
        if predictor.load_model(model_path):
            st.success(f"✅ 模型載入成功: {model_path}")
            return predictor
        else:
            st.error(f"❌ 模型載入失敗: {model_path}")
            return None
            
    except Exception as e:
        st.error(f"❌ 模型載入異常: {e}")
        st.info("程式將在沒有AI功能的情況下運行")
        return None

def load_uploaded_images(uploaded_files):
    """載入上傳的圖片檔案"""
    try:
        if not uploaded_files:
            st.error("❌ 沒有選擇檔案")
            return False
        
        image_files_list = []
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        for uploaded_file in uploaded_files:
            # 檢查檔案類型
            if uploaded_file.type not in ['image/png', 'image/jpeg', 'image/jpg']:
                st.warning(f"⚠️ 跳過非圖片檔案: {uploaded_file.name}")
                continue
            
            # 保存到臨時目錄
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 添加到列表
            image_files_list.append({
                'name': uploaded_file.name,
                'path': str(file_path),
                'original_label': SimpleCaptchaCorrector.extract_label_from_filename(uploaded_file.name),
                'is_uploaded': True
            })
        
        if not image_files_list:
            st.error("❌ 沒有有效的圖片檔案")
            return False
        
        # 按檔名排序
        image_files_list.sort(key=lambda x: x['name'])
        
        # 更新 session state
        st.session_state.folder_images = image_files_list
        st.session_state.current_index = 0
        st.session_state.ai_predictions = {}
        st.session_state.modified_count = 0
        st.session_state.modified_files = set()
        st.session_state.ai_accurate_count = 0
        st.session_state.temp_label = ""
        
        st.success(f"✅ 成功載入 {len(image_files_list)} 張圖片")
        return True
        
    except Exception as e:
        st.error(f"❌ 載入上傳檔案時異常: {e}")
        return False
def load_images_from_folder(folder_path: str):
    """從資料夾載入圖片檔案"""
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
                    'original_label': SimpleCaptchaCorrector.extract_label_from_filename(p.name),
                    'is_uploaded': False
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
    """獲取當前圖片的預設標籤 - 優先使用原始標籤"""
    if not st.session_state.folder_images:
        return ""
    
    current_idx = st.session_state.current_index
    current_img = st.session_state.folder_images[current_idx]
    
    # 如果已經有有效的temp_label，並且用戶正在編輯，保持它
    if (hasattr(st.session_state, 'temp_label') and 
        st.session_state.temp_label and 
        SimpleCaptchaCorrector.validate_label(st.session_state.temp_label) and
        st.session_state.get('user_editing', False)):
        return st.session_state.temp_label
    
    # 1. 優先使用從檔名提取的原始標籤
    original_label = current_img.get('original_label', '')
    if original_label and SimpleCaptchaCorrector.validate_label(original_label):
        return original_label
    
    # 2. 如果沒有原始標籤，嘗試從檔名重新提取
    extracted_label = SimpleCaptchaCorrector.extract_label_from_filename(current_img['name'])
    if extracted_label and SimpleCaptchaCorrector.validate_label(extracted_label):
        return extracted_label
    
    # 3. 最後才使用AI預測結果（僅當置信度很高時）
    if current_idx in st.session_state.ai_predictions:
        ai_pred = st.session_state.ai_predictions[current_idx]
        if (ai_pred['confidence'] > 0.9 and 
            SimpleCaptchaCorrector.validate_label(ai_pred['text'])):
            return ai_pred['text']
    
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
    """保存當前檔案，支援上傳和本地檔案"""
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
        
        # 判斷是否為上傳檔案
        if current_file.get('is_uploaded', False):
            # 上傳檔案：在同一個臨時目錄重命名
            new_path = old_path.parent / new_filename
        else:
            # 本地檔案：在原目錄重命名
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
        
        # 更新檔案信息
        st.session_state.folder_images[current_idx] = {
            'name': new_filename,
            'path': str(new_path),
            'original_label': new_label,
            'is_uploaded': current_file.get('is_uploaded', False)
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
    
    # 主標題 - 修正雲端顯示問題，添加後備方案
    try:
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
                background: radial-gradient(circle, rgba(39,174,96,0.2), transparent);
                border-radius: 50%;
            "></div>
            <h1 style="
                font-size: 2.5rem;
                font-weight: 900;
                margin: 0 0 10px 0;
                color: #3498db;
                position: relative;
                z-index: 2;
                letter-spacing: 1px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.2;
                text-shadow: 0 2px 4px rgba(0,0,0,0.5);
            ">🎯 AI驗證碼識別工具</h1>
            <p style="
                font-size: 1rem;
                color: #27ae60;
                margin: 0;
                font-weight: 600;
                position: relative;
                z-index: 2;
                letter-spacing: 2px;
                text-transform: uppercase;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            ">CRNN模型 | 4位大寫英文字母識別</p>
            <div style="
                width: 60px;
                height: 4px;
                background: linear-gradient(90deg, #3498db, #27ae60);
                margin: 15px auto 0;
                border-radius: 2px;
                position: relative;
                z-index: 2;
            "></div>
        </div>
        ''', unsafe_allow_html=True)
    except Exception:
        # 後備純文字標題
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #1a1a2e; border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: #3498db; font-size: 2rem; margin: 0;">🎯 AI驗證碼識別工具</h1>
            <p style="color: #27ae60; font-size: 1rem; margin: 10px 0 0 0;">CRNN模型 | 4位大寫英文字母識別</p>
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # 路徑控制 - 支援 Streamlit Cloud 和檔案上傳
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 2, 1])
    
    with col1:
        if st.button("📁專案", key="path_project", use_container_width=True):
            st.session_state.folder_path = "./massive_real_captchas"
            safe_rerun()
    with col2:
        if st.button("📂範例", key="path_samples", use_container_width=True):
            st.session_state.folder_path = "./samples"
            safe_rerun()
    with col3:
        if st.button("📤上傳", key="upload_files", use_container_width=True):
            st.session_state.folder_path = "UPLOAD_MODE"
            safe_rerun()
    with col4:
        if st.button("🧪偵錯", key="path_debug", use_container_width=True):
            st.session_state.folder_path = "./debug_captchas"
            safe_rerun()
    with col5:
        folder_path = st.text_input(
            "路徑",
            value=st.session_state.folder_path if st.session_state.folder_path != "UPLOAD_MODE" else "./massive_real_captchas",
            placeholder="PNG圖片資料夾路徑",
            key="folder_path_input",
            label_visibility="collapsed",
            disabled=st.session_state.folder_path == "UPLOAD_MODE"
        )
        if st.session_state.folder_path != "UPLOAD_MODE":
            st.session_state.folder_path = folder_path
    with col6:
        if st.button("🚀載入", type="primary", key="load_images", use_container_width=True):
            if st.session_state.folder_path == "UPLOAD_MODE":
                # 觸發檔案上傳模式
                pass
            elif folder_path.strip():
                if load_images_from_folder(folder_path.strip()):
                    if st.session_state.folder_images and predictor:
                        with st.spinner("🤖 AI識別中..."):
                            perform_batch_ai_prediction(predictor)
                    safe_rerun()
            else:
                st.error("❌ 請輸入路徑")
    
    # 檔案上傳區域
    if st.session_state.folder_path == "UPLOAD_MODE":
        st.markdown("### 📤 上傳驗證碼圖片")
        uploaded_files = st.file_uploader(
            "選擇 PNG 圖片檔案",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="可以一次選擇多個圖片檔案上傳"
        )
        
        if uploaded_files:
            if st.button("✅ 處理上傳的圖片", type="primary", key="process_uploads"):
                if load_uploaded_images(uploaded_files):
                    if st.session_state.folder_images and predictor:
                        with st.spinner("🤖 AI識別中..."):
                            perform_batch_ai_prediction(predictor)
                    safe_rerun()
            
            # 顯示上傳的檔案列表
            st.write(f"已選擇 {len(uploaded_files)} 個檔案：")
            for i, file in enumerate(uploaded_files[:5]):  # 只顯示前5個
                st.write(f"• {file.name}")
            if len(uploaded_files) > 5:
                st.write(f"... 還有 {len(uploaded_files) - 5} 個檔案")
    
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
            <p style="font-size: 1.1rem; margin-bottom: 15px; color: #ecf0f1;">選擇圖片來源</p>
            <div style="
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                max-width: 600px;
                margin: 20px auto;
            ">
                <div style="
                    background: rgba(52, 152, 219, 0.1); 
                    border: 2px solid #3498db; 
                    border-radius: 10px; 
                    padding: 20px;
                ">
                    <h4 style="color: #3498db; margin-bottom: 10px;">🌐 雲端資料夾</h4>
                    <p style="color: #ecf0f1; font-size: 0.9rem; line-height: 1.5;">
                        使用預設的專案資料夾或輸入雲端路徑
                    </p>
                    <ul style="text-align: left; color: #bdc3c7; font-size: 0.85rem; margin: 10px 0;">
                        <li>📁 專案資料夾</li>
                        <li>📂 範例圖片</li>
                        <li>🧪 偵錯資料</li>
                    </ul>
                </div>
                <div style="
                    background: rgba(39, 174, 96, 0.1); 
                    border: 2px solid #27ae60; 
                    border-radius: 10px; 
                    padding: 20px;
                ">
                    <h4 style="color: #27ae60; margin-bottom: 10px;">📤 上傳檔案</h4>
                    <p style="color: #ecf0f1; font-size: 0.9rem; line-height: 1.5;">
                        從您的電腦上傳驗證碼圖片
                    </p>
                    <ul style="text-align: left; color: #bdc3c7; font-size: 0.85rem; margin: 10px 0;">
                        <li>支援 PNG, JPG 格式</li>
                        <li>可批量上傳多個檔案</li>
                        <li>即時 AI 識別</li>
                    </ul>
                </div>
            </div>
            <div style="
                background: rgba(230, 126, 34, 0.1); 
                border: 2px solid #e67e22; 
                border-radius: 10px; 
                padding: 20px; 
                margin: 20px auto;
                max-width: 600px;
            ">
                <h4 style="color: #e67e22; margin-bottom: 10px;">🎯 功能特色</h4>
                <ul style="text-align: left; color: #ecf0f1; line-height: 1.6;">
                    <li>🤖 <strong>AI自動識別</strong> - 使用CRNN模型識別4位大寫英文字母</li>
                    <li>📝 <strong>手動修正</strong> - 可以手動編輯AI識別結果</li>
                    <li>📊 <strong>即時統計</strong> - 顯示處理進度和AI準確率</li>
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
                # 標題 - 左側面板，綠色主題
                st.markdown('''
                <div style="
                    background: linear-gradient(135deg, #27ae60, #2ecc71);
                    color: white;
                    text-align: center;
                    padding: 14px 18px;
                    border-radius: 10px;
                    margin-bottom: 18px;
                    box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
                    border: 2px solid #27ae60;
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
                                    
                                    # 簡化顯示格式 - 只顯示4位英文字母
                                    original_label = img_info.get('original_label', '')
                                    ai_pred = st.session_state.ai_predictions.get(left_idx, {})
                                    
                                    # 優先使用原始標籤，若無則使用AI預測，最後使用檔名提取
                                    if original_label:
                                        display_text = original_label
                                    elif ai_pred and SimpleCaptchaCorrector.validate_label(ai_pred.get('text', '')):
                                        display_text = ai_pred['text']
                                    else:
                                        # 從檔名提取4位字母
                                        extracted = SimpleCaptchaCorrector.extract_label_from_filename(img_info['name'])
                                        display_text = extracted if extracted else "----"
                                    
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
                                    
                                    # 簡化顯示格式 - 只顯示4位英文字母
                                    original_label = img_info.get('original_label', '')
                                    ai_pred = st.session_state.ai_predictions.get(right_idx, {})
                                    
                                    # 優先使用原始標籤，若無則使用AI預測，最後使用檔名提取
                                    if original_label:
                                        display_text = original_label
                                    elif ai_pred and SimpleCaptchaCorrector.validate_label(ai_pred.get('text', '')):
                                        display_text = ai_pred['text']
                                    else:
                                        # 從檔名提取4位字母
                                        extracted = SimpleCaptchaCorrector.extract_label_from_filename(img_info['name'])
                                        display_text = extracted if extracted else "----"
                                    
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
                # 標題 - 中央面板，藍色主題
                st.markdown('''
                <div style="
                    background: linear-gradient(135deg, #3498db, #2980b9);
                    color: white;
                    text-align: center;
                    padding: 16px 22px;
                    border-radius: 12px;
                    margin-bottom: 22px;
                    box-shadow: 0 5px 18px rgba(52, 152, 219, 0.4);
                    border: 3px solid #2980b9;
                    position: relative;
                    overflow: hidden;
                ">
                    <div style="
                        position: absolute;
                        top: -50%;
                        right: -20px;
                        width: 80px;
                        height: 80px;
                        background: rgba(255,255,255,0.1);
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
                        
                        # 檔案名稱顯示 - 優先顯示從檔名提取的4位字母，否則顯示AI識別結果
                        current_idx = st.session_state.current_index
                        
                        # 從檔名提取4位字母
                        label_from_filename = SimpleCaptchaCorrector.extract_label_from_filename(current_img["name"])
                        
                        # 如果檔名中沒有4位字母，嘗試使用AI識別結果
                        if not label_from_filename and current_idx in st.session_state.ai_predictions:
                            ai_pred = st.session_state.ai_predictions[current_idx]
                            if SimpleCaptchaCorrector.validate_label(ai_pred['text']):
                                label_from_filename = ai_pred['text']
                        
                        # 最終顯示：4位字母 > 完整檔名
                        display_filename = label_from_filename if label_from_filename else current_img["name"]
                        
                        st.markdown(f'''
                        <div class="filename-display">
                            <p class="filename-text">📄 {display_filename}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # 圖片顯示 - 使用兼容性函數
                        safe_image_display(image)
                        
                        # 快速信息 - 修正所有文字顏色為白色
                        current_idx = st.session_state.current_index
                        original_label = current_img.get('original_label', '')
                        
                        # 使用自定義的 metric 樣式，標籤和數值都是白色
                        info_col1, info_col2, info_col3 = st.columns(3)
                        with info_col1:
                            st.markdown(f'''
                            <div style="
                                background: rgba(39, 174, 96, 0.15);
                                border: 2px solid #27ae60;
                                border-radius: 8px;
                                padding: 12px;
                                text-align: center;
                                margin: 4px 0;
                            ">
                                <div style="color: #ffffff; font-size: 0.8rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">序號</div>
                                <div style="color: #ffffff; font-size: 1.5rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">#{current_idx + 1}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        with info_col2:
                            st.markdown(f'''
                            <div style="
                                background: rgba(52, 152, 219, 0.15);
                                border: 2px solid #3498db;
                                border-radius: 8px;
                                padding: 12px;
                                text-align: center;
                                margin: 4px 0;
                            ">
                                <div style="color: #ffffff; font-size: 0.8rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">原始標籤</div>
                                <div style="color: #ffffff; font-size: 1.5rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">{original_label or "無"}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        with info_col3:
                            if current_idx in st.session_state.ai_predictions:
                                ai_pred = st.session_state.ai_predictions[current_idx]
                                ai_text = f"{ai_pred['text']} ({ai_pred['confidence']:.0%})"
                            else:
                                ai_text = "等待中"
                            
                            st.markdown(f'''
                            <div style="
                                background: rgba(230, 126, 34, 0.15);
                                border: 2px solid #e67e22;
                                border-radius: 8px;
                                padding: 12px;
                                text-align: center;
                                margin: 4px 0;
                            ">
                                <div style="color: #ffffff; font-size: 0.8rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">AI識別</div>
                                <div style="color: #ffffff; font-size: 1.2rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">{ai_text}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ 無法載入圖片: {str(e)}")
        
        # 右側：控制面板
        with col3:
            with st.container():
                # 標題 - 右側面板，綠色主題
                st.markdown('''
                <div style="
                    background: linear-gradient(135deg, #2ecc71, #27ae60);
                    color: white;
                    text-align: center;
                    padding: 14px 18px;
                    border-radius: 10px;
                    margin-bottom: 18px;
                    box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
                    border: 2px solid #1e8449;
                    position: relative;
                ">
                    <div style="
                        position: absolute;
                        top: 5px;
                        right: 5px;
                        width: 8px;
                        height: 8px;
                        background: #3498db;
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
                    
                    # 1. AI識別結果
                    st.markdown("#### 🤖 AI識別")
                    
                    if current_idx in st.session_state.ai_predictions:
                        ai_pred = st.session_state.ai_predictions[current_idx]
                        confidence = ai_pred['confidence']
                        
                        # 使用自定義樣式顯示AI結果，確保白色文字
                        st.markdown(f'''
                        <div style="
                            background: rgba(52, 152, 219, 0.15);
                            border: 2px solid #3498db;
                            border-radius: 8px;
                            padding: 12px;
                            margin: 8px 0;
                            text-align: center;
                        ">
                            <div style="color: #ffffff; font-size: 0.9rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">AI結果</div>
                            <div style="color: #ffffff; font-size: 1.8rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.7); margin: 5px 0;">{ai_pred['text']}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        st.progress(confidence, text=f"置信度: {confidence:.1%}")
                        
                        if st.button("🎯 使用AI結果", key=f"ctrl_use_ai_{current_idx}", use_container_width=True):
                            if SimpleCaptchaCorrector.validate_label(ai_pred['text']):
                                st.session_state.temp_label = ai_pred['text']
                                trigger_key = f'update_input_{current_idx}'
                                st.session_state[trigger_key] = st.session_state.get(trigger_key, 0) + 1
                                st.success(f"✅ 已填入: {ai_pred['text']}")
                                safe_rerun()
                            else:
                                st.warning("⚠️ AI結果格式無效")
                    else:
                        st.markdown(f'''
                        <div style="
                            background: rgba(108, 117, 125, 0.15);
                            border: 2px solid #6c757d;
                            border-radius: 8px;
                            padding: 12px;
                            margin: 8px 0;
                            text-align: center;
                        ">
                            <div style="color: #ffffff; font-size: 1rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">等待AI識別...</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # 2. 標籤編輯
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
                    is_valid = SimpleCaptchaCorrector.validate_label(new_label)
                    
                    # 驗證狀態
                    if new_label:
                        if is_valid:
                            st.success("✅ 格式正確")
                        else:
                            st.error("❌ 需要4個大寫字母")
                    
                    # 3. 保存按鈕
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
                    
                    # 4. 導航
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
                    
                    # 5. 進度顯示
                    progress = (current_idx + 1) / len(st.session_state.folder_images)
                    st.progress(progress, text=f"進度: {current_idx + 1}/{len(st.session_state.folder_images)}")
                    
                    # 6. 快速跳轉 (上移)
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
                    
                    # 7. 統計區塊 (下移) - 所有文字都改為白色
                    st.markdown("#### 📊 統計")
                    
                    # 使用自定義樣式的 metric 顯示，標籤和數值都是白色文字
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.markdown(f'''
                        <div style="
                            background: rgba(39, 174, 96, 0.15);
                            border: 2px solid #27ae60;
                            border-radius: 8px;
                            padding: 12px;
                            text-align: center;
                            margin: 4px 0;
                        ">
                            <div style="color: #ffffff; font-size: 0.8rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">總檔案</div>
                            <div style="color: #ffffff; font-size: 1.8rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">{len(st.session_state.folder_images)}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with col_stat2:
                        st.markdown(f'''
                        <div style="
                            background: rgba(52, 152, 219, 0.15);
                            border: 2px solid #3498db;
                            border-radius: 8px;
                            padding: 12px;
                            text-align: center;
                            margin: 4px 0;
                        ">
                            <div style="color: #ffffff; font-size: 0.8rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">已修正</div>
                            <div style="color: #ffffff; font-size: 1.8rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">{st.session_state.modified_count}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # AI準確率單獨顯示 - 白色文字
                    if st.session_state.modified_count > 0:
                        ai_acc = (st.session_state.ai_accurate_count / st.session_state.modified_count) * 100
                        ai_acc_text = f"{ai_acc:.0f}%"
                    else:
                        ai_acc_text = "0%"
                    
                    st.markdown(f'''
                    <div style="
                        background: rgba(230, 126, 34, 0.15);
                        border: 2px solid #e67e22;
                        border-radius: 8px;
                        padding: 12px;
                        text-align: center;
                        margin: 8px 0;
                    ">
                        <div style="color: #ffffff; font-size: 0.8rem; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">AI準確率</div>
                        <div style="color: #ffffff; font-size: 1.8rem; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">{ai_acc_text}</div>
                    </div>
                    ''', unsafe_allow_html=True)

def main():
    """主函數，包含錯誤處理和優雅降級"""
    try:
        if 'initialized' not in st.session_state:
            init_session_state()
        
        # 載入模型（如果失敗會返回 None）
        predictor = load_crnn_model()
        
        # 顯示模型狀態
        if predictor is None:
            st.info("ℹ️ 運行於手動模式 - AI識別功能不可用，但仍可進行手動標籤編輯")
        
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
            st.write(f"PyTorch 可用: {predictor is not None}")
            
            # 顯示版本信息
            st.write("**版本信息:**")
            st.write(f"Streamlit 版本: {st.__version__}")
            st.write(f"Python 版本: {sys.version}")
            
            # 檢查 Streamlit 功能支援
            has_container_width = hasattr(st.image, '__code__') and 'use_container_width' in st.image.__code__.co_varnames
            st.write(f"支援 use_container_width: {has_container_width}")
            
            if predictor and predictor.torch:
                st.write(f"PyTorch 版本: {predictor.torch.__version__}")
                st.write(f"CUDA 可用: {predictor.torch.cuda.is_available()}")
                
    except Exception as e:
        st.error(f"❌ 應用程式錯誤: {e}")
        st.info("請重新整理頁面或聯繫開發者")
        
        # 顯示詳細錯誤信息（僅在調試模式）
        if st.checkbox("顯示詳細錯誤", key="show_detailed_error"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()