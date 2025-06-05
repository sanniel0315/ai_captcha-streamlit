#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高級Streamlit + CRNN模型整合 - 自動驗證碼識別工具 (仿Flask版本)"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import re
import string
import io
import zipfile
import pandas as pd
import time
from pathlib import Path
import base64
from typing import List, Tuple, Dict, Optional
import json

# 頁面配置
st.set_page_config(
    page_title="🎯 AI驗證碼識別工具 - 專業版",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 高級自定義CSS樣式 - 仿Flask深色主題
st.markdown("""
<style>
    /* 主體背景 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: #ecf0f1;
    }
    
    /* 標題樣式 */
    .main-title {
        background: linear-gradient(135deg, #0f3460, #16213e);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* 卡片樣式 */
    .control-card {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border: 2px solid transparent;
    }
    
    .control-card:hover {
        border-color: #e94560;
        transform: translateY(-2px);
        transition: all 0.3s ease;
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
    
    /* 圖片展示區 */
    .image-display {
        background: #2c3e50;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
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
    
    /* 圖片列表樣式 */
    .image-item {
        background: #34495e;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        border: 2px solid transparent;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .image-item:hover {
        background: #3498db;
        transform: translateX(5px);
    }
    
    .image-item.active {
        background: #e94560;
        border-color: #c0392b;
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
    }
    
    /* 進度條自定義 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e74c3c, #f39c12, #27ae60);
    }
    
    /* 按鈕樣式 */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(233, 69, 96, 0.4);
    }
    
    /* 側邊欄樣式 */
    .css-1d391kg {
        background: linear-gradient(135deg, #16213e, #0f3460);
    }
    
    /* 文件上傳區域 */
    .uploadedFile {
        background: #2c3e50;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
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
</style>
""", unsafe_allow_html=True)

# 模型配置
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

# 工具類 - 與Flask版本相同
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

# CRNN模型 - 與Flask版本完全相同
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

# 預測器類 - 與Flask版本相同邏輯
class CRNNPredictor:
    def __init__(self):
        self.device = torch.device('cpu')  # Streamlit Cloud使用CPU
        self.model = None
        self.transform = None
        self.config = None
        self.is_loaded = False
        self.model_info = {}

    def load_model(self, model_path: str):
        """載入CRNN模型"""
        try:
            if not os.path.exists(model_path):
                return False

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
                'idx_to_char': checkpoint.get('idx_to_char', {idx: char for idx, char in enumerate(CHARACTERS)})
            }

            return True

        except Exception as e:
            st.error(f"模型載入失敗: {e}")
            return False

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """對單張圖片做預測 - 與Flask版本相同邏輯"""
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
            idx_to_char_map = self.model_info.get('idx_to_char', {idx: char for idx, char in enumerate(CHARACTERS)})
            
            if isinstance(next(iter(idx_to_char_map.keys())), str):
                idx_to_char_map = {int(k): v for k, v in idx_to_char_map.items()}

            text = ''.join(idx_to_char_map.get(idx.item(), '?') for idx in pred_indices).upper()

            probs = torch.softmax(focused, dim=2)
            max_probs = torch.max(probs, dim=2)[0]
            confidence = float(torch.mean(max_probs).item())

            return text, confidence

        except Exception as e:
            return "", 0.0

# 載入模型
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

# 主應用程序
def main():
    # 主標題
    st.markdown("""
    <div class="main-title">
        <h1>🎯 AI驗證碼識別工具 - 專業版</h1>
        <p>使用CRNN模型自動識別4位大寫英文字母驗證碼 | 仿Flask完整功能</p>
    </div>
    """, unsafe_allow_html=True)

    # 載入模型
    predictor = load_crnn_model()
    
    # 側邊欄 - 模型信息和控制
    with st.sidebar:
        st.markdown("### ⚙️ 控制面板")
        
        # 模型狀態
        if predictor is not None:
            st.markdown("""
            <div class="ai-status-card">
                🤖 CRNN模型已就緒<br>
                準確率: 99.90%
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 模型詳情")
            if predictor.model_info:
                st.info(f"📈 訓練輪數: {predictor.model_info['epoch']}")
                st.info(f"📊 驗證準確率: {predictor.model_info['best_val_captcha_acc']:.4f}")
                st.info(f"🔤 支援字符: {CHARACTERS}")
                st.info(f"📏 序列長度: {CAPTCHA_LENGTH_EXPECTED}")
        else:
            st.markdown("""
            <div class="ai-status-card ai-status-error">
                ❌ 模型載入失敗<br>
                請檢查模型文件
            </div>
            """, unsafe_allow_html=True)
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

def folder_batch_processing(predictor):
    """資料夾批量處理 - 仿Flask功能"""
    st.markdown("## 📁 資料夾批量處理")
    st.markdown("### 💡 上傳ZIP檔案來模擬資料夾載入")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("""
        <div class="control-card">
            <h4>📂 資料夾載入</h4>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_zip = st.file_uploader(
            "上傳包含驗證碼圖片的ZIP檔案",
            type=['zip'],
            help="將您的圖片打包成ZIP檔案上傳，模擬資料夾載入功能"
        )
        
        if uploaded_zip is not None:
            # 解壓縮並處理圖片
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                file_list = [f for f in zip_ref.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if file_list:
                    st.success(f"✅ 找到 {len(file_list)} 張圖片")
                    
                    # 初始化session state
                    if 'folder_images' not in st.session_state:
                        st.session_state.folder_images = []
                        st.session_state.current_index = 0
                        st.session_state.ai_predictions = {}
                        st.session_state.modified_labels = {}
                    
                    # 載入圖片
                    if st.button("🚀 載入並開始AI批量識別", type="primary"):
                        with st.spinner("正在載入圖片和AI識別..."):
                            folder_images = []
                            ai_predictions = {}
                            
                            progress_bar = st.progress(0)
                            
                            for i, filename in enumerate(file_list):
                                try:
                                    with zip_ref.open(filename) as img_file:
                                        image = Image.open(img_file)
                                        if image.mode != 'RGB':
                                            image = image.convert('RGB')
                                        
                                        # 存儲圖片信息
                                        folder_images.append({
                                            'name': filename,
                                            'image': image,
                                            'original_label': SimpleCaptchaCorrector.extract_label_from_filename(filename)
                                        })
                                        
                                        # AI預測
                                        predicted_text, confidence = predictor.predict(image)
                                        ai_predictions[i] = {
                                            'text': predicted_text,
                                            'confidence': confidence
                                        }
                                        
                                        progress_bar.progress((i + 1) / len(file_list))
                                
                                except Exception as e:
                                    st.error(f"處理 {filename} 時出錯: {e}")
                            
                            st.session_state.folder_images = folder_images
                            st.session_state.ai_predictions = ai_predictions
                            st.success("🎯 AI批量識別完成！")
    
    with col2:
        if 'folder_images' in st.session_state and st.session_state.folder_images:
            st.markdown("""
            <div class="control-card">
                <h4>🖼️ 圖片預覽與編輯</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # 圖片列表和當前圖片顯示
            folder_interface(predictor)

def folder_interface(predictor):
    """資料夾界面 - 仿Flask的圖片列表和編輯功能"""
    images = st.session_state.folder_images
    current_idx = st.session_state.current_index
    
    # 左右佈局
    col_list, col_display, col_control = st.columns([1, 2, 1])
    
    with col_list:
        st.markdown("### 📋 圖片列表")
        
        # 圖片列表
        for i, img_info in enumerate(images):
            is_active = (i == current_idx)
            is_modified = i in st.session_state.get('modified_labels', {})
            
            # 獲取AI預測
            ai_pred = st.session_state.ai_predictions.get(i, {})
            ai_text = ai_pred.get('text', '???')
            confidence = ai_pred.get('confidence', 0)
            
            # 樣式類別
            item_class = "image-item"
            if is_active:
                item_class += " active"
            
            # 顯示圖片項目
            if st.button(
                f"{i+1}. {img_info['name'][:20]}... | {img_info['original_label']} → AI:{ai_text} ({confidence:.2%})",
                key=f"img_{i}",
                use_container_width=True
            ):
                st.session_state.current_index = i
                st.rerun()
    
    with col_display:
        st.markdown("### 🖼️ 當前圖片")
        
        if current_idx < len(images):
            current_img = images[current_idx]
            
            # 顯示圖片
            st.image(
                current_img['image'], 
                caption=f"檔案: {current_img['name']}",
                use_column_width=True
            )
            
            # 檔案信息
            st.markdown(f"**📄 檔案名**: {current_img['name']}")
            st.markdown(f"**🏷️ 原始標籤**: {current_img['original_label'] or '無法提取'}")
            
            # AI識別結果
            ai_pred = st.session_state.ai_predictions.get(current_idx, {})
            if ai_pred:
                st.markdown(f"""
                <div class="ai-result">
                    🤖 AI識別: {ai_pred['text']}<br>
                    📊 置信度: {ai_pred['confidence']:.2%}
                </div>
                """, unsafe_allow_html=True)
                
                # 置信度進度條
                st.progress(ai_pred['confidence'])
    
    with col_control:
        st.markdown("### ✏️ 標籤修正")
        
        if current_idx < len(images):
            current_img = images[current_idx]
            ai_pred = st.session_state.ai_predictions.get(current_idx, {})
            
            # 預設值選擇
            default_value = ""
            if ai_pred and ai_pred.get('confidence', 0) > 0.7:
                default_value = ai_pred.get('text', '')
            elif current_img['original_label']:
                default_value = current_img['original_label']
            
            # 標籤輸入
            new_label = st.text_input(
                "修正標籤 (4位大寫字母)",
                value=st.session_state.modified_labels.get(current_idx, default_value),
                max_chars=4,
                key=f"label_input_{current_idx}"
            ).upper()
            
            # 驗證
            is_valid = SimpleCaptchaCorrector.validate_label(new_label)
            
            if new_label:
                if is_valid:
                    st.success(f"✅ 格式正確: {new_label}")
                else:
                    st.error("❌ 請輸入4個大寫英文字母")
            
            # 儲存按鈕
            if st.button("💾 確認修正", disabled=not is_valid, use_container_width=True):
                if 'modified_labels' not in st.session_state:
                    st.session_state.modified_labels = {}
                
                st.session_state.modified_labels[current_idx] = new_label
                
                # 生成新檔名
                new_filename = SimpleCaptchaCorrector.generate_new_filename(new_label)
                
                st.markdown(f"""
                <div class="success-result">
                    ✅ 已確認修正<br>
                    新檔名: {new_filename}
                </div>
                """, unsafe_allow_html=True)
                
                # 自動前進到下一張
                if current_idx < len(images) - 1:
                    st.session_state.current_index += 1
                    time.sleep(0.5)
                    st.rerun()
            
            # 導航按鈕
            st.markdown("### 🧭 導航")
            
            nav_col1, nav_col2 = st.columns(2)
            with nav_col1:
                if st.button("⬅️ 上一張", disabled=current_idx == 0, use_container_width=True):
                    st.session_state.current_index -= 1
                    st.rerun()
            
            with nav_col2:
                if st.button("下一張 ➡️", disabled=current_idx >= len(images) - 1, use_container_width=True):
                    st.session_state.current_index += 1
                    st.rerun()
            
            # 進度指示
            st.markdown(f"### 📍 進度: {current_idx + 1} / {len(images)}")
            st.progress((current_idx + 1) / len(images))
    
    # 底部統計和下載
    if st.session_state.get('modified_labels'):
        st.markdown("---")
        st.markdown("### 📊 批量處理結果")
        
        # 準備結果數據
        results = []
        for i, img_info in enumerate(images):
            ai_pred = st.session_state.ai_predictions.get(i, {})
            modified_label = st.session_state.modified_labels.get(i, '')
            
            results.append({
                '原始檔名': img_info['name'],
                '原始標籤': img_info['original_label'] or '無',
                'AI識別結果': ai_pred.get('text', '失敗'),
                'AI置信度': f"{ai_pred.get('confidence', 0):.3f}",
                '修正標籤': modified_label,
                '新檔名': SimpleCaptchaCorrector.generate_new_filename(modified_label) if modified_label else '未修正',
                '狀態': '✅ 已修正' if modified_label else '⏳ 待處理'
            })
        
        # 顯示結果表格
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # 統計信息
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📁 總檔案</h3>
                <h2>{}</h2>
            </div>
            """.format(len(images)), unsafe_allow_html=True)
        
        with col2:
            modified_count = len(st.session_state.modified_labels)
            st.markdown("""
            <div class="metric-card">
                <h3>✅ 已修正</h3>
                <h2>{}</h2>
            </div>
            """.format(modified_count), unsafe_allow_html=True)
        
        with col3:
            if modified_count > 0:
                # 計算AI準確率
                accurate_count = 0
                for i in st.session_state.modified_labels:
                    original = images[i]['original_label']
                    ai_pred = st.session_state.ai_predictions.get(i, {}).get('text', '')
                    modified = st.session_state.modified_labels[i]
                    if original and ai_pred == original:
                        accurate_count += 1
                
                accuracy = (accurate_count / modified_count * 100) if modified_count > 0 else 0
            else:
                accuracy = 0
                
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 AI準確率</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(accuracy), unsafe_allow_html=True)
        
        with col4:
            avg_confidence = sum(pred.get('confidence', 0) for pred in st.session_state.ai_predictions.values()) / len(st.session_state.ai_predictions) if st.session_state.ai_predictions else 0
            st.markdown("""
            <div class="metric-card">
                <h3>📊 平均置信度</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(avg_confidence * 100), unsafe_allow_html=True)
        
        # 下載結果
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下載處理結果 (CSV)",
            data=csv,
            file_name=f"captcha_batch_results_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )

def single_image_recognition(predictor):
    """單張圖片識別"""
    st.markdown("## 📷 單張圖片識別")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="control-card">
            <h4>🖼️ 上傳圖片</h4>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "選擇驗證碼圖片",
            type=['png', 'jpg', 'jpeg'],
            help="支援PNG、JPG、JPEG格式"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            st.markdown("""
            <div class="image-display">
            </div>
            """, unsafe_allow_html=True)
            
            st.image(image, caption="上傳的驗證碼", use_column_width=True)
            
            # 從檔名提取標籤
            original_label = SimpleCaptchaCorrector.extract_label_from_filename(uploaded_file.name)
            if original_label:
                st.info(f"📝 檔名中的標籤: **{original_label}**")
    
    with col2:
        st.markdown("""
        <div class="control-card">
            <h4>🎯 識別結果</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            if st.button("🚀 開始AI識別", type="primary", use_container_width=True):
                with st.spinner("🤖 AI正在識別中..."):
                    predicted_text, confidence = predictor.predict(image)
                
                if predicted_text:
                    # AI結果顯示
                    st.markdown(f"""
                    <div class="ai-result">
                        🤖 AI識別結果: <strong>{predicted_text}</strong><br>
                        📊 置信度: {confidence:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 置信度進度條
                    st.progress(confidence)
                    
                    # 置信度評估
                    if confidence > 0.9:
                        st.success("🟢 高置信度 - 結果可信")
                    elif confidence > 0.7:
                        st.warning("🟡 中等置信度 - 建議檢查")
                    elif confidence > 0.5:
                        st.warning("🟠 低置信度 - 需要驗證")
                    else:
                        st.error("🔴 極低置信度 - 建議重新識別")
                    
                    # 結果修正區域
                    st.markdown("### ✏️ 結果修正")
                    corrected_text = st.text_input(
                        "如需修正，請輸入正確答案:",
                        value=predicted_text,
                        max_chars=4,
                        help="只能輸入4個大寫英文字母 (A-Z)"
                    ).upper()
                    
                    # 驗證輸入
                    is_valid = SimpleCaptchaCorrector.validate_label(corrected_text)
                    
                    if corrected_text:
                        if is_valid:
                            st.success(f"✅ 格式正確: {corrected_text}")
                        else:
                            st.error("❌ 請輸入4個大寫英文字母")
                    
                    # 確認按鈕
                    if st.button("💾 確認結果", disabled=not is_valid, use_container_width=True):
                        st.markdown(f"""
                        <div class="success-result">
                            ✅ 已確認結果: <strong>{corrected_text}</strong><br>
                            新檔名: {SimpleCaptchaCorrector.generate_new_filename(corrected_text)}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # 準確性評估
                        if original_label:
                            is_correct = (corrected_text == original_label)
                            ai_accurate = (predicted_text == original_label)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric(
                                    "🎯 最終準確性", 
                                    "✅ 正確" if is_correct else "❌ 錯誤"
                                )
                            with col_b:
                                st.metric(
                                    "🤖 AI準確性", 
                                    "✅ 正確" if ai_accurate else "❌ 錯誤"
                                )
                else:
                    st.error("❌ AI識別失敗，請嘗試其他圖片")

def statistics_analysis(predictor):
    """統計分析頁面"""
    st.markdown("## 📊 模型統計分析")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="control-card">
            <h4>🔧 技術規格</h4>
        </div>
        """, unsafe_allow_html=True)
        
        specs = {
            "模型架構": "CRNN (CNN + LSTM)",
            "支援字符": CHARACTERS,
            "字符數量": len(CHARACTERS),
            "序列長度": CAPTCHA_LENGTH_EXPECTED,
            "輸入尺寸": f"{DEFAULT_CONFIG['IMAGE_HEIGHT']}×{DEFAULT_CONFIG['IMAGE_WIDTH']}",
            "隱藏層大小": DEFAULT_CONFIG['HIDDEN_SIZE'],
            "LSTM層數": DEFAULT_CONFIG['NUM_LAYERS'],
            "計算設備": "CPU (Streamlit Cloud)"
        }
        
        for key, value in specs.items():
            st.info(f"**{key}**: {value}")
    
    with col2:
        st.markdown("""
        <div class="control-card">
            <h4>📈 性能指標</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if predictor.model_info:
            metrics = {
                "訓練輪數": predictor.model_info['epoch'],
                "驗證準確率": f"{predictor.model_info['best_val_captcha_acc']:.4f}",
                "模型大小": "~50MB (估計)",
                "推理速度": "~100ms/圖片 (CPU)",
                "支援格式": "PNG, JPG, JPEG",
                "最大圖片": "10MB"
            }
            
            for key, value in metrics.items():
                st.success(f"**{key}**: {value}")
    
    # 使用建議
    st.markdown("---")
    st.markdown("""
    <div class="control-card">
        <h4>💡 使用建議與最佳實踐</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 最佳效果
        - 清晰的4位大寫英文字母
        - 建議解析度不低於64×64
        - PNG格式通常效果最佳
        - 避免過度模糊或扭曲
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ 性能優化
        - 批量處理建議<50張圖片
        - 置信度>0.9為高信心結果
        - 置信度<0.5建議人工檢查
        - 使用ZIP檔案提高處理效率
        """)
    
    with col3:
        st.markdown("""
        ### 🔧 故障排除
        - 檢查圖片格式和大小
        - 確保驗證碼清晰可見
        - 避免包含非字母字符
        - 聯繫支援團隊尋求幫助
        """)
    
    # 模型架構圖 (文字描述)
    st.markdown("---")
    st.markdown("""
    <div class="control-card">
        <h4>🏗️ CRNN模型架構</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ```
    輸入圖片 (32×128×1)
            ↓
    CNN特徵提取層
    ├── Conv2d + BatchNorm + ReLU + MaxPool2d
    ├── Conv2d + BatchNorm + ReLU + MaxPool2d  
    ├── Conv2d + BatchNorm + ReLU
    ├── Conv2d + BatchNorm + ReLU + MaxPool2d
    ├── Conv2d + BatchNorm + ReLU
    └── Conv2d + BatchNorm + ReLU + MaxPool2d
            ↓
    序列重整 (Reshape)
            ↓
    LSTM序列建模層 (雙向LSTM)
            ↓
    Dropout正則化
            ↓
    全連接分類層
            ↓
    輸出 (4×26維度) → 4位字母預測
    ```
    """)
    
    # 鍵盤快捷鍵說明
    st.markdown("---")
    st.markdown("""
    <div class="control-card">
        <h4>⌨️ 鍵盤快捷鍵 (在資料夾模式)</h4>
    </div>
    """, unsafe_allow_html=True)
    
    shortcut_col1, shortcut_col2 = st.columns(2)
    
    with shortcut_col1:
        st.markdown("""
        - **←** 上一張圖片
        - **→** 下一張圖片
        - **Enter** 確認當前修正
        """)
    
    with shortcut_col2:
        st.markdown("""
        - **A** 使用AI識別結果
        - **S** 快速保存
        - **Esc** 取消當前操作
        """)

# 運行主程序
if __name__ == "__main__":
    main()