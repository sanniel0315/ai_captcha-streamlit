#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit + CRNN模型整合 - 自動驗證碼識別工具"""

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
import requests
from pathlib import Path
import pandas as pd
import time
from typing import List, Tuple, Dict, Optional

# 頁面配置
st.set_page_config(
    page_title="🎯 AI驗證碼識別工具",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS樣式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .ai-result {
        background: linear-gradient(135deg, #8e44ad, #9b59b6);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .success-result {
        background: linear-gradient(135deg, #27ae60, #229954);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e74c3c, #f39c12, #27ae60);
    }
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

CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}

# 工具類
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

# CRNN模型定義
class CRNN(nn.Module):
    """CRNN模型結構，與Flask版本完全一致"""
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

# 模型載入和預測類
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
                st.error(f"模型檔案不存在: {model_path}")
                return False

            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 獲取配置
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                self.config = DEFAULT_CONFIG.copy()

            # 確保配置完整
            for key, val in DEFAULT_CONFIG.items():
                self.config.setdefault(key, val)

            # 創建模型
            self.model = CRNN(
                img_height=self.config['IMAGE_HEIGHT'],
                img_width=self.config['IMAGE_WIDTH'],
                num_classes=self.config['NUM_CLASSES'],
                hidden_size=self.config['HIDDEN_SIZE'],
                num_layers=self.config['NUM_LAYERS']
            ).to(self.device)

            # 載入權重
            if 'model_state_dict' in checkpoint:
                sd_key = 'model_state_dict'
            elif 'state_dict' in checkpoint:
                sd_key = 'state_dict'
            else:
                st.error("找不到model_state_dict或state_dict")
                return False

            self.model.load_state_dict(checkpoint[sd_key])
            self.model.eval()

            # 創建transform
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

            return True

        except Exception as e:
            st.error(f"模型載入失敗: {e}")
            return False

    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """對單張圖片做預測"""
        if not self.is_loaded:
            return "", 0.0

        try:
            # 確保圖片為RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)

            _, width_cnn_output, _ = outputs.shape
            seq_len = self.config['SEQUENCE_LENGTH']

            # 處理輸出序列
            if width_cnn_output >= seq_len:
                start = (width_cnn_output - seq_len) // 2
                focused = outputs[:, start:start + seq_len, :]
            else:
                pad = seq_len - width_cnn_output
                focused = torch.cat([outputs, outputs[:, -1:, :].repeat(1, pad, 1)], dim=1)

            pred_indices = torch.argmax(focused, dim=2)[0]
            idx_to_char_map = self.model_info.get('idx_to_char', IDX_TO_CHAR)
            
            # 處理字典鍵的類型
            if isinstance(next(iter(idx_to_char_map.keys())), str):
                idx_to_char_map = {int(k): v for k, v in idx_to_char_map.items()}

            text = ''.join(idx_to_char_map.get(idx.item(), '?') for idx in pred_indices).upper()

            # 計算置信度
            probs = torch.softmax(focused, dim=2)
            max_probs = torch.max(probs, dim=2)[0]
            confidence = float(torch.mean(max_probs).item())

            return text, confidence

        except Exception as e:
            st.error(f"預測失敗: {e}")
            return "", 0.0

# 使用Streamlit的緩存裝飾器
@st.cache_resource
def load_crnn_model():
    """載入並緩存CRNN模型"""
    predictor = CRNNPredictor()
    
    # 檢查模型檔案
    model_files = ['best_crnn_captcha_model.pth', 'model.pth', 'crnn_model.pth']
    model_path = None
    
    for file in model_files:
        if os.path.exists(file):
            model_path = file
            break
    
    if model_path is None:
        st.error("未找到模型檔案。請確保模型檔案在專案根目錄中。")
        st.info("支援的檔案名: " + ", ".join(model_files))
        return None
    
    if predictor.load_model(model_path):
        st.success(f"✅ 模型載入成功: {model_path}")
        return predictor
    else:
        st.error("❌ 模型載入失敗")
        return None

def main():
    # 主標題
    st.markdown("""
    <div class="main-header">
        <h1>🎯 AI驗證碼識別工具</h1>
        <p>使用CRNN模型自動識別4位大寫英文字母驗證碼</p>
    </div>
    """, unsafe_allow_html=True)

    # 載入模型
    predictor = load_crnn_model()
    
    if predictor is None:
        st.stop()

    # 側邊欄 - 模型信息
    with st.sidebar:
        st.header("⚙️ 模型信息")
        
        if predictor.model_info:
            st.success("🤖 CRNN模型已就緒")
            st.info(f"📊 訓練輪數: {predictor.model_info['epoch']}")
            st.info(f"📈 準確率: {predictor.model_info['best_val_captcha_acc']:.4f}")
            st.info(f"🔤 支援字符: {CHARACTERS}")
            st.info(f"📏 序列長度: {CAPTCHA_LENGTH_EXPECTED}")
        
        st.header("📚 使用說明")
        st.markdown("""
        1. **單張識別**: 上傳單張驗證碼圖片
        2. **批量處理**: 上傳多張圖片批量識別
        3. **結果修正**: 可手動修正AI識別結果
        4. **數據下載**: 支援CSV格式結果下載
        """)

    # 主要功能標籤頁
    tab1, tab2, tab3 = st.tabs(["📷 單張識別", "📁 批量處理", "📊 統計信息"])

    with tab1:
        st.header("📷 單張圖片識別")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ 上傳圖片")
            uploaded_file = st.file_uploader(
                "選擇驗證碼圖片",
                type=['png', 'jpg', 'jpeg'],
                help="支援PNG、JPG、JPEG格式",
                key="single_upload"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="上傳的驗證碼", use_column_width=True)
                
                # 從檔名提取標籤
                original_label = SimpleCaptchaCorrector.extract_label_from_filename(uploaded_file.name)
                if original_label:
                    st.info(f"📝 檔名中的標籤: **{original_label}**")
        
        with col2:
            st.subheader("🎯 識別結果")
            
            if uploaded_file is not None:
                if st.button("🚀 開始AI識別", type="primary", key="single_predict"):
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
                        
                        # 結果修正區域
                        st.subheader("✏️ 結果修正")
                        corrected_text = st.text_input(
                            "如需修正，請輸入正確答案:",
                            value=predicted_text,
                            max_chars=4,
                            help="只能輸入4個大寫英文字母 (A-Z)",
                            key="single_correction"
                        ).upper()
                        
                        # 驗證輸入
                        is_valid = SimpleCaptchaCorrector.validate_label(corrected_text)
                        
                        if corrected_text:
                            if is_valid:
                                st.success(f"✅ 格式正確: {corrected_text}")
                            else:
                                st.error("❌ 請輸入4個大寫英文字母")
                        
                        # 確認按鈕
                        if st.button("💾 確認結果", disabled=not is_valid, key="single_confirm"):
                            st.markdown(f"""
                            <div class="success-result">
                                ✅ 已確認結果: <strong>{corrected_text}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 準確性評估
                            if original_label:
                                is_correct = (corrected_text == original_label)
                                ai_accurate = (predicted_text == original_label)
                                
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("🎯 最終準確性", "✅ 正確" if is_correct else "❌ 錯誤")
                                with col_b:
                                    st.metric("🤖 AI準確性", "✅ 正確" if ai_accurate else "❌ 錯誤")
                    else:
                        st.error("❌ AI識別失敗，請嘗試其他圖片")

    with tab2:
        st.header("📁 批量處理")
        
        uploaded_files = st.file_uploader(
            "選擇多張驗證碼圖片",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="一次可上傳多張圖片進行批量識別",
            key="batch_upload"
        )
        
        if uploaded_files:
            st.info(f"📊 已選擇 **{len(uploaded_files)}** 張圖片")
            
            # 預覽部分圖片
            if len(uploaded_files) <= 6:
                cols = st.columns(min(len(uploaded_files), 3))
                for i, file in enumerate(uploaded_files[:6]):
                    with cols[i % 3]:
                        image = Image.open(file)
                        st.image(image, caption=file.name, use_column_width=True)
            
            if st.button("🚀 開始批量識別", type="primary", key="batch_predict"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"正在處理: {file.name} ({i+1}/{len(uploaded_files)})")
                    
                    try:
                        image = Image.open(file)
                        predicted_text, confidence = predictor.predict(image)
                        original_label = SimpleCaptchaCorrector.extract_label_from_filename(file.name)
                        
                        # 判斷狀態
                        if confidence > 0.9:
                            status = "🟢 高信心"
                        elif confidence > 0.7:
                            status = "🟡 中等信心"
                        elif confidence > 0.5:
                            status = "🟠 低信心"
                        else:
                            status = "🔴 極低信心"
                        
                        # 準確性檢查
                        accuracy = ""
                        if original_label:
                            if predicted_text == original_label:
                                accuracy = "✅ 正確"
                            else:
                                accuracy = "❌ 錯誤"
                        
                        results.append({
                            "檔案名": file.name,
                            "原始標籤": original_label or "無",
                            "AI識別結果": predicted_text or "失敗",
                            "置信度": f"{confidence:.3f}",
                            "狀態": status,
                            "準確性": accuracy
                        })
                        
                    except Exception as e:
                        results.append({
                            "檔案名": file.name,
                            "原始標籤": "錯誤",
                            "AI識別結果": "處理失敗",
                            "置信度": "0.000",
                            "狀態": "🔴 失敗",
                            "準確性": "❌ 錯誤"
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                processing_time = time.time() - start_time
                status_text.success(f"✅ 批量處理完成！耗時: {processing_time:.2f} 秒")
                
                # 顯示結果
                st.subheader("📊 批量識別結果")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # 統計信息
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_files = len(results)
                    st.metric("總文件數", total_files)
                
                with col2:
                    high_confidence = len([r for r in results if float(r["置信度"]) > 0.9])
                    st.metric("高置信度", f"{high_confidence}/{total_files}")
                
                with col3:
                    if any(r["準確性"] for r in results if r["準確性"]):
                        accurate_count = len([r for r in results if r["準確性"] == "✅ 正確"])
                        total_with_labels = len([r for r in results if r["原始標籤"] != "無"])
                        accuracy_rate = (accurate_count / total_with_labels * 100) if total_with_labels > 0 else 0
                        st.metric("準確率", f"{accuracy_rate:.1f}%")
                    else:
                        st.metric("準確率", "無標籤數據")
                
                with col4:
                    avg_confidence = sum(float(r["置信度"]) for r in results) / len(results)
                    st.metric("平均置信度", f"{avg_confidence:.3f}")
                
                # 下載結果
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下載結果CSV",
                    data=csv,
                    file_name=f"captcha_results_{int(time.time())}.csv",
                    mime="text/csv"
                )

    with tab3:
        st.header("📊 模型統計信息")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔧 技術規格")
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
            st.subheader("📈 性能指標")
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
        
        st.subheader("💡 使用建議")
        st.markdown("""
        - **最佳效果**: 清晰的4位大寫英文字母驗證碼
        - **圖片質量**: 建議解析度不低於64×64像素
        - **格式支援**: PNG格式通常效果最佳
        - **批量處理**: 建議單次不超過50張圖片
        - **置信度**: >0.9為高信心結果，<0.5建議人工檢查
        """)

if __name__ == "__main__":
    main()