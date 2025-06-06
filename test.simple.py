#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""簡化版本 - 測試基本功能"""

import streamlit as st
import os
from pathlib import Path

# 頁面配置
st.set_page_config(
    page_title="🎯 測試應用",
    page_icon="🎯",
    layout="wide"
)

# 簡化的CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        background: #f8fafc;
    }
    
    .test-card {
        background: #60a5fa;
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """主應用程序"""
    st.markdown("""
    <div class="test-card">
        <h1>🎯 測試應用</h1>
        <p>如果您看到這個，說明基本功能正常</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("✅ Streamlit 基本功能正常！")
    
    # 檢查文件
    st.markdown("## 📁 檔案檢查")
    
    current_dir = Path(".")
    files = list(current_dir.glob("*.pth"))
    
    if files:
        st.success(f"找到 {len(files)} 個模型檔案:")
        for file in files:
            st.write(f"📦 {file.name}")
    else:
        st.warning("未找到 .pth 模型檔案")
    
    # 簡單互動
    if st.button("🎉 測試按鈕"):
        st.balloons()
        st.success("按鈕功能正常！")
    
    # 文字輸入測試
    test_input = st.text_input("輸入測試", placeholder="輸入任何文字")
    if test_input:
        st.write(f"您輸入了: {test_input}")
    
    # 側邊欄測試
    with st.sidebar:
        st.markdown("### 側邊欄測試")
        st.info("側邊欄功能正常")
        
        option = st.selectbox("選擇測試", ["選項1", "選項2", "選項3"])
        st.write(f"您選擇了: {option}")

if __name__ == "__main__":
    main()