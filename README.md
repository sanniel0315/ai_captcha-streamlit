# 🎯 AI驗證碼識別工具 (Streamlit版)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/yourusername/ai-captcha-streamlit/main/app.py)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一個基於CRNN深度學習模型的智能驗證碼識別工具，專門用於識別4位大寫英文字母驗證碼。採用現代化的Streamlit界面，支援單張識別和批量處理。

## ✨ 功能特點

- 🤖 **AI自動識別**: 使用CRNN (CNN+LSTM) 模型自動識別驗證碼
- 📷 **單張處理**: 支援單張圖片即時識別，實時顯示置信度
- 📁 **批量處理**: 支援多張圖片批量識別，提供詳細統計
- 📊 **置信度評估**: 提供識別結果的置信度分析和可視化
- ✏️ **結果修正**: 支援手動修正AI識別結果
- 📈 **統計分析**: 提供準確率、平均置信度等詳細統計
- 💾 **結果導出**: 支援CSV格式結果下載
- 📱 **響應式設計**: 完美支援手機、平板和電腦訪問
- 🎨 **美觀界面**: 現代化UI設計，深色主題

## 🚀 線上體驗

🌐 **[立即使用線上版本](https://share.streamlit.io/sanniel0315/ai-captcha-streamlit/main/app.py)**

點擊上方鏈接直接使用，無需安裝任何軟件！

## 📸 應用截圖

| 功能 | 截圖 |
|------|------|
| 主界面 | ![主界面](docs/images/main_interface.png) |
| 單張識別 | ![單張識別](docs/images/single_recognition.png) |
| 批量處理 | ![批量處理](docs/images/batch_processing.png) |
| 統計分析 | ![統計分析](docs/images/statistics.png) |

## 🛠️ 技術架構

- **前端框架**: [Streamlit](https://streamlit.io/) - 現代化Web應用框架
- **深度學習**: [PyTorch](https://pytorch.org/) - 深度學習框架
- **模型架構**: CRNN (Convolutional Recurrent Neural Network)
- **圖像處理**: PIL/Pillow - 圖像處理庫
- **數據處理**: Pandas - 數據分析庫
- **部署平台**: Streamlit Cloud - 免費雲端部署

## 📊 模型性能

| 指標 | 數值 |
|------|------|
| **準確率** | >95% (測試集) |
| **支援字符** | A-Z (26個大寫英文字母) |
| **序列長度** | 4位字符 |
| **推理速度** | ~100ms/圖片 (CPU) |
| **模型大小** | ~50MB |
| **支援格式** | PNG, JPG, JPEG |

## 📦 本地運行

### 環境要求

- Python 3.8 或更高版本
- 建議使用虛擬環境

### 快速開始

1. **克隆倉庫**
```bash
git clone https://github.com/sanniel0315/ai-captcha-streamlit.git
cd ai-captcha-streamlit
```

2. **創建虛擬環境** (推薦)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安裝依賴**
```bash
pip install -r requirements.txt
```

4. **準備模型**
- 將CRNN模型檔案放在專案根目錄
- 支援的檔名: `best_crnn_captcha_model.pth`, `model.pth`, `crnn_model.pth`

5. **啟動應用**
```bash
streamlit run app.py
```

6. **訪問應用**
- 瀏覽器自動打開 http://localhost:8501
- 或手動訪問上述地址

## 🔧 使用說明

### 單張識別

1. 選擇 **"📷 單張識別"** 標籤頁
2. 點擊上傳區域或拖拽圖片上傳
3. 支援格式: PNG, JPG, JPEG
4. 點擊 **"🚀 開始AI識別"** 按鈕
5. 查看識別結果和置信度
6. 如需要可在 **"結果修正"** 區域手動調整
7. 點擊 **"💾 確認結果"** 完成

### 批量處理

1. 選擇 **"📁 批量處理"** 標籤頁
2. 一次選擇多張驗證碼圖片
3. 系統會顯示預覽（最多6張）
4. 點擊 **"🚀 開始批量識別"** 
5. 等待處理完成，可查看實時進度
6. 查看詳細結果表格，包含：
   - 檔案名
   - 原始標籤（從檔名提取）
   - AI識別結果
   - 置信度
   - 狀態評估
   - 準確性分析
7. 點擊 **"📥 下載結果CSV"** 導出數據

### 統計信息

1. 選擇 **"📊 統計信息"** 標籤頁
2. 查看模型技術規格
3. 查看性能指標
4. 閱讀使用建議

## 📁 項目結構

```
ai-captcha-streamlit/
├── app.py                    # 主應用程序
├── requirements.txt          # Python依賴列表
├── README.md                # 本文件
├── LICENSE                  # 許可證文件
├── .gitignore               # Git忽略規則
├── .streamlit/              # Streamlit配置
│   └── config.toml         # 主題和服務器配置
├── models/                  # 模型文件夾
│   └── best_crnn_captcha_model.pth  # CRNN模型文件
├── docs/                    # 文檔和截圖
│   ├── images/             # 應用截圖
│   └── deployment.md       # 部署指南
├── sample_images/          # 示例圖片
│   ├── sample1.png
│   ├── sample2.png
│   └── sample3.png
└── tests/                  # 測試文件 (可選)
    └── test_model.py
```

## 🚀 部署到Streamlit Cloud

### 方法一：一鍵部署 (推薦)

1. **Fork本倉庫** 到您的GitHub帳戶
2. **訪問** [Streamlit Cloud](https://share.streamlit.io/)
3. **使用GitHub帳戶登入**
4. **點擊 "New app"**
5. **填寫部署信息**:
   - Repository: `sannil/ai-captcha-streamlit`
   - Branch: `main`
   - Main file path: `app.py`
6. **點擊 "Deploy!"**
7. **等待部署完成** (約5-10分鐘)
8. **獲取永久URL** 並開始使用

### 方法二：手動部署

詳細步驟請參考 [部署指南](docs/deployment.md)

## ⚠️ 重要說明

### 模型文件處理

- **小於100MB**: 可直接上傳到GitHub
- **大於100MB**: 需要使用Git LFS或雲端存儲
- **模型格式**: PyTorch `.pth` 檔案
- **必需內容**: 完整的checkpoint，包含模型權重和配置

### 性能限制

- **Streamlit Cloud**: 1GB RAM限制
- **建議**: 單次批量處理不超過50張圖片
- **優化**: 大圖片會自動調整尺寸以節省記憶體

### 瀏覽器支援

- ✅ **推薦**: Chrome, Firefox, Safari, Edge
- ✅ **移動端**: iOS Safari, Android Chrome
- ⚠️ **需要**: JavaScript和現代CSS支援

## 📚 API參考

### 主要類別

#### `CRNNPredictor`
```python
class CRNNPredictor:
    def load_model(self, model_path: str) -> bool
    def predict(self, image: PIL.Image) -> Tuple[str, float]
```

#### `SimpleCaptchaCorrector`
```python
class SimpleCaptchaCorrector:
    @staticmethod
    def extract_label_from_filename(filename: str) -> str
    @staticmethod
    def validate_label(label: str) -> bool
```

### 配置參數

```python
DEFAULT_CONFIG = {
    'IMAGE_HEIGHT': 32,      # 輸入圖片高度
    'IMAGE_WIDTH': 128,      # 輸入圖片寬度
    'INPUT_CHANNELS': 1,     # 輸入通道數 (灰階)
    'SEQUENCE_LENGTH': 4,    # 序列長度
    'NUM_CLASSES': 26,       # 類別數 (A-Z)
    'HIDDEN_SIZE': 256,      # LSTM隱藏層大小
    'NUM_LAYERS': 2          # LSTM層數
}
```

## 🐛 故障排除

### 常見問題

**Q1: 模型載入失敗**
```
解決方案:
1. 檢查模型文件是否存在
2. 確認文件名正確
3. 檢查文件權限
4. 驗證模型格式完整性
```

**Q2: 記憶體不足錯誤**
```
解決方案:
1. 減少批量處理的圖片數量
2. 壓縮圖片尺寸
3. 重啟Streamlit應用
4. 使用本地運行代替雲端
```

**Q3: 部署失敗**
```
解決方案:
1. 檢查requirements.txt格式
2. 確認Python版本兼容性
3. 查看部署日誌錯誤信息
4. 檢查GitHub倉庫權限設置
```

**Q4: 識別準確率低**
```
可能原因:
1. 圖片質量不佳
2. 驗證碼格式不符合訓練數據
3. 圖片尺寸過小或過大
4. 模型文件損壞
```

### 調試模式

啟用調試模式查看詳細信息:
```bash
streamlit run app.py --logger.level=debug
```

## 🤝 貢獻指南

歡迎所有形式的貢獻！

### 貢獻方式

1. **報告問題**: 在 [Issues](https://github.com/sanniel03135/ai-captcha-streamlit/issues) 中報告Bug
2. **功能建議**: 提出新功能想法
3. **代碼貢獻**: 提交Pull Request
4. **文檔改進**: 完善說明文檔
5. **測試用例**: 添加測試覆蓋

### 開發流程

1. **Fork本項目**
2. **創建特性分支**: `git checkout -b feature/AmazingFeature`
3. **提交更改**: `git commit -m 'Add some AmazingFeature'`
4. **推送分支**: `git push origin feature/AmazingFeature`
5. **提交Pull Request**

### 代碼規範

- 遵循PEP 8 Python代碼風格
- 添加適當的註釋和文檔字符串
- 確保新功能有相應的測試
- 保持向後兼容性

## 📋 更新日誌

### v1.0.0 (2024-12-XX)
- ✨ 初始版本發布
- 🤖 集成CRNN深度學習模型
- 📷 實現單張圖片識別功能
- 📁 實現批量處理功能
- 📊 添加統計分析面板
- 💾 支援CSV結果導出
- 🎨 現代化UI設計
- 📱 響應式移動端支援

### 計劃功能 (v1.1.0)
- 🔄 支援更多驗證碼格式
- 🌐 多語言界面支援
- 📈 更詳細的統計圖表
- 🔐 用戶認證系統
- 💾 結果歷史記錄

## 📄 許可證

本項目採用 [MIT License](LICENSE) 許可證 - 查看 [LICENSE](LICENSE) 文件了解詳情。

### 使用條款

- ✅ 商業使用
- ✅ 修改
- ✅ 分發
- ✅ 私人使用
- ❌ 責任
- ❌ 保證

## 👨‍💻 作者信息

 Sanniel shi
- 🐙 GitHub: [@sanniel0315](https://github.com/sanniel0315)
- 📧 Email: sannielshi@gmail.com


## 🙏 致謝

特別感謝以下開源項目和貢獻者：

- [PyTorch](https://pytorch.org/) - 強大的深度學習框架
- [Streamlit](https://streamlit.io/) - 優秀的Web應用框架
- [Pillow](https://pillow.readthedocs.io/) - Python圖像處理庫
- [Pandas](https://pandas.pydata.org/) - 數據分析工具
- CRNN論文作者：[An End-to-End Trainable Neural OCR](https://arxiv.org/abs/1507.05717)

## 🔗 相關鏈接

- 📖 [Streamlit文檔](https://docs.streamlit.io/)
- 🔥 [PyTorch教程](https://pytorch.org/tutorials/)
- 📚 [CRNN論文](https://arxiv.org/abs/1507.05717)
- 🎯 [項目靈感來源](https://github.com/original-project)

## 📞 支援與反饋

如果您遇到問題或有建議，歡迎通過以下方式聯繫：

1. **GitHub Issues**: [提交問題](https://github.com/sanniel0315/ai-captcha-streamlit/issues)
2. **Email**: your.email@example.com
3. **討論區**: [GitHub Discussions](https://github.com/sannielshi/ai-captcha-streamlit/discussions)

---

## ⭐ 如果這個項目對您有幫助，請給一個Star！

[![GitHub stars](https://img.shields.io/github/stars/yourusername/ai-captcha-streamlit.svg?style=social)](https://github.com/yourusername/ai-captcha-streamlit/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/ai-captcha-streamlit.svg?style=social)](https://github.com/yourusername/ai-captcha-streamlit/network)

**享受AI驗證碼識別的便利吧！** 🚀