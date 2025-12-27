# n

https://mwpdrygpgvq6uosuuahhnn.streamlit.app/


# 😃 FER2013 Facial Expression Recognition (BlurPool CNN + Streamlit)

本專案使用 **FER2013 人臉表情資料集**，以 **PyTorch** 實作一個加入 **BlurPool（Anti-Aliasing）** 的卷積神經網路（CNN），  
並將訓練完成的模型 **部署至 Streamlit**，提供可上傳圖片的即時臉部表情辨識網頁應用。

---

## 📌 專案特色

- ✅ 使用 FER2013 資料集（7 類表情）
- ✅ CNN 架構中加入 **BlurPool**，降低下採樣造成的混疊（aliasing）
- ✅ PyTorch 訓練、驗證、測試完整流程
- ✅ 支援 **Streamlit Web App** 圖片上傳即時推論
- ✅ 可部署至 **Streamlit Community Cloud**

---

## 🧠 表情分類類別（FER2013）

| 編號 | 類別 |
|----|------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |
| 6 | Surprise |

---

## 🏗️ 專案結構

```text
hw4/
├── streamlit_app.py          # Streamlit 推論主程式
├── requirements.txt          # Python 套件需求
├── README.md                 # 專案說明文件
└── export/
    ├── fer_cnn_blurpool.pth  # 訓練完成的模型權重
    └── class_names.json      # 類別名稱對應


## ⚙️ 模型架構簡介

輸入影像大小：48 × 48 × 3

卷積層：4 層 Conv2D + ReLU

下採樣方式：

BlurPool（平均模糊）

MaxPooling

全連接層：

FC (512) → Dropout → FC (7)

Loss Function：CrossEntropyLoss

Optimizer：SGD (momentum = 0.9)


## 📚 資料集來源

FER2013 Dataset
https://www.kaggle.com/datasets/msambare/fer2013


## 📖 延伸與改進方向

🔹 加入人臉偵測（OpenCV / Haar / MTCNN）

🔹 使用 TTA（Test-Time Augmentation）

🔹 模型壓縮（TorchScript / ONNX）

🔹 改用 ResNet / EfficientNet + Anti-Aliasing

