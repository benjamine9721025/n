# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json, os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 模型結構（需與訓練時完全一致）
# ----------------------------
def blur_pool(x, kernel_size=3):
    blur_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=x.device) / (kernel_size ** 2)
    blur_kernel = blur_kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, blur_kernel, stride=1, padding=kernel_size // 2, groups=x.shape[1])

class Net(nn.Module):
    def __init__(self, input_size=(48, 48), num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = blur_pool(x)
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = blur_pool(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3)
        x = self.fc2(x)
        return x

# ----------------------------
# 參數 & 路徑
# ----------------------------
MODEL_DIR = os.path.join(".", "export")
MODEL_PATH = os.path.join(MODEL_DIR, "fer_cnn_blurpool.pth")
CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

# 與訓練一致的前處理（⭐ 一致非常重要）
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ⭐ 與訓練端一致
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

@st.cache_resource
def load_model_and_classes():
    # 讀類別名稱
    if os.path.exists(CLASS_PATH):
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        # 萬一你沒放 class_names.json，就用 FER2013 常見的七類順序當後備
        class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    num_classes = len(class_names)

    # 載入模型
    device = torch.device("cpu")
    model = Net(num_classes=num_classes)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, class_names

def predict_image(model, img_pil, class_names):
    # img_pil: PIL.Image
    tensor = preprocess(img_pil).unsqueeze(0)  # [1,3,48,48]
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [C]
        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
    return pred_label, probs

def plot_probs(probs, labels):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    idx = np.arange(len(labels))
    ax.bar(idx, probs)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    fig.tight_layout()
    return fig

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="FER2013 表情辨識", page_icon="🙂", layout="centered")

st.title("😃 FER2013 臉部表情辨識（BlurPool CNN）")
st.caption("上傳單張人臉圖片，模型將預測其表情類別。請確保圖片中有明顯人臉。")

# 載入模型
if not os.path.exists(MODEL_PATH):
    st.error(f"找不到模型權重：{MODEL_PATH}。請先將訓練好的權重上傳到 export/。")
    st.stop()

model, class_names = load_model_and_classes()

uploaded = st.file_uploader("上傳圖片（jpg/png）", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="上傳影像預覽", use_container_width=True)

    if st.button("開始辨識"):
        with st.spinner("模型推論中..."):
            label, probs = predict_image(model, img, class_names)
        st.success(f"預測結果：**{label}**")

        fig = plot_probs(probs, class_names)
        st.pyplot(fig)

        # 顯示前 3 名
        topk = np.argsort(probs)[::-1][:3]
        st.subheader("Top-3 可信度")
        for i in topk:
            st.write(f"- {class_names[i]}：{probs[i]:.3f}")
else:
    st.info("請先上傳一張圖片。")
