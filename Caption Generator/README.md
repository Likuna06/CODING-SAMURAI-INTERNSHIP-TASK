# 🤖 AI Image Caption Generator — Project 6

A beautiful, AI-powered image captioning web app using Deep Learning concepts (CNN + RNN pipeline) via the Anthropic Claude API.

---

## 📁 File Structure

```
project6/
├── index.html   ← Main HTML structure
├── style.css    ← All styles (dark theme, animations, layout)
├── app.js       ← JavaScript logic (upload, API call, results)
└── README.md    ← This file
```

---

## 🚀 Setup & Run

### Step 1 — Get an API Key
1. Go to https://console.anthropic.com
2. Create an account and generate an API key

### Step 2 — Add Your API Key
Open `app.js` and replace line 3:
```js
const API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE";
```
with your actual key:
```js
const API_KEY = "sk-ant-api03-...";
```

### Step 3 — Open in Browser
Simply open `index.html` in your browser.

> ⚠️ **Note**: Direct browser API calls require the `anthropic-dangerous-direct-browser-access` header (already included). For a production app, route API calls through a backend server to keep your key secret.

---

## ✨ Features

- 📤 **Drag & Drop** image upload (PNG, JPG, WEBP)
- 🎨 **4 Caption Styles**: Descriptive, Poetic, Technical, Social Media
- 🏷️ **Auto Keyword Tags** extracted from the image
- 📋 **One-click Copy** for generated captions
- 🔄 **CNN → RNN Pipeline** visualization
- 💜 **Dark theme** with animated background grid

---

## 🧠 How It Works (Deep Learning Concepts)

| Stage | Component | Role |
|---|---|---|
| 1 | **CNN Encoder** | Extracts visual feature maps from the image |
| 2 | **Feature Vector** | Compact representation of image content |
| 3 | **RNN Decoder** | Generates word sequences from features |
| 4 | **Caption Output** | Natural language description |

In this project, Claude's vision model acts as the CNN+RNN pipeline — it encodes the image and decodes it into natural language, mimicking the classic Show and Tell architecture.

---

## 🛠️ Technologies Used

- **HTML5 / CSS3 / Vanilla JS** — No frameworks needed
- **Anthropic Claude API** (claude-sonnet-4) — Vision + NLP
- **Syne & DM Mono** — Google Fonts
- **Tabler Icons** — Icon library

---

## 📚 Reference

- Video: [Image Captioning with Deep Learning](https://youtu.be/fUSTbGrL1tc?si=MnwnP36mMgJhGdvf)
- Anthropic Docs: https://docs.anthropic.com
