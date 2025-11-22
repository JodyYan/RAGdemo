# ⚡ Groq RAG System (Streamlit Cloud Ready)

這是一個基於 **RAG (Retrieval-Augmented Generation)** 架構的極速問答系統。

本專案採用 **Groq** 作為推論引擎，利用其 LPU (Language Processing Unit) 技術實現了比傳統 GPU 快 10 倍以上的推論速度。同時結合 **HuggingFace** 的輕量級向量模型，讓整套系統無需 GPU 即可運行，並可直接部署至 **Streamlit Community Cloud**，擁有專屬的 `streamlit.app` 網址。

## 🚀 Key Features (核心功能)

-   **Blazing Fast**: 使用 **Groq (Llama 3)** 進行推論，幾秒內即可生成長篇回答。
-   **Cloud Deployable**: 完整支援部署至 Streamlit Cloud，輕鬆分享給他人使用。
-   **Free Tier**: 利用 Groq 目前提供的免費 API 額度，以及 HuggingFace 的開源 Embedding 模型，實現零成本運作。
-   **Privacy Aware**: 透過 RAG 技術，模型僅根據提供的私有文檔回答，減少幻覺。

## 🛠️ Tech Stack (技術棧)

-   **Language**: Python 3.9+
-   **Frontend**: Streamlit
-   **LLM Provider**: [Groq](https://groq.com/) (Model: Llama3-8b-8192)
-   **Embeddings**: [HuggingFace](https://huggingface.co/) (Model: all-MiniLM-L6-v2)
-   **Vector Store**: FAISS (CPU)
-   **Framework**: LangChain

## ⚙️ Setup & Installation (安裝與執行)

### 1. Pre-requisites (前置準備)
前往 [Groq Cloud Console](https://console.groq.com/keys) 申請免費的 API Key (`gsk_...`)。

### 2. Local Development (本機執行)

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 啟動應用程式
streamlit run app.py