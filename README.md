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
```

## 🧪 Testing Guide (測試指南)

啟動 App 後，請依照以下步驟驗證 RAG 系統的準確度。

### 步驟 1：輸入測試資料

請複製以下 「模擬技術公告」，貼入 App 左側的 「文檔內容」 欄位，並點擊 「建立索引」。

【技術部公告】核心交易系統遷移計畫 (Project Phoenix)

1. 專案代號：Project Phoenix (鳳凰計畫)
2. 啟動日期：2025 年 3 月 15 日

3. 架構變更重點：
   (A) 資料庫遷移：
       - 從原本的 MySQL 5.7 全面遷移至 PostgreSQL 16。
       - 原因：為了支援更複雜的地理資訊查詢 (PostGIS)。
   
   (B) 後端語言升級：
       - 交易核心模組：將從 PHP Laravel 改寫為 Rust (以提升高併發效能)。
       - 報表模組：維持使用 Python，但升級至 3.12 版本。

4. 部署規則 (Deployment Policy)：
   - 正式環境 (Production) 禁止在週五下午 2:00 後進行任何部署。
   - 測試環境 (Staging) 則不受此限制，隨時可部署。

5. 緊急聯絡人：
   - 系統架構師 (Architect)：David Lin (工號 D-1024)
   - 資料庫管理員 (DBA)：Sarah Wu (工號 S-2048)


### 步驟 2：提問與驗證

請在右側對話框輸入以下問題，檢查 AI 回答是否正確。

測試類型

建議問題 (Prompt)

預期正確答案

精確查找

Project Phoenix 的系統架構師是誰？工號多少？

David Lin，工號 D-1024

邏輯判斷

今天是週五下午 4 點，我可以部署正式環境嗎？

不行。因為公告規定週五下午 2:00 後禁止部署。

技術細節

為什麼我們要遷移到 PostgreSQL？

為了支援地理資訊查詢 (PostGIS)。