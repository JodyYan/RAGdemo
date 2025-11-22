import streamlit as st
import os

# --- 1. 引入 Groq 與 HuggingFace ---
try:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    st.error("缺少必要套件，請檢查 requirements.txt")
    st.stop()

# --- 2. 基礎組件 ---
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# --- 3. 彈性引入 Chain (解決版本相容性) ---
try:
    # 優先嘗試新版 LCEL 語法
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    USE_LCEL = True
except ImportError:
    # 如果失敗，回退到舊版 QA Chain (這通常都存在)
    from langchain.chains import RetrievalQA
    USE_LCEL = False

# 設定頁面
st.set_page_config(page_title="Archie's RAG (Groq Edition)", layout="wide")

class GroqRAGEngine:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API Key is required")
        
        os.environ["GROQ_API_KEY"] = api_key
        
        # 初始化 Embedding
        with st.spinner("正在載入向量模型..."):
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 初始化 LLM
        self.llm = ChatGroq(
            model="llama3-8b-8192", 
            temperature=0
        )
        self.chain = None

    def ingest_text(self, text: str):
        docs = [Document(page_content=text, metadata={"source": "upload"})]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        if not splits:
            raise ValueError("文本太短")

        vector_store = FAISS.from_documents(splits, self.embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # 根據引入結果選擇不同的建立方式
        if USE_LCEL:
            prompt = ChatPromptTemplate.from_template("""
            你是一個專業的 AI 助手。請根據以下上下文回答問題：
            <context>
            {context}
            </context>
            問題：{input}
            """)
            doc_chain = create_stuff_documents_chain(self.llm, prompt)
            self.chain = create_retrieval_chain(retriever, doc_chain)
        else:
            # 舊版 Fallback 邏輯
            self.chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever
            )

    def ask(self, query: str):
        if not self.chain:
            return "請先建立索引"
        
        if USE_LCEL:
            response = self.chain.invoke({"input": query})
            return {"answer": response["answer"], "context": response.get("context", [])}
        else:
            # 舊版回傳格式不同
            response = self.chain.invoke({"query": query})
            return {"answer": response["result"], "context": response.get("source_documents", [])}

# --- UI ---
def main():
    st.title("⚡ 極速 RAG 系統 (Fixed Version)")
    st.caption("Deployment Fix: Version Pinning & Import Fallback")
    
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        api_key = st.text_input("Groq API Key", type="password")
        txt_input = st.text_area("文檔內容", height=250)
        
        if st.button("建立索引"):
            if api_key and txt_input:
                try:
                    engine = GroqRAGEngine(api_key)
                    engine.ingest_text(txt_input)
                    st.session_state.engine = engine
                    st.session_state.messages = []
                    st.success("索引完成！")
                except Exception as e:
                    st.error(f"錯誤: {e}")
            else:
                st.warning("請輸入 Key 與內容")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("請輸入問題..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        if st.session_state.engine:
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.engine.ask(prompt)
                    answer = result["answer"]
                    st.chat_message("assistant").write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"生成失敗: {e}")
        else:
            st.error("請先建立索引")

if __name__ == "__main__":
    main()