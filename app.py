import streamlit as st
import os

# --- 1. 引入 Groq 與 HuggingFace ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --- 2. 基礎組件 ---
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 設定頁面
st.set_page_config(page_title="Archie's RAG (Groq Edition)", layout="wide")

class GroqRAGEngine:
    """
    基於 Groq (雲端極速推論) 的 RAG 引擎。
    可部署至 Streamlit Cloud。
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API Key is required")
        
        # 設定 Groq API Key
        os.environ["GROQ_API_KEY"] = api_key
        
        # 1. 初始化 Embedding
        # 使用 HuggingFace 的免費模型 (all-MiniLM-L6-v2)
        # 這會直接在 Streamlit Cloud 的 CPU 上執行，完全免費
        with st.spinner("正在載入向量模型 (首次執行需約 30 秒)..."):
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. 初始化 LLM (使用 Groq 的 Llama 3)
        self.llm = ChatGroq(
            model="llama3-8b-8192", # Groq 上的 Llama 3 模型
            temperature=0
        )
        self.chain = None

    def ingest_text(self, text: str):
        # 切分文本
        docs = [Document(page_content=text, metadata={"source": "upload"})]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        if not splits:
            raise ValueError("文本太短")

        # 建立向量庫
        vector_store = FAISS.from_documents(splits, self.embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # 建立 Chain
        prompt = ChatPromptTemplate.from_template("""
        你是一個專業的 AI 助手。請根據以下上下文回答問題：
        <context>
        {context}
        </context>
        問題：{input}
        """)

        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        self.chain = create_retrieval_chain(retriever, doc_chain)

    def ask(self, query: str):
        if not self.chain:
            return "請先建立索引"
        
        response = self.chain.invoke({"input": query})
        return {
            "answer": response["answer"],
            "context": response["context"]
        }

# --- UI ---
def main():
    st.title("⚡ 極速 RAG 系統 (Powered by Groq)")
    st.caption("使用 Llama 3 on Groq + HuggingFace Embeddings。完全免費且可部署。")
    
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        # 讓使用者輸入 Groq Key
        api_key = st.text_input("Groq API Key", type="password", help="請至 console.groq.com 免費申請")
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
            with st.spinner("Groq 正在極速思考..."):
                result = st.session_state.engine.ask(prompt)
                answer = result["answer"]
                st.chat_message("assistant").write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("請先建立索引")

if __name__ == "__main__":
    main()