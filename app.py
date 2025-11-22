import streamlit as st
import os

# --- 1. 引入必要套件 ---
try:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    st.error(f"環境錯誤: {e}")
    st.stop()

# 設定頁面
st.set_page_config(page_title="Archie's RAG (Llama 3.3)", layout="wide")

class ManualGroqEngine:
    """
    手動版 Groq RAG 引擎。
    使用最新的 Llama 3.3 模型。
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API Key is required")
        
        os.environ["GROQ_API_KEY"] = api_key
        
        # 1. 初始化 Embedding (HuggingFace)
        with st.spinner("正在初始化向量模型..."):
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. 初始化 LLM (Groq)
        # 更新為 Llama 3.3 (70B Versatile) - 目前 Groq 最強模型
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            temperature=0
        )
        self.vector_store = None

    def ingest_text(self, text: str):
        # 切分文本
        docs = [Document(page_content=text, metadata={"source": "upload"})]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        if not splits:
            raise ValueError("文本太短")

        # 建立向量資料庫
        self.vector_store = FAISS.from_documents(splits, self.embeddings)

    def ask(self, query: str):
        if not self.vector_store:
            return {"answer": "請先建立索引", "context": []}
        
        # --- 手動 RAG 流程 ---
        
        # 1. 檢索
        docs = self.vector_store.similarity_search(query, k=3)
        
        # 2. 整理上下文
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 3. 組合 Prompt
        prompt = ChatPromptTemplate.from_template("""
        你是一個專業的 AI 助手。請"只"根據以下的上下文資訊來回答使用者的問題。
        如果上下文沒有答案，請直接說「我不知道」。

        [上下文開始]
        {context}
        [上下文結束]

        使用者問題：{input}
        """)
        
        # 4. 呼叫模型
        final_prompt = prompt.format_messages(context=context_text, input=query)
        response = self.llm.invoke(final_prompt)
        
        return {
            "answer": response.content,
            "context": docs
        }

# --- UI ---
def main():
    st.title("⚡ 極速 RAG 系統 (Llama 3.3)")
    st.caption("Model: Llama-3.3-70b-versatile on Groq")
    
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
                    engine = ManualGroqEngine(api_key)
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
            with st.spinner("Llama 3.3 正在思考..."):
                try:
                    result = st.session_state.engine.ask(prompt)
                    answer = result["answer"]
                    
                    st.chat_message("assistant").write(answer)
                    
                    if result["context"]:
                        with st.expander("參考來源"):
                            for doc in result["context"]:
                                st.info(doc.page_content[:200] + "...")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"生成失敗: {e}")
        else:
            st.error("請先建立索引")

if __name__ == "__main__":
    main()