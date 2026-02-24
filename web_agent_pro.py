import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="完全体老王 (双脑驱动)", page_icon="🧠")
st.title("🧠 完全体老王 (公网 + 私有知识库)")

# 初始化核心引擎 (大模型 + 搜索引擎)
llm = ChatOpenAI(
    api_key=st.secrets["API_KEY"], 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-max"
)
tavily_client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# 🧠 初始化老王的永久记忆支架
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'learned_files' not in st.session_state:
    st.session_state.learned_files = []

# 动态读取并转换 PDF 的核心函数
@st.cache_resource(show_spinner=False)
def process_new_pdf(file_bytes, file_name):
    with open("temp_upload.pdf", "wb") as f:
        f.write(file_bytes)
        
    loader = PyPDFLoader("temp_upload.pdf") 
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    splits = text_splitter.split_documents(docs)
    
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=st.secrets["API_KEY"], 
        model="text-embedding-v3" 
    )
    return FAISS.from_documents(splits, embeddings)

# ==========================================
# 🚨 终极进化：侧边栏与记忆融合操作台
# ==========================================
with st.sidebar:
    st.header("📂 老王的永久记忆库")
    uploaded_file = st.file_uploader("上传《软件设计师》新资料", type=["pdf"])
    
    if st.button("🧠 开始融合学习") and uploaded_file is not None:
        if uploaded_file.name in st.session_state.learned_files:
            st.warning(f"这份《{uploaded_file.name}》老王已经倒背如流啦！")
        else:
            with st.spinner(f"正在将《{uploaded_file.name}》融入大脑..."):
                try:
                    new_db = process_new_pdf(uploaded_file.getvalue(), uploaded_file.name)
                    
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = new_db
                    else:
                        st.session_state.vectorstore.merge_from(new_db)
                    
                    st.session_state.learned_files.append(uploaded_file.name)
                    st.success(f"✅ 成功融合！目前已掌握 {len(st.session_state.learned_files)} 份资料。")
                except Exception as e:
                    st.error(f"❌ 抓到真凶了！真实报错是：{str(e)}")

    if st.session_state.learned_files:
        st.write("---")
        st.write("📚 目前已掌握的知识：")
        for f_name in st.session_state.learned_files:
            st.caption(f"• {f_name}")

# 下面保留你的 @tool 技能代码和聊天界面代码，不需要动！

# ==========================================
# 🛠️ 技能 1：公网搜索 (保持不变)
# ==========================================
@tool
def web_search(query: str) -> str:
    """当需要查询实时信息、新闻、不知道的客观知识时，调用此工具全网搜索。"""
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=2)
        return "\n\n".join([f"标题: {res['title']}\n内容: {res['content']}" for res in response['results']])
    except Exception as e:
        return f"搜索失败：{str(e)}"

# ==========================================
# 🛠️ 技能 2：私有知识库搜索 (破除线程壁垒版)
# ==========================================
# 🚨 核心魔法：在主线程里先把脑子拿出来，放进一个普通变量里，让子线程也能摸得到
GLOBAL_BRAIN = st.session_state.get('vectorstore', None)

@tool
def search_internal_doc(query: str) -> str:
    """当用户询问关于上传的PDF文件、内部知识、复习资料时，调用此工具。"""
    if GLOBAL_BRAIN is None:
        return "请礼貌地告诉用户：老王目前脑子里空空如也，请先上传 PDF 资料！"
    
    retriever = GLOBAL_BRAIN.as_retriever()
    results = retriever.invoke(query)
    return "\n\n".join([res.page_content for res in results])

# 将两个技能装进大脑
agent_executor = create_react_agent(llm, [web_search, search_internal_doc])


# ---------------- 下面是网页界面的常规逻辑 ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个幽默的全能AI助理老王。"}]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if user_input := st.chat_input("试试连招：今天的微博热搜是什么？那软件设计师的口诀呢？"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("老王正在左右脑同时运转..."):
            response = agent_executor.invoke({"messages": st.session_state.messages})
            ai_reply = response["messages"][-1].content
            st.markdown(ai_reply)
            
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})