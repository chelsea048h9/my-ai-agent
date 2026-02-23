import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="完全体老王 (双脑驱动)", page_icon="🧠")
st.title("🧠 完全体老王 (公网 + 私有知识库)")

# 1. 初始化
llm = ChatOpenAI(
    api_key=st.secrets["API_KEY"], 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-coder-plus"
)
tavily_client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# ==========================================
# 🚨 新增魔法：把知识库缓存在网页内存里！
# 使用 @st.cache_resource 防止每次聊天都重新读取文件
# ==========================================
@st.cache_resource
def load_knowledge_base():
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=st.secrets["API_KEY"], 
        model="text-embedding-v3" 
    )
    return FAISS.from_documents(splits, embeddings)

vectorstore = load_knowledge_base()

# ==========================================
# 🛠️ 技能 1：公网搜索
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
# 🛠️ 技能 2：私有知识库搜索 (RAG)
# ==========================================
@tool
def search_internal_doc(query: str) -> str:
    """当用户询问关于'软件设计师'考试口诀、李四老板的日语学习情况、或者绝密档案时，必须调用此工具查询内部知识库。"""
    retriever = vectorstore.as_retriever()
    results = retriever.invoke(query)
    return "\n\n".join([res.page_content for res in results])

# 将两个技能都装进大脑
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