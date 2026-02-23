import streamlit as st
import os  # 👈 新增：用于保存临时上传的文件
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
) # 👈 检查这里！是不是少了这个反括号？

tavily_client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"]) # 👈 还有这里，是不是拼写不完整？

# ==========================================
# 🚨 终极进化：网页侧边栏上传组件
# ==========================================
with st.sidebar:
    st.header("📂 老王的记忆插槽")
    uploaded_file = st.file_uploader("请喂给老王一份新的 PDF 秘籍", type=["pdf"])

# 动态读取并缓存上传的文件（把文件字节流传进来，只要传了新文件，就会自动刷新脑子）
@st.cache_resource(show_spinner=False)
def load_knowledge_base(file_bytes):
    with open("temp_upload.pdf", "wb") as f:
        f.write(file_bytes)
        
    # 🚨 魔法觉醒：开启 extract_images=True，老王就会自动调用 OCR 引擎去“看”图片里的字！
    loader = PyPDFLoader("temp_upload.pdf", extract_images=True) 
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)
    
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=st.secrets["API_KEY"], 
        model="text-embedding-v3" 
    )
    return FAISS.from_documents(splits, embeddings)

# 修改前面的 vectorstore 判断逻辑
vectorstore = None
if uploaded_file is not None:
    with st.spinner("老王正在疯狂速读 PDF..."):
        try:
            # 尝试提取真实文件数据喂给大模型
            vectorstore = load_knowledge_base(uploaded_file.getvalue())
            st.sidebar.success("✅ 秘籍吸收完毕！可随时提问。")
        except Exception as e:
            # 如果抓到报错（比如读不到文字导致 IndexError），就拦截下来并提示用户
            st.sidebar.error("❌ 哎呀，老王没法从这个 PDF 里提取出文字！它可能是个纯扫描件或者图片哦，请换一份能用鼠标复制文字的 PDF 试试！")
else:
    st.sidebar.info("👈 请先上传 PDF，否则老王的私有记忆库是空的哦！")

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
# 🛠️ 技能 2：私有知识库搜索 (增加判空逻辑)
# ==========================================
@tool
def search_internal_doc(query: str) -> str:
    """当用户询问关于上传的PDF文件、内部知识、复习资料时，调用此工具。"""
    if vectorstore is None:
        return "请礼貌地告诉用户：老王目前没有拿到 PDF 文件，请先在网页左侧上传！"
    
    retriever = vectorstore.as_retriever()
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