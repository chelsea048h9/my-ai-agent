import streamlit as st
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader


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
# 👇 🚨 新增：准备一个大书包，用来装整本书的纯文本
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""

# 动态读取并转换 PDF 的核心函数
@st.cache_resource(show_spinner=False)
def process_new_document(file_bytes, file_name):
    # 提取文件的后缀名 (比如 .pdf, .txt)
    ext = os.path.splitext(file_name)[1].lower()
    
    # 动态生成临时文件名（保留原后缀）
    temp_file_path = f"temp_upload{ext}"
    with open(temp_file_path, "wb") as f:
        f.write(file_bytes)
        
    # 🚨 核心路由逻辑：根据不同格式调用不同的解析器
    if ext == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif ext in [".txt", ".md", ".csv"]:
        # 纯文本类的文件，用 TextLoader 通杀
        loader = TextLoader(temp_file_path, encoding='utf-8')
    else:
        raise ValueError(f"哎呀，老王还不认识 {ext} 这种格式的文件！")
        
    docs = loader.load()
    
    # 提取完整纯文本
    full_text = "\n".join([doc.page_content for doc in docs])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    splits = text_splitter.split_documents(docs)
    
    embeddings = DashScopeEmbeddings(
        dashscope_api_key=st.secrets["API_KEY"], 
        model="text-embedding-v3" 
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore, full_text

# ==========================================
# 🚨 终极进化：侧边栏与记忆融合操作台
# ==========================================
with st.sidebar:
    # 加在侧边栏的最下面
    st.write("---")
    st.header("⚙️ 团队偏好设置")
    need_translate = st.checkbox("🌐 召唤渡边 (将老王的回答翻译为纯正日文)", value=False)
    st.header("📂 老王的永久记忆库")
    
    # 🚨 核心修改 1：放宽格式限制，并开启 accept_multiple_files=True
    uploaded_files = st.file_uploader(
        "批量上传秘籍 (支持 PDF/TXT/MD)", 
        type=["pdf", "txt", "md"], 
        accept_multiple_files=True  # 魔法开关在这里！
    )
    
    if st.button("🧠 开始批量融合学习") and uploaded_files:
        # 🚨 核心修改 2：把传入的列表做个 for 循环，挨个吃掉
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.learned_files:
                st.warning(f"《{uploaded_file.name}》老王已经倒背如流啦，跳过！")
                continue # 学过的直接跳过，学下一本
                
            with st.spinner(f"正在将《{uploaded_file.name}》融入大脑..."):
                try:
                    # 调用刚才写好的全格式解析器
                    new_db, new_text = process_new_document(uploaded_file.getvalue(), uploaded_file.name)
                    
                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = new_db
                    else:
                        st.session_state.vectorstore.merge_from(new_db)
                    
                    st.session_state.raw_text += f"\n\n---《{uploaded_file.name}》---\n\n{new_text}"
                    st.session_state.learned_files.append(uploaded_file.name)
                    st.success(f"✅ 《{uploaded_file.name}》融合完毕！")
                except Exception as e:
                    st.error(f"❌ 融合《{uploaded_file.name}》时出错：{str(e)}")

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
# ==========================================
# 🛠️ 技能 3：全局文档分析 (破除线程壁垒版)
# ==========================================
# 🚨 核心魔法：提前把纯文本拿出来，供子线程随时取用
GLOBAL_RAW_TEXT = st.session_state.get('raw_text', "")

@tool
def analyze_whole_document(query: str) -> str:
    """当用户要求“总结全文”、“整理思维导图”、“提取大纲”等涉及宏观全局分析时，强制调用此工具。"""
    if not GLOBAL_RAW_TEXT:
        return "老王脑子里还没有完整的文档，请先上传 PDF 资料！"
    
    # 截取前 30000 个字符
    text_to_analyze = GLOBAL_RAW_TEXT[:30000]
    
    summary_llm = ChatOpenAI(
        api_key=st.secrets["API_KEY"], 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-max"
    )
    
    prompt = f"你是一个资深的系统架构师。请基于以下我提供的完整文档内容，完成用户的任务：\n\n用户任务：{query}\n\n文档核心内容：\n{text_to_analyze}"
    
    response = summary_llm.invoke(prompt)
    return response.content

# ==========================================
# 🏢 AI 创业公司：多 Agent 协作系统架构
# ==========================================


# 1. 定义公司的“共享黑板” (State)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    need_translate: bool  # 🚨 新增：记录老板是否需要翻译的旨意

# 2. 实例化一号员工：【研究员老王】 (他带着那三个技能工具干活)
researcher_agent = create_react_agent(llm, [web_search, search_internal_doc, analyze_whole_document])

def researcher_node(state: AgentState):
    """老王的工作流：接单 -> 用工具查资料 -> 整理出中文技术大纲"""
    result = researcher_agent.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}

# 3. 实例化二号员工：【日籍翻译官渡边】
def translator_node(state: AgentState):
    """渡边的工作流：拿到老王的中文大纲 -> 转化为纯正的日本 IT 职场报告"""
    laowang_report = state["messages"][-1].content
    
    sys_prompt = """你叫渡边，是一位在东京涩谷工作了10年的资深IT系统架构师。
    请接收下面这份来自中文研究员的技术报告，将其完美翻译并润色为【地道、专业的日文 IT 业务报告】。
    要求：
    1. 必须使用标准 N2/N1 级别的日文商务/IT 术语。
    2. 保持原有的思维导图或层级大纲格式，排版要极其清晰。
    3. 在开头用日文跟用户打个招呼（比如：お疲れ様です、渡辺です...）。"""
    
    response = llm.invoke([
        {"role": "system", "content": sys_prompt}, 
        {"role": "user", "content": f"请翻译这份报告：\n{laowang_report}"}
    ])
    return {"messages": [response]}
# 👇 🚨 新增：调度员函数
def route_after_research(state: AgentState):
    """根据老板的旨意，决定老王干完活后是直接交差，还是递交给渡边"""
    if state.get("need_translate", False):
        return "Translator"
    else:
        return END

# 4. 包工头排班：用 Graph 把员工连成流水线
workflow = StateGraph(AgentState)

workflow.add_node("Researcher", researcher_node)
workflow.add_node("Translator", translator_node)

workflow.add_edge(START, "Researcher")

# 👇 🚨 核心修改：把原来的 workflow.add_edge("Researcher", "Translator") 删掉，换成这行“条件连线”！
workflow.add_conditional_edges("Researcher", route_after_research, {"Translator": "Translator", END: END})

workflow.add_edge("Translator", END)

# 正式挂牌营业！
multi_agent_app = workflow.compile()

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
        # 智能提示语：根据开关状态显示谁在干活
        status_text = "老王正在查阅资料，渡边正在准备日文翻译..." if need_translate else "老王正在疯狂速读并总结..."
        
        with st.spinner(status_text):
            # 🚨 核心修改：把 need_translate 传进公司黑板！
            response = multi_agent_app.invoke({
                "messages": st.session_state.messages,
                "need_translate": need_translate 
            })
            ai_reply = response["messages"][-1].content
            st.markdown(ai_reply)
            
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})