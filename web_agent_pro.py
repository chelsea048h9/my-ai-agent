import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

# 1. 网页配置
st.set_page_config(page_title="终极老王 (LangChain版)", page_icon="👑")
st.title("👑 终极老王 Web 聊天室 (大厂框架驱动)")

# 2. 初始化核心引擎 (大模型 + 搜索引擎)
llm = ChatOpenAI(
    api_key=st.secrets["API_KEY"], 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-coder-plus"
)
tavily_client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# ==========================================
# 🚨 终极魔法 1：极简工具定义
# 不需要写任何 JSON！直接用 @tool 装饰器！
# ==========================================
@tool
def web_search(query: str) -> str:
    """当需要查询实时信息、新闻、不知道的知识时，调用此工具全网搜索。"""
    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        return "\n\n".join([f"标题: {res['title']}\n内容: {res['content']}" for res in response['results']])
    except Exception as e:
        return f"搜索失败：{str(e)}"

# ==========================================
# 🚨 终极魔法 2：一行代码组装智能体！
# ==========================================
agent_executor = create_react_agent(llm, [web_search])

# 3. Streamlit 网页记忆初始化
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个幽默、全能的资深AI助理老王。你有完美的记忆力。"}
    ]

# 4. 渲染历史聊天气泡
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 5. 核心交互逻辑
if user_input := st.chat_input("跟注入了 LangChain 灵魂的老王聊聊吧！比如：今天A股收盘点数？"):
    
    # 显示用户的输入
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("老王正在通过 LangChain 引擎飞速思考并检索全网..."):
            
            # ==========================================
            # 🚨 终极魔法 3：告别繁琐的工具调用循环！
            # 直接把整个聊天记录扔给 agent_executor，
            # 它会自动帮你判断要不要用工具、自动调用、自动总结！
            # ==========================================
            response = agent_executor.invoke({"messages": st.session_state.messages})
            
            # 从 LangChain 的返回结果中，提取最后一句 AI 说的话
            ai_reply = response["messages"][-1].content
            
            # 显示在网页上
            st.markdown(ai_reply)
            
    # 把回答存入记忆
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})