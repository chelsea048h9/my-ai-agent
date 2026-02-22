import json
import streamlit as st
from openai import OpenAI
from tavily import TavilyClient # 👈 换上正规军 Tavily！

st.set_page_config(page_title="全能老王 (满血联网版)", page_icon="🌐")
st.title("🌐 全能老王的专属 Web 聊天室 (Tavily 强力驱动)")

client = OpenAI(
    api_key=st.secrets["API_KEY"], 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 初始化专业的搜索客户端
tavily_client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# ==========================================
# 🚨 核心换血：大厂级别的搜索工具
# ==========================================
def web_search(query):
    try:
        # 使用 Tavily 专门为 AI 提供的搜索方法
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        # 提取真实网页内容给老王
        results = [f"标题: {res['title']}\n内容: {res['content']}" for res in response['results']]
        return "\n\n".join(results)
    except Exception as e:
        return f"搜索失败，网络小差：{str(e)}"
tools = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "当用户询问实时信息、新闻、不知道的知识时，必须调用此工具进行联网搜索。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "提取用户问题中的核心搜索关键词"
                }
            },
            "required": ["query"],
        },
    }
}]
# ==========================================

# 4. 初始化记忆
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个幽默、全能的资深AI助理老王。你可以使用 web_search 工具获取最新资讯。回答要自然，结合搜索结果给出答案。"}
    ]

# 5. 渲染历史记录
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 6. 处理最新输入
if user_input := st.chat_input("考考老王，比如：今天A股收盘是多少点？或者 今天的微博热搜是什么？"):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("老王正在飞速思考中..."):
            response = client.chat.completions.create(
                model="qwen-coder-plus", 
                messages=st.session_state.messages,
                tools=tools
            )
            
            message = response.choices[0].message
            
            # 判断是否需要联网
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                search_query = args.get("query")
                
                # 网页提示动画
                st.info(f"🌐 触发技能：老王正在全网搜索【{search_query}】...")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message.content or "", 
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                # 真正去互联网上查资料！
                search_result = web_search(search_query)
                
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": search_result
                })
                
                # 拿着搜到的真实网页数据，再次呼叫大模型
                final_response = client.chat.completions.create(
                    model="qwen-coder-plus",
                    messages=st.session_state.messages
                )
                ai_reply = final_response.choices[0].message.content
                
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            else:
                ai_reply = message.content
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})