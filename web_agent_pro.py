import json
import streamlit as st
from openai import OpenAI

# 1. 网页配置
st.set_page_config(page_title="全能老王", page_icon="🛠️")
st.title("🛠️ 全能老王的专属 Web 聊天室")

# 2. 初始化 API
# 修改后：从 Streamlit 云端保险柜读取密码
client = OpenAI(
    api_key=st.secrets["API_KEY"], # 👈 🚨 核心修改：让代码去云端保险柜找钥匙
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 3. 本地工具库（老王的手脚）
def get_weather(location):
    if "北京" in location: return "狂风暴雨，气温 10 度"
    elif "深圳" in location: return "阳光明媚，气温 28 度"
    else: return "未知天气"

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市真实天气",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }
}]

# 4. 初始化网页版记忆
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个幽默、全能的资深AI助理老王。你有完美的记忆力，必须严格根据我们之前的聊天记录回答问题。绝对不能说'每次对话都是独立的'这种废话！"}
    ]

# 5. 把之前的聊天记录画在网页上（过滤掉系统偷看的工具记录，保持界面清爽）
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 6. 处理你的最新输入
if user_input := st.chat_input("考考老王，比如：我是新老板李四，深圳今天天气咋样？"):
    
    # a. 显示并记录你的话
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # b. 老王开始处理
    with st.chat_message("assistant"):
        # 加上大厂同款的“转圈圈加载动画”
        with st.spinner("老王正在飞速思考中..."):
            response = client.chat.completions.create(
                model="qwen-coder-plus", 
                messages=st.session_state.messages,
                tools=tools
            )
            
            message = response.choices[0].message
            
            # 核心判断：老王需要用工具吗？
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                city = args.get("location")
                
                # 在网页上优雅地提示用户，AI 正在调用工具
                st.info(f"🔧 触发技能：老王正在调用本地代码，查询【{city}】的天气...")
                
                # 记录动作（包含之前修好的防失忆格式）
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
                
                # 执行本地代码
                weather_result = get_weather(city)
                
                # 记录结果
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": weather_result
                })
                
                # 拿着结果进行第二次呼叫
                final_response = client.chat.completions.create(
                    model="qwen-coder-plus",
                    messages=st.session_state.messages
                )
                ai_reply = final_response.choices[0].message.content
                
                # 画出最终回复并存入记忆
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                
            else:
                # 不需要工具，直接正常聊天
                ai_reply = message.content
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})