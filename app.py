import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI

# 页面配置
st.set_page_config(
    page_title="个人知识库助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 标题
st.title("🤖 个人知识库助手 (My Knowledge Bot)")

# 左侧边栏
with st.sidebar:
    st.header("📄 文档上传")
    
    # 文件上传器
    uploaded_file = st.file_uploader(
        "上传 PDF 文档",
        type=["pdf"],
        help="支持上传 PDF 格式的文档"
    )
    
    # 提示文字
    st.caption("请上传你的文档，我会基于它回答问题。")
    
    # 如果上传了文件，显示文件信息
    if uploaded_file is not None:
        st.success(f"✅ 已上传: {uploaded_file.name}")
        st.info(f"文件大小: {uploaded_file.size / 1024:.2f} KB")

# 主区域
st.divider()

# 从 Streamlit Secrets 读取 API Key
try:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
except KeyError:
    api_key = None
    st.error("⚠️ 管理员未配置密钥")

# PDF 解析逻辑
if uploaded_file is not None:
    try:
        # 读取 PDF 文件
        pdf_reader = PdfReader(uploaded_file)
        
        # 提取所有文本内容
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # 将文本保存到 session_state，供后续使用
        st.session_state.document_text = text
        
        # 显示成功提示
        st.success(f"✅ 成功读取文档！共检测到 {len(text)} 个字符。")
        
        # 显示文档内容预览（前 1000 个字符）
        with st.expander("查看文档内容"):
            preview_text = text[:1000] if len(text) > 1000 else text
            st.text(preview_text)
            if len(text) > 1000:
                st.caption(f"（仅显示前 1000 个字符，文档共 {len(text)} 个字符）")
    
    except Exception as e:
        st.error(f"❌ 读取 PDF 文件时出错: {str(e)}")

# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 聊天输入框
user_question = st.chat_input("向文档提问...")

# RAG 问答逻辑
if user_question:
    # 检查 API Key
    if not api_key:
        st.warning("⚠️ 管理员未配置密钥")
    # 检查文档内容是否存在
    elif "document_text" not in st.session_state or not st.session_state.document_text:
        st.warning("⚠️ 请先上传并解析 PDF 文档")
    else:
        try:
            # 保存用户问题到历史记录
            st.session_state.messages.append({"role": "user", "content": user_question})
            
            # 显示用户问题
            with st.chat_message("user"):
                st.write(user_question)
            
            # 初始化 OpenAI 客户端（适配 DeepSeek）
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
            # 构建 RAG Prompt（每次都要带上 PDF 内容作为背景知识）
            text = st.session_state.document_text
            prompt = f"你是一个智能助手。请基于以下文档内容回答用户问题。\n\n文档内容：{text}\n\n用户问题：{user_question}"
            
            # 调用 DeepSeek API
            with st.chat_message("assistant"):
                with st.spinner("🤔 正在思考中..."):
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                
                # 获取 AI 回答
                ai_answer = response.choices[0].message.content
                
                # 显示 AI 回答
                st.write(ai_answer)
            
            # 保存 AI 回答到历史记录
            st.session_state.messages.append({"role": "assistant", "content": ai_answer})
            
        except Exception as e:
            st.error(f"❌ 调用 API 时出错: {str(e)}")

