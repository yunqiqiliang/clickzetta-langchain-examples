import os, sys, tempfile, streamlit as st, uuid
from datetime import datetime

# Add parent directory to path for importing components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_clickzetta import (
    ClickZettaEngine,
    ClickZettaVectorStore,
    ClickZettaChatMessageHistory
)
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage

# Import common components
from components.common import UIComponents
from config.clickzetta_config import load_app_config

# 应用配置
app_config = load_app_config("qa")

# Streamlit app configuration
st.set_page_config(
    page_title="ClickZetta Intelligent Q&A",
    page_icon="🤖",
    layout="wide"
)

st.title('🤖 ClickZetta 智能问答系统')
st.markdown("*基于 ClickZetta 的企业级文档问答系统，支持多轮对话和记忆功能*")

# Render environment configuration status
env_config, env_file_exists, clickzetta_configured, dashscope_configured = UIComponents.render_env_config_status()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'loaded_doc' not in st.session_state:
    st.session_state.loaded_doc = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'engine' not in st.session_state:
    st.session_state.engine = None

# Sidebar configuration
with st.sidebar:
    st.header("🔧 系统配置")

    # Render environment configuration in sidebar
    UIComponents.render_env_config_sidebar(env_config, env_file_exists)

    # ClickZetta Configuration (with environment defaults)
    with st.expander("ClickZetta 连接设置", expanded=not clickzetta_configured):
        clickzetta_service = st.text_input("Service", value=env_config['clickzetta_service'], help="ClickZetta 服务地址")
        clickzetta_instance = st.text_input("Instance", value=env_config['clickzetta_instance'], help="实例名称")
        clickzetta_workspace = st.text_input("Workspace", value=env_config['clickzetta_workspace'], help="工作空间")
        clickzetta_schema = st.text_input("Schema", value=env_config['clickzetta_schema'], help="模式名称")
        clickzetta_username = st.text_input("Username", value=env_config['clickzetta_username'], help="用户名")
        clickzetta_password = st.text_input("Password", value=env_config['clickzetta_password'], type="password", help="密码")
        clickzetta_vcluster = st.text_input("VCluster", value=env_config['clickzetta_vcluster'], help="虚拟集群")

    # AI Model Configuration
    with st.expander("DashScope 模型设置", expanded=not dashscope_configured):
        api_key = st.text_input("DashScope API Key", value=env_config['dashscope_api_key'], type="password")
        embedding_model_options = ["text-embedding-v4", "text-embedding-v3"]
        embedding_model_index = 0
        if env_config['embedding_model'] in embedding_model_options:
            embedding_model_index = embedding_model_options.index(env_config['embedding_model'])
        embedding_model = st.selectbox("嵌入模型", embedding_model_options, index=embedding_model_index)

        llm_model_options = ["qwen-plus", "qwen-turbo", "qwen-max"]
        llm_model_index = 0
        if env_config['llm_model'] in llm_model_options:
            llm_model_index = llm_model_options.index(env_config['llm_model'])
        llm_model = st.selectbox("问答模型", llm_model_options, index=llm_model_index)

    # QA Configuration
    with st.expander("问答设置"):
        search_k = st.slider("检索数量", 1, 10, 5, help="检索相关文档的数量")
        temperature = st.slider("创造性", 0.0, 1.0, 0.1, help="回答的创造性程度")
        memory_window = st.slider("记忆窗口", 3, 20, 10, help="保持的对话轮数")

    # Session Management
    with st.expander("会话管理"):
        if st.button("🔄 新建会话"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.session_state.memory = None
            st.rerun()

        if st.button("🗑️ 清空历史"):
            st.session_state.chat_history = []
            st.session_state.memory = None
            st.rerun()

        st.caption(f"当前会话ID: {st.session_state.session_id[:8]}...")

    # Document Upload
    st.header("📄 文档管理")
    source_doc = st.file_uploader(
        "上传知识库文档",
        type=["pdf"],
        help="上传PDF文档作为问答知识库"
    )

    # Connection Status
    st.header("📊 连接状态")
    clickzetta_configured = all([
        clickzetta_service, clickzetta_instance, clickzetta_workspace,
        clickzetta_schema, clickzetta_username, clickzetta_password, clickzetta_vcluster
    ])

    if clickzetta_configured:
        st.success("✅ ClickZetta 已配置")
    else:
        st.warning("⚠️ 请完成 ClickZetta 配置")

    if api_key:
        st.success("✅ DashScope API 已配置")
    else:
        st.warning("⚠️ 请配置 DashScope API Key")

    if source_doc:
        st.success(f"✅ 文档已上传: {source_doc.name}")
    else:
        st.info("📎 请上传知识库文档")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 智能问答")

    # Initialize ClickZetta connection if configured
    if clickzetta_configured and api_key and not st.session_state.engine:
        try:
            with st.spinner("🔗 正在连接 ClickZetta..."):
                engine = ClickZettaEngine(
                    service=clickzetta_service,
                    instance=clickzetta_instance,
                    workspace=clickzetta_workspace,
                    schema=clickzetta_schema,
                    username=clickzetta_username,
                    password=clickzetta_password,
                    vcluster=clickzetta_vcluster,
                    connection_timeout=60,
                    query_timeout=1800
                )

                # Test connection
                engine.execute_query("SELECT 1 as test")
                st.session_state.engine = engine
                st.success("✅ ClickZetta 连接成功")

                # Initialize chat memory
                chat_memory = ClickZettaChatMessageHistory(
                    engine=engine,
                    session_id=st.session_state.session_id,
                    table_name=app_config.get_chat_table_name("qa")
                )

                # 使用新的内存管理方式 (避免弃用警告)
                st.session_state.chat_memory = chat_memory
                st.session_state.memory_window = memory_window

        except Exception as e:
            st.error(f"❌ ClickZetta 连接失败: {e}")

    # Document processing
    if source_doc and st.session_state.engine and st.session_state.loaded_doc != source_doc:
        try:
            with st.spinner("🔄 正在处理文档..."):
                # Load document
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(source_doc.read())
                loader = PyPDFLoader(tmp_file.name)
                pages = loader.load_and_split()
                os.remove(tmp_file.name)

                # Initialize embeddings
                embeddings = DashScopeEmbeddings(
                    dashscope_api_key=api_key,
                    model=embedding_model
                )

                # Create vector store
                vectorstore = ClickZettaVectorStore(
                    engine=st.session_state.engine,
                    embeddings=embeddings,
                    table_name=app_config.get_vector_table_name("qa"),
                    distance_metric="cosine"
                )

                # Add documents
                vectorstore.add_documents(pages)
                st.session_state.retriever = vectorstore.as_retriever(
                    search_kwargs={"k": search_k}
                )
                st.session_state.loaded_doc = source_doc

                st.success(f"✅ 文档处理完成: {len(pages)} 页")

        except Exception as e:
            st.error(f"❌ 文档处理失败: {e}")

    # Chat interface
    if st.session_state.retriever and hasattr(st.session_state, 'chat_memory'):
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📜 对话历史")
            for i, (role, message, timestamp) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.chat_message("user").write(f"**{timestamp}**\n{message}")
                else:
                    st.chat_message("assistant").write(f"**{timestamp}**\n{message}")

        # Query input
        query = st.chat_input("💭 请输入您的问题...")

        if query:
            # Add user message to history
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append(("user", query, timestamp))

            # Display user message
            st.chat_message("user").write(f"**{timestamp}**\n{query}")

            try:
                with st.spinner("🤔 正在思考..."):
                    # Initialize LLM
                    llm = Tongyi(
                        dashscope_api_key=api_key,
                        model_name=llm_model,
                        temperature=temperature
                    )

                    # Create QA chain (现代方式，避免弃用警告)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.retriever,
                        verbose=True
                    )

                    # Get response (使用 invoke 而不是 run)
                    result = qa_chain.invoke({"query": query})
                    response = result.get("result", str(result))

                    # Add AI response to history
                    response_timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.chat_history.append(("assistant", response, response_timestamp))

                    # Display AI response
                    st.chat_message("assistant").write(f"**{response_timestamp}**\n{response}")

            except Exception as e:
                st.error(f"❌ 问答处理失败: {e}")
    else:
        st.info("📋 请先完成配置并上传文档，然后开始问答")

with col2:
    st.subheader("📈 实时统计")

    if st.session_state.chat_history:
        # Chat statistics
        total_questions = len([msg for msg in st.session_state.chat_history if msg[0] == "user"])
        total_responses = len([msg for msg in st.session_state.chat_history if msg[0] == "assistant"])

        st.metric("❓ 总问题数", total_questions)
        st.metric("💬 总回答数", total_responses)
        st.metric("🔄 对话轮数", min(total_questions, total_responses))

    # System info
    st.subheader("🔧 系统信息")
    if st.session_state.engine:
        st.success("🟢 ClickZetta 已连接")
    else:
        st.error("🔴 ClickZetta 未连接")

    if st.session_state.retriever:
        st.success("🟢 知识库已加载")
    else:
        st.error("🔴 知识库未加载")

    if st.session_state.memory:
        st.success("🟢 记忆系统已启用")
    else:
        st.error("🔴 记忆系统未启用")

    # Advanced features
    st.subheader("🚀 高级功能")

    if st.button("📊 查看检索详情", disabled=not st.session_state.retriever):
        if st.session_state.retriever and st.session_state.chat_history:
            # Get last user query
            last_query = None
            for role, message, _ in reversed(st.session_state.chat_history):
                if role == "user":
                    last_query = message
                    break

            if last_query:
                with st.spinner("🔍 检索相关文档..."):
                    docs = st.session_state.retriever.get_relevant_documents(last_query)

                    st.write(f"**检索到 {len(docs)} 个相关文档片段:**")
                    for i, doc in enumerate(docs):
                        with st.expander(f"文档片段 {i+1}"):
                            st.write(doc.page_content[:300] + "...")
                            if doc.metadata:
                                st.json(doc.metadata)

    if st.button("💾 导出对话历史", disabled=not st.session_state.chat_history):
        if st.session_state.chat_history:
            # Create export data
            export_data = []
            for role, message, timestamp in st.session_state.chat_history:
                export_data.append({
                    "timestamp": timestamp,
                    "role": role,
                    "message": message
                })

            import json
            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

            st.download_button(
                label="📥 下载 JSON 格式",
                data=json_str,
                file_name=f"chat_history_{st.session_state.session_id[:8]}.json",
                mime="application/json"
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>🚀 Powered by <strong>ClickZetta</strong> + <strong>LangChain</strong></p>
        <p>💡 支持多轮对话、记忆功能和企业级部署</p>
    </div>
    """,
    unsafe_allow_html=True
)