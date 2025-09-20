import os, tempfile, streamlit as st, sys
from dotenv import load_dotenv
from langchain_clickzetta import ClickZettaEngine, ClickZettaVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader

# 添加父目录到路径以导入通用组件
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.clickzetta_config import load_app_config

# 应用配置
app_config = load_app_config("summary")

# Load environment variables from current and parent directory
load_dotenv()  # 当前目录
load_dotenv('../.env')  # 父目录

# Load configuration from environment variables
def load_env_config():
    return {
        'clickzetta_service': os.getenv('CLICKZETTA_SERVICE', ''),
        'clickzetta_instance': os.getenv('CLICKZETTA_INSTANCE', ''),
        'clickzetta_workspace': os.getenv('CLICKZETTA_WORKSPACE', ''),
        'clickzetta_schema': os.getenv('CLICKZETTA_SCHEMA', ''),
        'clickzetta_username': os.getenv('CLICKZETTA_USERNAME', ''),
        'clickzetta_password': os.getenv('CLICKZETTA_PASSWORD', ''),
        'clickzetta_vcluster': os.getenv('CLICKZETTA_VCLUSTER', ''),
        'dashscope_api_key': os.getenv('DASHSCOPE_API_KEY', ''),
        'embedding_model': os.getenv('DASHSCOPE_EMBEDDING_MODEL', 'text-embedding-v4'),
        'llm_model': os.getenv('DASHSCOPE_LLM_MODEL', 'qwen-plus'),
    }

# Streamlit app configuration
st.set_page_config(
    page_title="ClickZetta Document Summary",
    page_icon="📄",
    layout="wide"
)

st.title('📄 ClickZetta 文档智能摘要')
st.markdown("*基于 ClickZetta 向量存储的企业级文档摘要系统*")

# Configuration status banner
env_config = load_env_config()
clickzetta_configured = all([
    env_config['clickzetta_service'], env_config['clickzetta_instance'],
    env_config['clickzetta_workspace'], env_config['clickzetta_schema'],
    env_config['clickzetta_username'], env_config['clickzetta_password'],
    env_config['clickzetta_vcluster']
])
dashscope_configured = bool(env_config['dashscope_api_key'])

# Status banner
col1, col2, col3 = st.columns(3)
with col1:
    env_file_exists = os.path.exists('.env') or os.path.exists('../.env')
    if env_file_exists:
        st.success("✅ 配置文件已加载")
        # 显示实际找到的 .env 文件路径
        if os.path.exists('.env'):
            st.caption("📁 位置: ./.env")
        elif os.path.exists('../.env'):
            st.caption("📁 位置: ../.env")
    else:
        st.error("❌ 缺少 .env 配置文件")

with col2:
    if clickzetta_configured:
        st.success("✅ ClickZetta 连接配置完成")
    else:
        st.warning("⚠️ ClickZetta 配置不完整")

with col3:
    if dashscope_configured:
        st.success("✅ DashScope API 配置完成")
    else:
        st.warning("⚠️ DashScope API 未配置")

st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 配置设置")

    # Display environment configuration status
    st.subheader("📋 环境配置状态")

    # Show configuration source
    # 使用已定义的 env_file_exists 变量（检查当前目录和父目录）
    if env_file_exists:
        st.success("✅ 已加载 .env 配置文件")
        # 显示实际找到的 .env 文件路径
        if os.path.exists('.env'):
            st.caption("📁 位置: ./.env")
        elif os.path.exists('../.env'):
            st.caption("📁 位置: ../.env")
    else:
        st.warning("⚠️ 未找到 .env 配置文件")

    # ClickZetta Configuration
    st.subheader("ClickZetta 连接")

    # Show loaded values with option to override
    clickzetta_service = st.text_input(
        "Service",
        value=env_config['clickzetta_service'],
        help="ClickZetta 服务地址"
    )
    clickzetta_instance = st.text_input(
        "Instance",
        value=env_config['clickzetta_instance'],
        help="实例名称"
    )
    clickzetta_workspace = st.text_input(
        "Workspace",
        value=env_config['clickzetta_workspace'],
        help="工作空间"
    )
    clickzetta_schema = st.text_input(
        "Schema",
        value=env_config['clickzetta_schema'],
        help="模式名称"
    )
    clickzetta_username = st.text_input(
        "Username",
        value=env_config['clickzetta_username'],
        help="用户名"
    )
    clickzetta_password = st.text_input(
        "Password",
        value=env_config['clickzetta_password'],
        type="password",
        help="密码"
    )
    clickzetta_vcluster = st.text_input(
        "VCluster",
        value=env_config['clickzetta_vcluster'],
        help="虚拟集群"
    )

    # AI Model Configuration
    st.subheader("DashScope 模型设置")
    api_key = st.text_input(
        "DashScope API Key",
        value=env_config['dashscope_api_key'],
        type="password"
    )

    embedding_model_options = ["text-embedding-v4", "text-embedding-v3"]
    embedding_model_index = 0
    if env_config['embedding_model'] in embedding_model_options:
        embedding_model_index = embedding_model_options.index(env_config['embedding_model'])
    embedding_model = st.selectbox("嵌入模型", embedding_model_options, index=embedding_model_index)

    llm_model_options = ["qwen-plus", "qwen-turbo", "qwen-max"]
    llm_model_index = 0
    if env_config['llm_model'] in llm_model_options:
        llm_model_index = llm_model_options.index(env_config['llm_model'])
    llm_model = st.selectbox("语言模型", llm_model_options, index=llm_model_index)

    # Summary Configuration
    st.subheader("摘要设置")
    summary_language = st.selectbox("摘要语言", ["中文", "English", "自动检测"])
    summary_length = st.slider("摘要长度 (字数)", 100, 500, 200)
    summary_style = st.selectbox("摘要风格", ["简洁概述", "详细分析", "要点列表"])

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    source_doc = st.file_uploader(
        "📎 上传文档",
        type=["pdf"],
        help="支持 PDF 格式文档，建议文件大小不超过 10MB"
    )

    if source_doc:
        st.info(f"📋 已选择文件: {source_doc.name} ({source_doc.size / 1024 / 1024:.1f} MB)")

with col2:
    st.markdown("### 📊 系统状态")

    # Environment file status
    if env_file_exists:
        st.success("✅ .env 配置文件已加载")
    else:
        st.warning("⚠️ .env 配置文件未找到")

    # Connection status check
    clickzetta_configured = all([
        clickzetta_service, clickzetta_instance, clickzetta_workspace,
        clickzetta_schema, clickzetta_username, clickzetta_password, clickzetta_vcluster
    ])

    if clickzetta_configured:
        st.success("✅ ClickZetta 配置完成")
    else:
        st.warning("⚠️ 请完成 ClickZetta 配置")

    if api_key:
        st.success("✅ DashScope API 已配置")
        # Show masked API key
        masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else api_key[:4] + "*" * 4
        st.caption(f"API Key: {masked_key}")
    else:
        st.warning("⚠️ 请配置 DashScope API Key")

    # Configuration details
    with st.expander("📋 查看配置详情"):
        st.write("**ClickZetta 配置:**")
        st.write(f"• Service: `{clickzetta_service or '未配置'}`")
        st.write(f"• Instance: `{clickzetta_instance or '未配置'}`")
        st.write(f"• Workspace: `{clickzetta_workspace or '未配置'}`")
        st.write(f"• Schema: `{clickzetta_schema or '未配置'}`")
        st.write(f"• Username: `{clickzetta_username or '未配置'}`")
        st.write(f"• VCluster: `{clickzetta_vcluster or '未配置'}`")

        st.write("**DashScope 配置:**")
        st.write(f"• 嵌入模型: `{embedding_model}`")
        st.write(f"• 语言模型: `{llm_model}`")

# Summarize button
if st.button("🚀 开始摘要", type="primary", use_container_width=True):
    # Validation
    if not clickzetta_configured:
        st.error("❌ 请完成 ClickZetta 连接配置")
    elif not api_key.strip():
        st.error("❌ 请提供 API Key")
    elif not source_doc:
        st.error("❌ 请上传文档")
    else:
        try:
            with st.spinner('🔄 正在处理文档，请稍候...'):
                # Initialize ClickZetta Engine
                engine = ClickZettaEngine(
                    service=clickzetta_service,
                    instance=clickzetta_instance,
                    workspace=clickzetta_workspace,
                    schema=clickzetta_schema,
                    username=clickzetta_username,
                    password=clickzetta_password,
                    vcluster=clickzetta_vcluster
                )

                # Test connection
                try:
                    engine.execute_query("SELECT 1 as test")
                    st.success("✅ ClickZetta 连接成功")
                except Exception as e:
                    st.error(f"❌ ClickZetta 连接失败: {e}")
                    st.stop()

                # Save uploaded file temporarily and load documents
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(source_doc.read())
                loader = PyPDFLoader(tmp_file.name)
                pages = loader.load_and_split()
                os.remove(tmp_file.name)

                st.info(f"📄 文档已加载: {len(pages)} 页")

                # Initialize embeddings and LLM
                embeddings = DashScopeEmbeddings(
                    dashscope_api_key=api_key,
                    model=embedding_model
                )
                llm = Tongyi(
                    dashscope_api_key=api_key,
                    model_name=llm_model,
                    temperature=0.1
                )

                # Create ClickZetta vector store
                vectorstore = ClickZettaVectorStore(
                    engine=engine,
                    embeddings=embeddings,
                    table_name=app_config.get_vector_table_name("summary"),
                    distance_metric="cosine"
                )

                # Add documents to vector store
                vectorstore.add_documents(pages)
                st.success("✅ 文档向量化完成")

                # Prepare summary prompt based on settings
                if summary_language == "中文":
                    language_instruction = "请用中文"
                elif summary_language == "English":
                    language_instruction = "Please use English"
                else:
                    language_instruction = "请根据文档语言自动选择中文或英文"

                style_instructions = {
                    "简洁概述": "用简洁的语言概述文档主要内容",
                    "详细分析": "详细分析文档的核心观点、逻辑结构和关键信息",
                    "要点列表": "以要点列表的形式总结文档的关键信息"
                }

                # Perform similarity search to get relevant content
                query = "文档摘要 document summary"
                relevant_docs = vectorstore.similarity_search(query, k=min(len(pages), 10))

                # Create custom summarization prompt
                from langchain.prompts import PromptTemplate

                summary_prompt = PromptTemplate(
                    input_variables=["text"],
                    template=f"""
{language_instruction}{style_instructions[summary_style]}，
字数控制在{summary_length}字以内。

文档内容：
{{text}}

摘要：
"""
                )

                # Initialize summarization chain with custom prompt
                chain = load_summarize_chain(
                    llm,
                    chain_type="stuff",
                    prompt=summary_prompt
                )

                # Generate summary (使用 invoke 而不是 run)
                result = chain.invoke({"input_documents": relevant_docs})
                summary = result if isinstance(result, str) else result.get("output_text", str(result))

                # Display results
                st.markdown("## 📝 文档摘要")
                st.markdown("---")

                # Summary display with styling
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f0f2f6;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 5px solid #1f77b4;
                    ">
                        {summary}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Additional information
                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("📄 页数", len(pages))

                with col2:
                    st.metric("🎯 相关片段", len(relevant_docs))

                with col3:
                    st.metric("📊 摘要长度", f"{len(summary)} 字")

                # Technical details (expandable)
                with st.expander("🔧 技术详情"):
                    st.write(f"**向量存储表**: document_summary_vectors")
                    st.write(f"**嵌入模型**: {embedding_model}")
                    st.write(f"**语言模型**: {llm_model}")
                    st.write(f"**距离度量**: cosine")
                    st.write(f"**处理的文档片段**: {len(relevant_docs)} / {len(pages)}")

        except Exception as e:
            st.error(f"❌ 处理过程中发生错误: {str(e)}")
            st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>🚀 Powered by <strong>ClickZetta</strong> + <strong>LangChain</strong></p>
        <p>📧 如有问题，请联系技术支持</p>
    </div>
    """,
    unsafe_allow_html=True
)