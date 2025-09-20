import os, sys, tempfile, streamlit as st, uuid
import json
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
from components.common import UIComponents, display_table_schema
from config.clickzetta_config import load_app_config

# 应用配置
app_config = load_app_config("qa")

# Helper function to show educational help documentation
def show_help_documentation():
    """显示详细的帮助文档"""
    st.markdown("# 📚 ClickZetta 智能问答系统 - 学习指南")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 系统概述",
        "🏗️ 技术架构",
        "💡 代码示例",
        "🔧 最佳实践"
    ])

    with tab1:
        st.markdown("## 📋 系统功能概述")

        st.markdown("""
        ### 🎯 核心功能

        **ClickZetta 智能问答系统** 是一个基于 **RAG (检索增强生成) 架构** 的企业级问答解决方案，集成了多个ClickZetta存储组件。

        #### 🔍 主要特点：
        - **🧠 VectorStore**: 存储文档向量，支持语义相似性检索
        - **💬 ChatMessageHistory**: 持久化对话历史，支持多轮会话
        - **🤖 智能检索**: 结合向量检索和生成式AI的RAG架构
        - **📊 会话管理**: 独立会话ID，支持多用户并发使用
        - **🔄 实时交互**: 流式对话界面，提供即时反馈
        """)

        st.markdown("---")

        st.markdown("## 🏢 企业应用场景")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 📚 智能知识库
            - **企业文档查询**: 快速查找公司政策、流程文档
            - **技术支持**: 基于产品文档的自动客服
            - **培训助手**: 员工培训材料的智能问答
            """)

            st.markdown("""
            #### 🏥 专业领域应用
            - **医疗诊断辅助**: 基于医学文献的辅助诊断
            - **法律咨询**: 法律条文和案例的智能检索
            - **学术研究**: 研究论文的智能摘要和问答
            """)

        with col2:
            st.markdown("""
            #### 💼 业务效率提升
            - **会议助手**: 会议纪要的智能问答
            - **销售支持**: 产品资料的快速检索
            - **项目管理**: 项目文档的智能查询
            """)

            st.markdown("""
            #### 🔍 个人知识管理
            - **学习笔记**: 个人笔记的智能整理
            - **文档归档**: 自动分类和检索文档
            - **信息发现**: 发现文档间的潜在联系
            """)

    with tab2:
        st.markdown("## 🏗️ 技术架构深度解析")

        # Architecture diagram
        st.markdown("""
        ### 📐 RAG (检索增强生成) 架构图

        ```
        用户提问
            ↓
        ┌─────────────────────┐
        │   问题预处理          │ ← 查询优化层
        │ (Query Processing)  │
        └─────────────────────┘
            ↓
        ┌─────────────────────┐
        │ ClickZetta          │ ← 向量检索层
        │ VectorStore         │
        │ 语义相似性搜索        │
        └─────────────────────┘
            ↓
        ┌─────────────────────┐
        │   检索结果           │ ← 上下文构建层
        │   + 历史对话         │
        └─────────────────────┘
            ↓
        ┌─────────────────────┐
        │   通义千问 AI        │ ← 生成回答层
        │   (RAG提示词)        │
        └─────────────────────┘
            ↓
        ┌─────────────────────┐
        │ ClickZetta          │ ← 记忆存储层
        │ ChatMessageHistory  │
        └─────────────────────┘
            ↓
        ┌─────────────────────┐
        │   用户界面展示        │ ← 交互展示层
        └─────────────────────┘
        ```
        """)

        st.markdown("---")

        st.markdown("## 🗄️ ClickZetta 存储组件详解")

        # Multi-component explanation
        st.markdown("""
        ### 🧠 VectorStore + 💬 ChatMessageHistory - 双存储架构

        本应用同时使用了两个核心ClickZetta存储组件，实现完整的RAG+记忆功能：
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 🧠 VectorStore (向量存储)
            **类比**: 像一个**超级智能的图书索引**
            - 📚 将文档转换为数学向量表示
            - 🔍 支持"找相似内容"而非"找关键词"
            - ⚡ 毫秒级语义检索性能
            - 🎯 为RAG提供相关上下文
            """)

        with col2:
            st.markdown("""
            #### 💬 ChatMessageHistory (对话存储)
            **类比**: 像一个**永不遗忘的对话记录员**
            - 💾 持久化存储每轮对话
            - 🔄 支持多会话并发管理
            - 📊 提供会话统计和分析
            - 🧠 为AI提供上下文记忆
            """)

        st.markdown("""
        #### 🔧 技术特性对比

        | 特性 | VectorStore | ChatMessageHistory |
        |------|-------------|-------------------|
        | **数据类型** | 文档向量+元数据 | 结构化对话记录 |
        | **查询方式** | 相似性搜索 | 时间/会话ID查询 |
        | **主要用途** | 知识检索 | 对话记忆 |
        | **表结构** | `{vector_table}` | `{chat_table}` |
        | **索引类型** | 向量索引(HNSW) | B+树索引 |
        """.format(
            vector_table=app_config.get_vector_table_name("qa"),
            chat_table=app_config.get_chat_table_name("qa")
        ))

        st.markdown("---")

        st.markdown("## 🤖 RAG 工作流程详解")

        # RAG workflow explanation
        st.markdown("""
        ### 🔄 问答生成完整流程

        #### 1️⃣ 文档预处理阶段
        ```python
        # PDF文档加载和分页
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        # 向量化并存储到ClickZetta VectorStore
        vectorstore.add_documents(pages)
        ```

        #### 2️⃣ 用户提问阶段
        ```python
        # 用户问题向量化
        query_embedding = embeddings.embed_query(user_question)

        # 语义相似性检索
        relevant_docs = vectorstore.similarity_search(user_question, k=5)
        ```

        #### 3️⃣ 上下文构建阶段
        ```python
        # 组合检索结果和历史对话
        context = "\\n".join([doc.page_content for doc in relevant_docs])
        chat_history = chat_memory.get_messages()
        ```

        #### 4️⃣ AI回答生成阶段
        ```python
        # 使用RAG提示词生成答案
        qa_chain = RetrievalQA.from_chain_type(
            llm=tongyi_llm,
            retriever=vectorstore.as_retriever()
        )
        answer = qa_chain.invoke({"query": user_question})
        ```

        #### 5️⃣ 记忆存储阶段
        ```python
        # 存储对话到ChatMessageHistory
        chat_memory.add_user_message(user_question)
        chat_memory.add_ai_message(answer)
        ```
        """)

    with tab3:
        st.markdown("## 💡 核心代码示例")

        st.markdown("### 🔧 双存储组件初始化")

        st.code("""
# 1. ClickZetta 引擎初始化
engine = ClickZettaEngine(
    service="your-service",
    instance="your-instance",
    workspace="your-workspace",
    schema="your-schema",
    username="your-username",
    password="your-password",
    vcluster="your-vcluster"
)

# 2. VectorStore 初始化 (知识库)
vectorstore = ClickZettaVectorStore(
    engine=engine,
    embeddings=DashScopeEmbeddings(
        dashscope_api_key="your-api-key",
        model="text-embedding-v4"
    ),
    table_name="qa_knowledge_vectors",     # 向量表
    distance_metric="cosine"
)

# 3. ChatMessageHistory 初始化 (对话记忆)
chat_memory = ClickZettaChatMessageHistory(
    engine=engine,
    session_id="unique-session-id",
    table_name="qa_chat_history"           # 对话表
)

# 4. 通义千问语言模型配置
llm = Tongyi(
    dashscope_api_key="your-api-key",
    model_name="qwen-plus",
    temperature=0.1                        # 问答需要较低创造性
)
        """, language="python")

        st.markdown("---")

        st.markdown("### 🎯 RAG 问答链构建")

        st.code("""
# 构建检索问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",                    # 将检索内容组合后提问
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 5}             # 检索Top5相关文档
    ),
    verbose=True                           # 显示检索过程
)

# 执行问答
result = qa_chain.invoke({
    "query": "用户问题"
})

# 提取答案
answer = result.get("result", str(result))
        """, language="python")

        st.markdown("---")

        st.markdown("### 💬 会话记忆管理")

        st.code("""
# 会话记忆操作示例
class ChatSession:
    def __init__(self, engine, session_id):
        self.chat_memory = ClickZettaChatMessageHistory(
            engine=engine,
            session_id=session_id,
            table_name="qa_chat_history"
        )

    def add_conversation(self, user_msg, ai_response):
        # 添加用户消息
        self.chat_memory.add_user_message(user_msg)
        # 添加AI回复
        self.chat_memory.add_ai_message(ai_response)

    def get_history(self):
        # 获取完整对话历史
        return self.chat_memory.messages

    def clear_history(self):
        # 清空当前会话历史
        self.chat_memory.clear()

# 使用示例
session = ChatSession(engine, "user-session-123")
session.add_conversation("什么是机器学习？", "机器学习是...")
        """, language="python")

        st.markdown("---")

        st.markdown("### 📊 数据表结构示例")

        st.code("""
-- VectorStore 表结构 (知识库)
CREATE TABLE qa_knowledge_vectors (
    id String,                    -- 文档片段唯一标识
    content String,               -- 原始文档内容
    metadata String,              -- JSON格式元数据
    embedding Array(Float32),     -- 1536维向量表示
    created_at DateTime           -- 创建时间
) ENGINE = ReplicatedMergeTree()
ORDER BY id;

-- ChatMessageHistory 表结构 (对话记录)
CREATE TABLE qa_chat_history (
    session_id String,            -- 会话唯一标识
    message_id String,            -- 消息唯一标识
    message_type String,          -- human/ai 消息类型
    content String,               -- 消息内容
    timestamp DateTime,           -- 消息时间戳
    metadata String               -- 扩展元数据
) ENGINE = ReplicatedMergeTree()
ORDER BY (session_id, timestamp);

-- 常用查询示例
-- 1. 获取会话历史
SELECT message_type, content, timestamp
FROM qa_chat_history
WHERE session_id = 'session-123'
ORDER BY timestamp;

-- 2. 向量相似性搜索
SELECT id, content,
       cosineDistance(embedding, [0.1, 0.2, ...]) as similarity
FROM qa_knowledge_vectors
ORDER BY similarity ASC
LIMIT 5;
        """, language="sql")

    with tab4:
        st.markdown("## 🔧 最佳实践与优化建议")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### ⚡ 性能优化

            #### 🧠 向量检索优化
            - **检索数量**: 调整k值平衡精度和性能(通常3-10)
            - **向量维度**: 使用合适的嵌入模型维度
            - **距离度量**: cosine适合文本，euclidean适合数值

            #### 💬 对话记忆优化
            - **会话管理**: 及时清理过期会话数据
            - **记忆窗口**: 限制历史消息数量(5-20轮)
            - **并发控制**: 使用唯一session_id避免冲突

            #### 🤖 AI回答优化
            - **温度设置**: 问答任务使用低温度(0.1-0.3)
            - **提示词优化**: 明确指定回答格式和要求
            - **上下文长度**: 控制检索内容长度避免超限
            """)

        with col2:
            st.markdown("""
            ### 🛡️ 企业级部署

            #### 🔐 安全与权限
            - **数据隔离**: 不同用户使用独立schema
            - **访问控制**: 基于角色的数据访问权限
            - **敏感信息**: 避免在对话中暴露隐私数据

            #### 📊 监控与运维
            - **会话统计**: 监控活跃会话数和问答质量
            - **性能监控**: 跟踪检索延迟和AI响应时间
            - **容量规划**: 定期清理历史数据控制存储成本

            #### 🔄 可扩展性
            - **水平扩展**: 利用ClickZetta分布式架构
            - **负载均衡**: 多实例部署分散用户请求
            - **缓存策略**: 热点问题使用缓存提升响应速度
            """)

        st.markdown("---")

        st.markdown("## 🎓 学习建议")

        st.markdown("""
        ### 📚 循序渐进的学习路径

        #### 🟢 初级阶段 (理解基础概念)
        1. **体验问答流程**: 上传文档，进行简单问答
        2. **观察检索过程**: 点击"检索详情"了解RAG工作原理
        3. **测试会话记忆**: 进行多轮对话，观察上下文保持

        #### 🟡 中级阶段 (掌握技术细节)
        1. **理解RAG架构**: 学习检索+生成的组合机制
        2. **调试检索效果**: 调整检索参数优化答案质量
        3. **管理会话状态**: 理解session_id和记忆窗口概念

        #### 🔴 高级阶段 (企业级应用)
        1. **性能调优**: 优化大规模文档的检索性能
        2. **多租户部署**: 设计多用户隔离的部署架构
        3. **业务集成**: 与企业现有系统的API集成

        ### 📖 相关资源
        - **[ClickZetta 官方文档](https://www.yunqi.tech/documents/)**: 获取最新的平台功能和最佳实践
        - **[LangChain RAG指南](https://docs.langchain.com/docs/use-cases/question-answering)**: 深入了解RAG架构
        - **[通义千问 API](https://help.aliyun.com/zh/dashscope/)**: DashScope 平台使用指南
        """)

# Streamlit app configuration
st.set_page_config(
    page_title="ClickZetta Intelligent Q&A",
    page_icon="🤖",
    layout="wide"
)

# Main navigation
st.sidebar.markdown("## 📋 导航菜单")
page_selection = st.sidebar.selectbox(
    "选择功能页面",
    ["🚀 智能问答", "📚 学习指南"],
    key="qa_page_selection"
)

if page_selection == "📚 学习指南":
    show_help_documentation()
    st.stop()

st.title('🤖 ClickZetta 智能问答系统')
st.markdown("*基于 ClickZetta VectorStore + ChatMessageHistory + 通义千问 AI 的企业级RAG问答系统*")

# Add educational info banner
st.info("""
🎯 **系统特色**:
• **🧠 VectorStore**: 使用 `{vector_table}` 表存储文档向量，支持语义检索
• **💬 ChatMessageHistory**: 使用 `{chat_table}` 表存储对话历史，支持多轮会话
• **🤖 RAG架构**: 检索增强生成，结合向量检索和AI生成的最佳实践

💡 **使用提示**: 点击侧边栏的"📚 学习指南"了解RAG架构和双存储组件的详细原理
""".format(
    vector_table=app_config.get_vector_table_name("qa"),
    chat_table=app_config.get_chat_table_name("qa")
))

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
        embedding_model_options = ["text-embedding-v4", "text-embedding-v3", "text-embedding-v2", "text-embedding-v1"]
        embedding_model_index = 0  # 默认使用 text-embedding-v4
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

    # Data Management
    with st.expander("🗑️ 数据管理"):
        st.write("**清空知识库数据**")
        st.caption("删除所有向量数据和聊天历史，重新开始")

        if st.button("🗑️ 清空所有数据", type="secondary", help="删除向量数据和聊天历史"):
            if clickzetta_configured:
                try:
                    from langchain_clickzetta import ClickZettaEngine
                    engine = ClickZettaEngine(
                        service=clickzetta_service,
                        instance=clickzetta_instance,
                        workspace=clickzetta_workspace,
                        schema=clickzetta_schema,
                        username=clickzetta_username,
                        password=clickzetta_password,
                        vcluster=clickzetta_vcluster
                    )

                    # 清空向量表
                    vector_table = f"langchain_qa_vectors"
                    delete_query = f"DELETE FROM {vector_table}"
                    engine.execute_query(delete_query)

                    # 清空聊天历史表
                    chat_table = f"langchain_qa_chat_history"
                    delete_chat_query = f"DELETE FROM {chat_table}"
                    try:
                        engine.execute_query(delete_chat_query)
                    except:
                        pass  # 聊天表可能不存在

                    # 重置session状态
                    st.session_state.retriever = None
                    st.session_state.loaded_doc = None
                    st.session_state.chat_history = []
                    st.session_state.memory = None

                    st.success("✅ 数据已清空，请重新上传文档")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 清空数据失败: {e}")
            else:
                st.warning("⚠️ 请先配置ClickZetta连接")

        st.write("**统计信息**")
        if clickzetta_configured:
            try:
                from langchain_clickzetta import ClickZettaEngine
                engine = ClickZettaEngine(
                    service=clickzetta_service,
                    instance=clickzetta_instance,
                    workspace=clickzetta_workspace,
                    schema=clickzetta_schema,
                    username=clickzetta_username,
                    password=clickzetta_password,
                    vcluster=clickzetta_vcluster
                )

                # 检查向量数据
                vector_table = f"langchain_qa_vectors"
                try:
                    count_query = f"SELECT COUNT(*) as count FROM {vector_table}"
                    result, _ = engine.execute_query(count_query)
                    if result:
                        vector_count = result[0]['count']
                        st.metric("📄 向量数据", vector_count)
                except:
                    st.metric("📄 向量数据", "表不存在")

                # 检查聊天记录
                chat_table = f"langchain_qa_chat_history"
                try:
                    count_query = f"SELECT COUNT(*) as count FROM {chat_table}"
                    result, _ = engine.execute_query(count_query)
                    if result:
                        chat_count = result[0]['count']
                        st.metric("💬 聊天记录", chat_count)
                except:
                    st.metric("💬 聊天记录", "表不存在")

            except Exception as e:
                st.caption(f"无法获取统计信息: {str(e)[:50]}...")
        else:
            st.caption("请先配置ClickZetta连接")

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

                # 自动检测并加载已有的向量数据
                if not st.session_state.retriever:
                    try:
                        vector_table = app_config.get_vector_table_name("qa")

                        # 先检查表是否存在
                        show_tables_query = f"SHOW TABLES LIKE '{vector_table}'"
                        tables_result, _ = engine.execute_query(show_tables_query)

                        if tables_result and len(tables_result) > 0:
                            # 表存在，检查数据量
                            count_query = f"SELECT COUNT(*) as count FROM {vector_table}"
                            count_result, _ = engine.execute_query(count_query)

                            if count_result and len(count_result) > 0:
                                vector_count = count_result[0]['count']

                                if vector_count > 0:
                                    # 自动加载现有数据，不阻塞流程
                                    try:
                                        embeddings = DashScopeEmbeddings(
                                            dashscope_api_key=api_key,
                                            model="text-embedding-v4"
                                        )

                                        vectorstore = ClickZettaVectorStore(
                                            engine=engine,
                                            embeddings=embeddings,
                                            table_name=vector_table,
                                            metric="cosine"
                                        )

                                        st.session_state.retriever = vectorstore.as_retriever(
                                            search_kwargs={"k": 5}
                                        )

                                        st.info(f"🎉 自动加载知识库成功！已有 {vector_count} 条向量数据，可直接开始问答")
                                        st.session_state.loaded_doc = "已有数据"  # 标记为已加载状态
                                    except Exception as e:
                                        if "dimension" in str(e) or "COSINE_DISTANCE" in str(e):
                                            st.warning(f"⚠️ 检测到向量维度不匹配（表中有 {vector_count} 条数据）。请在侧边栏选择正确的embedding模型或清空数据。")
                                        else:
                                            st.error(f"❌ 加载失败: {e}")
                                # else: 表存在但无数据，正常情况，不显示任何信息
                        # else: 表不存在，正常情况，不显示任何信息

                    except Exception as e:
                        # 表不存在或查询失败，这是正常情况，用户需要先上传文档
                        # 只在真正的错误情况下显示（非表不存在的情况）
                        pass

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

    if st.session_state.chat_memory:
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

    with st.expander("🗄️ 查看存储表结构", expanded=False):
        if st.session_state.engine:
            try:
                st.subheader("📊 ClickZetta 存储表详情")

                # Vector Store Table
                vector_table = app_config.get_vector_table_name("qa")
                st.write(f"**🧠 VectorStore 表**: `{vector_table}`")

                try:
                    # 先检查表是否存在
                    show_tables_query = f"SHOW TABLES LIKE '{vector_table}'"
                    tables_result, _ = st.session_state.engine.execute_query(show_tables_query)

                    if tables_result and len(tables_result) > 0:
                        # 表存在，获取schema信息
                        vector_schema_query = f"DESCRIBE TABLE EXTENDED {vector_table}"
                        vector_result, vector_description = st.session_state.engine.execute_query(vector_schema_query)

                        if vector_result and len(vector_result) > 0:
                            # 使用通用的表结构显示函数
                            display_table_schema(vector_result)

                        # Get vector count
                        vector_count_query = f"SELECT count(*) as total_vectors FROM {vector_table}"
                        vector_count_result, _ = st.session_state.engine.execute_query(vector_count_query)
                        if vector_count_result:
                            vector_count = vector_count_result[0]['total_vectors']
                            st.metric("🧠 存储的文档向量数", vector_count)
                    else:
                        st.info(f"📋 表 `{vector_table}` 尚未创建。上传文档后会自动创建。")
                except Exception as e:
                    st.warning(f"⚠️ 无法获取 VectorStore 表信息。请检查Lakehouse连接。")

                st.markdown("---")

                # Chat Message History Table
                chat_table = app_config.get_chat_table_name("qa")
                st.write(f"**💬 ChatMessageHistory 表**: `{chat_table}`")

                try:
                    # 先检查表是否存在
                    show_tables_query = f"SHOW TABLES LIKE '{chat_table}'"
                    tables_result, _ = st.session_state.engine.execute_query(show_tables_query)

                    if tables_result and len(tables_result) > 0:
                        # 表存在，获取schema信息
                        chat_schema_query = f"DESCRIBE TABLE EXTENDED {chat_table}"
                        chat_result, chat_description = st.session_state.engine.execute_query(chat_schema_query)

                        if chat_result and len(chat_result) > 0:
                            # 使用通用的表结构显示函数
                            display_table_schema(chat_result)

                        # Get message count for current session
                        message_count_query = f"SELECT count(*) as total_messages FROM {chat_table} WHERE session_id = '{st.session_state.session_id}'"
                        message_count_result, _ = st.session_state.engine.execute_query(message_count_query)
                        if message_count_result:
                            message_count = message_count_result[0]['total_messages']
                            st.metric("💬 当前会话消息数", message_count)

                        # Get total sessions count
                        session_count_query = f"SELECT COUNT(DISTINCT session_id) as total_sessions FROM {chat_table}"
                        session_count_result, _ = st.session_state.engine.execute_query(session_count_query)
                        if session_count_result:
                            session_count = session_count_result[0]['total_sessions']
                            st.metric("📊 历史会话总数", session_count)
                    else:
                        st.info(f"📋 表 `{chat_table}` 尚未创建。开始聊天后会自动创建。")
                except Exception as e:
                    st.warning(f"⚠️ 无法获取 ChatMessageHistory 表信息。请检查Lakehouse连接。")

                st.write("**📖 更多信息**: 访问 [ClickZetta 官方文档](https://www.yunqi.tech/documents/) 了解存储组件详细功能")

            except Exception as e:
                st.error(f"Lakehouse连接错误: {e}")
        else:
            st.info("⚠️ 请先连接 ClickZetta Lakehouse")

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