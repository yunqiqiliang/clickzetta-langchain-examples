import os, tempfile, streamlit as st, sys
import pandas as pd
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

# Helper function to show educational help documentation
def show_help_documentation():
    """显示详细的帮助文档"""
    st.markdown("# 📚 ClickZetta 文档智能摘要系统 - 学习指南")

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

        **ClickZetta 文档智能摘要系统** 是一个基于 **ClickZetta VectorStore + 通义千问 AI** 的企业级文档摘要解决方案。

        #### 🔍 主要特点：
        - **📄 PDF文档解析**: 使用 LangChain PyPDFLoader 智能解析PDF文档
        - **🧠 向量化存储**: 利用 ClickZetta VectorStore 存储文档向量表示
        - **🤖 AI智能摘要**: 集成通义千问大语言模型生成高质量摘要
        - **🎛️ 个性化配置**: 支持摘要语言、长度、风格的自定义设置
        - **📊 实时监控**: 提供详细的处理状态和技术指标展示
        """)

        st.markdown("---")

        st.markdown("## 🏢 企业应用场景")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 📊 商业文档处理
            - **合同摘要**: 快速提取合同关键条款
            - **报告总结**: 生成财务、市场报告摘要
            - **政策解读**: 将复杂政策文件转化为要点
            """)

            st.markdown("""
            #### 📚 知识管理
            - **技术文档**: 提取技术规范核心内容
            - **培训材料**: 生成培训文档精华版本
            - **研究论文**: 快速获取学术论文要点
            """)

        with col2:
            st.markdown("""
            #### 🏛️ 组织效率提升
            - **会议纪要**: 从长篇会议记录提取决策要点
            - **法律文件**: 法律条文的通俗化解释
            - **产品手册**: 复杂产品说明的简化版本
            """)

            st.markdown("""
            #### 🔍 信息检索增强
            - **文档索引**: 为大量文档建立语义索引
            - **内容发现**: 通过向量搜索发现相关内容
            - **智能归档**: 基于内容特征自动分类
            """)

    with tab2:
        st.markdown("## 🏗️ 技术架构深度解析")

        # Architecture diagram
        st.markdown("""
        ### 📐 系统架构图

        ```
        用户上传PDF文档
              ↓
        ┌─────────────────────┐
        │   PyPDFLoader      │ ← 文档解析层
        │   文档分页加载       │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │  DashScope嵌入模型  │ ← 向量化层
        │  text-embedding-v4  │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │ ClickZetta          │ ← 存储层
        │ VectorStore         │
        │ (向量数据库)         │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │  相似性搜索          │ ← 检索层
        │  (Cosine距离)       │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │   通义千问 AI        │ ← AI处理层
        │   (qwen-plus)       │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │   智能摘要结果       │ ← 输出层
        └─────────────────────┘
        ```
        """)

        st.markdown("---")

        st.markdown("## 🗄️ ClickZetta 存储组件详解")

        # VectorStore detailed explanation
        st.markdown("""
        ### 🧠 VectorStore (向量存储) - 本应用的核心存储组件

        **类比理解**: VectorStore 就像是一个**超级智能的图书管理员大脑**
        - 📚 **传统图书馆**: 按照分类号排列书籍 (关键词检索)
        - 🧠 **VectorStore**: 理解书籍的"语义含义"，能找到意思相近的内容 (语义检索)

        #### 🔧 技术特性
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **📊 数据存储结构**
            - **表名**: `{table_prefix}_summary_vectors`
            - **向量维度**: 1536维 (text-embedding-v4)
            - **距离度量**: Cosine相似度
            - **索引类型**: HNSW高性能向量索引
            """.format(table_prefix=app_config.get_vector_table_name("summary").split('_')[0]))

        with col2:
            st.markdown("""
            **⚡ 性能优化**
            - **批量插入**: 支持大量文档快速存储
            - **增量更新**: 新文档无需重建整个索引
            - **分布式存储**: 利用ClickZetta分布式架构
            - **内存优化**: 高效的向量压缩算法
            """)

        st.markdown("---")

        st.markdown("## 🤖 AI 处理流程详解")

        # AI processing workflow
        st.markdown("""
        ### 🔄 摘要生成工作流

        #### 1️⃣ 文档预处理阶段
        ```python
        # PDF文档加载和分页
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        # 每页都会成为一个独立的文档块
        ```

        #### 2️⃣ 向量化存储阶段
        ```python
        # 文档向量化并存储到ClickZetta
        vectorstore = ClickZettaVectorStore(
            engine=engine,
            embeddings=DashScopeEmbeddings(),
            table_name="summary_vectors"
        )
        vectorstore.add_documents(pages)
        ```

        #### 3️⃣ 智能检索阶段
        ```python
        # 使用语义检索找到最相关的文档片段
        relevant_docs = vectorstore.similarity_search(
            "文档摘要", k=10
        )
        ```

        #### 4️⃣ AI摘要生成阶段
        ```python
        # 使用通义千问生成个性化摘要
        chain = load_summarize_chain(
            llm=Tongyi(),
            chain_type="stuff",
            prompt=custom_prompt
        )
        summary = chain.invoke({"input_documents": relevant_docs})
        ```
        """)

    with tab3:
        st.markdown("## 💡 核心代码示例")

        st.markdown("### 🔧 关键组件初始化")

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

# 2. DashScope 嵌入模型配置
embeddings = DashScopeEmbeddings(
    dashscope_api_key="your-api-key",
    model="text-embedding-v4"  # 最新的嵌入模型
)

# 3. 通义千问语言模型配置
llm = Tongyi(
    dashscope_api_key="your-api-key",
    model_name="qwen-plus",      # 平衡性能和成本
    temperature=0.1              # 低温度确保摘要稳定性
)

# 4. ClickZetta向量存储初始化
vectorstore = ClickZettaVectorStore(
    engine=engine,
    embeddings=embeddings,
    table_name="document_summary_vectors",
    distance_metric="cosine"     # 适合文本相似性计算
)
        """, language="python")

        st.markdown("---")

        st.markdown("### 🎯 自定义摘要提示词")

        st.code("""
# 构建个性化摘要提示词
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=f'''
{language_instruction}{style_instruction}，
字数控制在{summary_length}字以内。

文档内容：
{text}

摘要：
'''
)

# 摘要链配置
chain = load_summarize_chain(
    llm,
    chain_type="stuff",        # 适合中短文档的处理方式
    prompt=summary_prompt
)

# 执行摘要生成
result = chain.invoke({"input_documents": relevant_docs})
        """, language="python")

        st.markdown("---")

        st.markdown("### 📊 数据表结构示例")

        st.code("""
-- ClickZetta VectorStore 表结构
CREATE TABLE document_summary_vectors (
    id String,                    -- 文档唯一标识
    content String,               -- 原始文档内容
    metadata String,              -- JSON格式元数据
    embedding Array(Float32),     -- 1536维向量表示
    created_at DateTime           -- 创建时间
) ENGINE = ReplicatedMergeTree()
ORDER BY id;

-- 示例查询：相似性搜索
SELECT id, content, metadata,
       cosineDistance(embedding, [0.1, 0.2, ...]) as similarity
FROM document_summary_vectors
ORDER BY similarity ASC
LIMIT 10;
        """, language="sql")

    with tab4:
        st.markdown("## 🔧 最佳实践与优化建议")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### ⚡ 性能优化

            #### 📄 文档处理优化
            - **文件大小**: 建议单个PDF不超过10MB
            - **页数限制**: 超过100页的文档建议分片处理
            - **内容质量**: 确保PDF文本可提取（非扫描版）

            #### 🧠 向量存储优化
            - **批量处理**: 一次处理多个文档提高效率
            - **索引维护**: 定期优化向量索引性能
            - **存储清理**: 删除过期或重复的向量数据

            #### 🤖 AI调用优化
            - **温度设置**: 摘要任务使用低温度(0.1-0.3)
            - **长度控制**: 根据使用场景调整摘要长度
            - **并发限制**: 避免过多并发API调用
            """)

        with col2:
            st.markdown("""
            ### 🛡️ 安全与稳定性

            #### 🔐 数据安全
            - **环境变量**: 所有敏感信息使用环境变量
            - **连接加密**: 确保数据库连接使用SSL
            - **权限控制**: 最小权限原则配置数据库访问

            #### 🔄 错误处理
            - **连接重试**: 网络异常时自动重试机制
            - **优雅降级**: API失败时的备用方案
            - **日志记录**: 详细的错误日志便于排查

            #### 📊 监控告警
            - **性能监控**: 跟踪API调用延迟和成功率
            - **存储监控**: 监控向量存储的使用情况
            - **成本控制**: 设置API调用频率限制
            """)

        st.markdown("---")

        st.markdown("## 🎓 学习建议")

        st.markdown("""
        ### 📚 循序渐进的学习路径

        #### 🟢 初级阶段 (理解基础概念)
        1. **熟悉界面操作**: 上传文档，尝试不同摘要设置
        2. **观察处理流程**: 注意文档加载→向量化→摘要生成的各个步骤
        3. **对比摘要质量**: 使用不同的语言模型和参数设置

        #### 🟡 中级阶段 (理解技术原理)
        1. **学习向量检索**: 理解相似性搜索的工作原理
        2. **研究提示词工程**: 尝试修改摘要提示词模板
        3. **探索存储结构**: 查看ClickZetta中的实际数据表

        #### 🔴 高级阶段 (深度定制开发)
        1. **性能调优**: 优化大文档的处理流程
        2. **功能扩展**: 添加多语言支持、图表提取等
        3. **集成部署**: 将系统集成到企业现有工作流中

        ### 📖 相关资源
        - **[ClickZetta 官方文档](https://www.yunqi.tech/documents/)**: 获取最新的平台功能和最佳实践
        - **[LangChain 文档](https://docs.langchain.com/)**: 深入了解 LangChain 框架
        - **[通义千问 API](https://help.aliyun.com/zh/dashscope/)**: DashScope 平台使用指南
        """)

# Streamlit app configuration
st.set_page_config(
    page_title="ClickZetta Document Summary",
    page_icon="📄",
    layout="wide"
)

# Main navigation
st.sidebar.markdown("## 📋 导航菜单")
page_selection = st.sidebar.selectbox(
    "选择功能页面",
    ["🚀 文档摘要", "📚 学习指南"],
    key="summary_page_selection"
)

if page_selection == "📚 学习指南":
    show_help_documentation()
    st.stop()

st.title('📄 ClickZetta 文档智能摘要')
st.markdown("*基于 ClickZetta VectorStore + 通义千问 AI 的企业级文档摘要系统*")

# Add educational info banner
st.info("""
🎯 **系统特色**:
• **🧠 VectorStore**: 使用 `{table_name}` 表存储文档向量，支持语义相似性检索
• **🤖 通义千问**: 集成 qwen-plus 模型，提供高质量的中英文摘要生成
• **📊 智能检索**: 通过向量相似性搜索找到最相关的文档片段进行摘要

💡 **使用提示**: 点击侧边栏的"📚 学习指南"了解详细的技术原理和最佳实践
""".format(table_name=app_config.get_vector_table_name("summary")))

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

    # 数据管理
    st.subheader("🗑️ 数据管理")

    # 统计信息
    with st.expander("📊 统计信息"):
        if clickzetta_configured:
            try:
                engine = ClickZettaEngine(
                    service=clickzetta_service,
                    instance=clickzetta_instance,
                    workspace=clickzetta_workspace,
                    schema=clickzetta_schema,
                    username=clickzetta_username,
                    password=clickzetta_password,
                    vcluster=clickzetta_vcluster
                )

                table_name = app_config.get_vector_table_name("summary")

                try:
                    # 检查表是否存在
                    show_tables_query = f"SHOW TABLES LIKE '{table_name}'"
                    tables_result, _ = engine.execute_query(show_tables_query)

                    if tables_result and len(tables_result) > 0:
                        # 获取向量数据数量
                        count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                        count_result, _ = engine.execute_query(count_query)
                        if count_result and len(count_result) > 0:
                            vector_count = count_result[0]['count']
                            st.metric("🧠 向量数据", f"{vector_count} 条")

                            if vector_count > 0:
                                st.info(f"💡 检测到已有 {vector_count} 条文档向量，可直接进行摘要")
                        else:
                            st.warning("⚠️ 无法获取数据统计")
                    else:
                        st.info("📋 暂无向量数据表")

                except Exception as e:
                    st.warning(f"⚠️ 无法获取统计信息: {e}")

            except Exception as e:
                st.error(f"❌ 无法获取统计信息: {e}")
        else:
            st.warning("⚠️ 请先配置ClickZetta连接")

    # 清空数据功能
    with st.expander("🗑️ 数据清空"):
        st.write("**清空文档向量数据**")
        st.caption("删除所有向量数据，重新开始")

        if st.button("🗑️ 清空所有向量数据", type="secondary", help="删除向量数据"):
            if clickzetta_configured:
                try:
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
                    table_name = app_config.get_vector_table_name("summary")
                    delete_query = f"DELETE FROM {table_name}"
                    engine.execute_query(delete_query)

                    st.success("✅ 向量数据已清空，请重新上传文档")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 清空数据失败: {e}")
            else:
                st.warning("⚠️ 请先配置ClickZetta连接")

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
                    table_name = app_config.get_vector_table_name("summary")
                    st.write(f"**📊 ClickZetta VectorStore 存储详情**:")
                    st.write(f"• **向量存储表**: `{table_name}`")
                    st.write(f"• **嵌入模型**: `{embedding_model}` (1536维向量)")
                    st.write(f"• **语言模型**: `{llm_model}` (通义千问)")
                    st.write(f"• **距离度量**: `cosine` 相似度")
                    st.write(f"• **处理的文档片段**: {len(relevant_docs)} / {len(pages)}")

                    # Add table inspection functionality
                    if st.button("🔍 查看向量表结构", key="inspect_vector_table"):
                        try:
                            # Get table schema
                            schema_query = f"DESCRIBE TABLE {table_name}"
                            schema_result, schema_description = engine.execute_query(schema_query)

                            if schema_result and schema_description and len(schema_result) > 0:
                                st.write("**📋 表结构信息**:")
                                # Handle duplicate column names
                                column_names = [desc[0] for desc in schema_description]
                                unique_column_names = []
                                name_counts = {}
                                for name in column_names:
                                    if name in name_counts:
                                        name_counts[name] += 1
                                        unique_column_names.append(f"{name}_{name_counts[name]}")
                                    else:
                                        name_counts[name] = 0
                                        unique_column_names.append(name)

                                schema_df = pd.DataFrame(schema_result, columns=unique_column_names)
                                st.dataframe(schema_df, use_container_width=True)

                                # Get record count
                                count_query = f"SELECT count(*) as total_vectors FROM {table_name}"
                                count_result, _ = engine.execute_query(count_query)
                                if count_result:
                                    total_count = count_result[0]['total_vectors']
                                    st.metric("📊 向量总数", total_count)
                            else:
                                st.warning(f"⚠️ 表 `{table_name}` 不存在或为空。请先使用摘要功能添加一些文档。")

                        except Exception as e:
                            st.warning(f"暂无法获取表结构信息: {e}")

                    st.write("**📖 更多信息**: 访问 [ClickZetta 官方文档](https://www.yunqi.tech/documents/) 了解VectorStore详细功能")

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