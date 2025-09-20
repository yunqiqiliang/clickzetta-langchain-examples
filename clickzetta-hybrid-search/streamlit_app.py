import streamlit as st
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json

# 添加父目录到路径以导入通用组件
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_clickzetta import (
    ClickZettaEngine,
    ClickZettaHybridStore,
    ClickZettaUnifiedRetriever,
    ClickZettaFullTextRetriever
)
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from config.clickzetta_config import (
    ClickZettaConfig,
    DashScopeConfig,
    load_app_config,
    render_clickzetta_config_form,
    render_dashscope_config_form,
    render_config_status
)
from components.common import (
    ClickZettaManager,
    DocumentProcessor,
    UIComponents,
    SessionManager,
    ValidationHelper
)

# 应用配置
app_config = load_app_config("hybrid_search")

# Helper function to show educational help documentation
def show_help_documentation():
    """显示详细的帮助文档"""
    st.markdown("# 📚 ClickZetta 混合搜索系统 - 学习指南")

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

        **ClickZetta 混合搜索系统** 是一个先进的企业级搜索解决方案，基于 **HybridStore + UnifiedRetriever** 架构，融合了向量搜索和全文搜索的优势。

        #### 🔍 主要特点：
        - **🧠 HybridStore**: 统一存储向量和全文索引，支持多种搜索模式
        - **🔄 UnifiedRetriever**: 智能检索器，可动态调整搜索策略
        - **⚖️ 权重调节**: 灵活调整向量搜索和全文搜索的权重比例
        - **📊 实时统计**: 详细的搜索性能监控和优化建议
        - **🧠 智能摘要**: 基于搜索结果的AI智能摘要生成
        """)

        st.markdown("---")

        st.markdown("## 🆚 三种搜索模式对比")

        # Search modes comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            #### 🧠 向量搜索 (Vector Search)
            **类比**: 像一个**理解语义的智能助手**
            - 📚 理解词汇的含义和上下文
            - 🔍 找到"意思相近"的内容
            - 🎯 适合概念性查询和语义搜索
            - ⚡ 对同义词、近义词敏感

            **优势**: 语义理解能力强
            **局限**: 对精确关键词匹配较弱
            """)

        with col2:
            st.markdown("""
            #### 📝 全文搜索 (Full-text Search)
            **类比**: 像一个**精确的文档索引员**
            - 🔍 精确匹配关键词和短语
            - ⚡ 支持复杂的布尔查询逻辑
            - 📊 基于词频和文档频率排序
            - 🇨🇳 智能中文分词处理

            **优势**: 精确匹配，速度快
            **局限**: 缺乏语义理解能力
            """)

        with col3:
            st.markdown("""
            #### ⚖️ 混合搜索 (Hybrid Search)
            **类比**: 像一个**全能的搜索专家**
            - 🎯 结合两种搜索方式的优势
            - ⚖️ 可调节权重比例 (alpha参数)
            - 🏆 获得最佳的搜索效果
            - 📈 适应不同类型的查询需求

            **优势**: 综合效果最佳
            **推荐**: 大多数场景的首选
            """)

        st.markdown("---")

        st.markdown("## 🏢 企业应用场景")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 📚 企业知识管理
            - **政策文档检索**: 快速找到相关的公司政策
            - **技术文档搜索**: 精确匹配技术术语和概念
            - **员工手册查询**: 支持自然语言和关键词搜索
            """)

            st.markdown("""
            #### 🔬 研究与开发
            - **论文文献检索**: 语义搜索相关研究方向
            - **专利技术查询**: 精确匹配技术关键词
            - **知识发现**: 发现概念间的潜在联系
            """)

        with col2:
            st.markdown("""
            #### 💼 商业智能
            - **市场报告分析**: 多维度搜索市场数据
            - **竞品情报收集**: 结合关键词和语义搜索
            - **客户反馈分析**: 理解客户真实意图
            """)

            st.markdown("""
            #### 📖 内容管理
            - **新闻内容检索**: 支持主题和关键词搜索
            - **法律条文查询**: 精确匹配法律术语
            - **教育资源搜索**: 智能推荐相关学习材料
            """)

    with tab2:
        st.markdown("## 🏗️ 技术架构深度解析")

        # Architecture diagram
        st.markdown("""
        ### 📐 混合搜索架构图

        ```
        用户查询输入
              ↓
        ┌─────────────────────┐
        │   查询预处理         │ ← 查询优化层
        │   (Query Processing) │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │ ClickZetta          │ ← 统一存储层
        │ HybridStore         │
        │ 向量+全文双索引      │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │ Unified Retriever   │ ← 智能检索层
        │ 权重调节 (alpha)     │
        └─────────────────────┘
             ↙    ↓    ↘
        ┌─────────────────────┐
        │  向量搜索    混合搜索   全文搜索  │ ← 多模式检索
        │  (Vector)  (Hybrid)  (Fulltext)│
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │   结果融合与排序     │ ← 结果处理层
        │   (Result Fusion)   │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │   AI智能摘要         │ ← 增强服务层
        │   (通义千问)         │
        └─────────────────────┘
        ```
        """)

        st.markdown("---")

        st.markdown("## 🗄️ ClickZetta 存储组件详解")

        # HybridStore detailed explanation
        st.markdown("""
        ### ⚖️ HybridStore - 一体化混合存储

        **类比理解**: HybridStore 就像是一个**超级智能的双语图书馆**
        - 📚 **向量语言**: 理解文档的语义含义 (embeddings)
        - 📝 **关键词语言**: 精确记录每个词汇 (full-text index)
        - 🔄 **智能翻译**: 在两种"语言"间无缝切换
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 🧠 向量存储部分
            - **嵌入维度**: 1536维 (text-embedding-v4)
            - **距离度量**: Cosine相似度
            - **索引类型**: HNSW高性能向量索引
            - **适用场景**: 语义搜索、概念匹配
            """)

        with col2:
            st.markdown("""
            #### 📝 全文索引部分
            - **分词器**: ik智能分词 (中文优化)
            - **索引结构**: 倒排索引 (Inverted Index)
            - **查询语法**: 支持布尔查询、短语查询
            - **适用场景**: 精确匹配、关键词搜索
            """)

        st.markdown("""
        #### 🔧 技术参数详情

        | 特性 | 向量搜索 | 全文搜索 | 混合搜索 |
        |------|---------|----------|----------|
        | **查询类型** | 语义相似 | 关键词匹配 | 两者结合 |
        | **响应速度** | 中等 | 快速 | 中等 |
        | **精确度** | 语义精确 | 字面精确 | 综合最优 |
        | **召回率** | 高 | 中等 | 最高 |
        | **参数控制** | embedding模型 | 分词器设置 | alpha权重 |
        | **存储表** | `{table_name}` | 同一张表 | 同一张表 |
        """.format(table_name=app_config.get_vector_table_name("hybrid_search")))

        st.markdown("---")

        st.markdown("## 🎛️ UnifiedRetriever 检索器详解")

        st.markdown("""
        ### 🔄 智能检索策略

        UnifiedRetriever 是混合搜索的核心控制器，通过 **alpha 参数** 灵活调节搜索策略：

        #### ⚖️ Alpha 权重参数 (0.0 - 1.0)
        - **alpha = 0.0**: 纯全文搜索 (100% Full-text)
        - **alpha = 0.3**: 全文为主 (30% Vector + 70% Full-text)
        - **alpha = 0.7**: 向量为主 (70% Vector + 30% Full-text) ← **推荐**
        - **alpha = 1.0**: 纯向量搜索 (100% Vector)

        #### 🎯 不同查询类型的最佳权重建议：
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **适合高alpha值 (0.7-1.0)**:
            - 概念性问题："什么是机器学习？"
            - 语义搜索："类似的解决方案"
            - 意图理解："如何提高效率？"
            - 同义词查询："优化" vs "改进"
            """)

        with col2:
            st.markdown("""
            **适合低alpha值 (0.0-0.3)**:
            - 精确关键词："API接口文档"
            - 专有名词："ClickZetta配置"
            - 代码片段："def function_name"
            - 数字编号："第三章节"
            """)

    with tab3:
        st.markdown("## 💡 核心代码示例")

        st.markdown("### 🔧 HybridStore 初始化")

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

# 2. 嵌入模型配置
embeddings = DashScopeEmbeddings(
    dashscope_api_key="your-api-key",
    model="text-embedding-v4"
)

# 3. HybridStore 初始化 (核心组件)
hybrid_store = ClickZettaHybridStore(
    engine=engine,
    embeddings=embeddings,
    table_name="hybrid_search_store",
    text_analyzer="ik",               # 中文智能分词
    distance_metric="cosine"          # 向量相似度度量
)

# 4. 添加文档到混合存储
hybrid_store.add_documents(documents)
        """, language="python")

        st.markdown("---")

        st.markdown("### 🎯 UnifiedRetriever 配置")

        st.code("""
# 创建统一检索器
retriever = ClickZettaUnifiedRetriever(
    hybrid_store=hybrid_store,
    search_type="hybrid",             # 搜索模式: hybrid/vector/fulltext
    alpha=0.7,                        # 权重: 0.7=向量70% + 全文30%
    k=5                               # 返回结果数量
)

# 动态调整搜索参数
retriever.search_type = "vector"     # 切换为纯向量搜索
retriever.alpha = 1.0                # 调整权重
retriever.k = 10                     # 增加结果数量

# 执行搜索
results = retriever.invoke("用户查询")

# 访问不同搜索模式
vector_results = retriever.get_relevant_documents(
    query="查询内容",
    search_type="vector"
)

fulltext_results = retriever.get_relevant_documents(
    query="查询内容",
    search_type="fulltext"
)
        """, language="python")

        st.markdown("---")

        st.markdown("### 🔍 搜索模式对比")

        st.code("""
# 搜索查询示例
query = "如何优化数据库性能"

# 1. 纯向量搜索 (语义理解)
vector_retriever = ClickZettaUnifiedRetriever(
    hybrid_store=hybrid_store,
    search_type="vector",
    alpha=1.0,
    k=5
)
vector_results = vector_retriever.invoke(query)
# 可能找到: "提升数据库效率", "数据库调优方案", "性能优化策略"

# 2. 纯全文搜索 (关键词匹配)
fulltext_retriever = ClickZettaUnifiedRetriever(
    hybrid_store=hybrid_store,
    search_type="fulltext",
    alpha=0.0,
    k=5
)
fulltext_results = fulltext_retriever.invoke(query)
# 可能找到: 包含"优化"、"数据库"、"性能"等关键词的文档

# 3. 混合搜索 (最佳效果)
hybrid_retriever = ClickZettaUnifiedRetriever(
    hybrid_store=hybrid_store,
    search_type="hybrid",
    alpha=0.7,
    k=5
)
hybrid_results = hybrid_retriever.invoke(query)
# 结合两种方式的优势，获得最佳搜索效果
        """, language="python")

        st.markdown("---")

        st.markdown("### 📊 数据表结构示例")

        st.code("""
-- HybridStore 表结构 (统一存储)
CREATE TABLE hybrid_search_store (
    id String,                         -- 文档唯一标识
    content String,                    -- 原始文档内容
    metadata String,                   -- JSON格式元数据
    embedding Array(Float32),          -- 1536维向量 (向量搜索用)
    content_tokens Array(String),      -- 分词结果 (全文搜索用)
    content_fulltext String,           -- 全文索引字段
    created_at DateTime,               -- 创建时间

    -- 向量索引 (用于向量搜索)
    INDEX vec_idx embedding TYPE vector(1536) METRIC cosine,

    -- 全文索引 (用于全文搜索)
    INDEX ft_idx content_fulltext TYPE fulltext('ik')
) ENGINE = ReplicatedMergeTree()
ORDER BY id;

-- 向量搜索查询
SELECT id, content, metadata,
       cosineDistance(embedding, [0.1, 0.2, ...]) as similarity
FROM hybrid_search_store
ORDER BY similarity ASC
LIMIT 5;

-- 全文搜索查询
SELECT id, content, metadata,
       ftsScore(content_fulltext, '优化 AND 数据库') as score
FROM hybrid_search_store
WHERE ftsMatch(content_fulltext, '优化 AND 数据库')
ORDER BY score DESC
LIMIT 5;

-- 混合搜索 (由 UnifiedRetriever 自动处理)
-- 结合向量相似度和全文相关性分数，按权重融合排序
        """, language="sql")

    with tab4:
        st.markdown("## 🔧 最佳实践与优化建议")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### ⚡ 搜索性能优化

            #### 🎛️ 参数调优
            - **Alpha权重**: 根据查询类型动态调整
              - 概念查询: 0.7-1.0 (偏向向量)
              - 关键词查询: 0.0-0.3 (偏向全文)
              - 混合查询: 0.5-0.7 (平衡)
            - **返回数量**: 通常5-10个结果最优
            - **分词器选择**:
              - `ik`: 中文文档 (推荐)
              - `standard`: 英文文档
              - `keyword`: 精确匹配场景

            #### 🧠 向量优化
            - **模型选择**: text-embedding-v4 (最新)
            - **向量维度**: 1536维 (平衡精度和性能)
            - **批量处理**: 大量文档分批添加
            """)

        with col2:
            st.markdown("""
            ### 🔍 查询优化策略

            #### 📝 查询重写
            - **同义词扩展**: "优化" → "改进,提升,调优"
            - **查询补全**: "数据库" → "数据库性能优化"
            - **意图识别**: 自动判断查询类型

            #### 🎯 场景化配置
            - **FAQ搜索**: alpha=0.8 (语义优先)
            - **代码搜索**: alpha=0.2 (关键词优先)
            - **文档检索**: alpha=0.7 (混合最优)
            - **实体查找**: alpha=0.0 (精确匹配)

            #### 📊 结果优化
            - **去重策略**: 避免相似内容重复
            - **排序融合**: 综合多种相关性分数
            - **多样性**: 确保结果覆盖不同方面
            """)

        st.markdown("---")

        st.markdown("## 🎓 学习建议")

        st.markdown("""
        ### 📚 循序渐进的学习路径

        #### 🟢 初级阶段 (理解搜索模式)
        1. **体验三种模式**: 用同一查询测试向量、全文、混合搜索
        2. **观察结果差异**: 比较不同模式返回的文档内容
        3. **调整alpha参数**: 感受权重变化对结果的影响

        #### 🟡 中级阶段 (掌握参数调优)
        1. **分析查询类型**: 学会识别概念性vs关键词性查询
        2. **优化搜索策略**: 为不同场景选择最佳参数
        3. **监控性能指标**: 关注搜索延迟和结果质量

        #### 🔴 高级阶段 (企业级部署)
        1. **自定义分词器**: 针对专业领域优化分词
        2. **查询理解**: 实现智能查询重写和扩展
        3. **个性化搜索**: 基于用户历史优化搜索结果

        ### 📖 相关资源
        - **[ClickZetta 官方文档](https://www.yunqi.tech/documents/)**: 获取最新的平台功能和最佳实践
        - **[Elasticsearch指南](https://www.elastic.co/guide/)**: 深入了解全文搜索原理
        - **[Vector Search最佳实践](https://docs.langchain.com/docs/modules/indexes/vectorstores/)**: 向量搜索优化技巧
        """)

# 页面配置
# Main navigation for help documentation
if 'page_mode' not in st.session_state:
    st.session_state.page_mode = "main"

# Sidebar navigation
with st.sidebar:
    st.markdown("## 📋 导航菜单")
    page_selection = st.selectbox(
        "选择功能页面",
        ["🚀 混合搜索", "📚 学习指南"],
        key="hybrid_search_page_selection"
    )

    if page_selection == "📚 学习指南":
        st.session_state.page_mode = "help"
    else:
        st.session_state.page_mode = "main"

if st.session_state.page_mode == "help":
    show_help_documentation()
    st.stop()

# Original app header for main mode
UIComponents.render_app_header(
    app_config,
    "基于 ClickZetta HybridStore + UnifiedRetriever 的企业级混合搜索系统"
)

# Add educational info banner
st.info("""
🎯 **系统特色**:
• **⚖️ HybridStore**: 使用 `{table_name}` 表统一存储向量和全文索引
• **🔄 UnifiedRetriever**: 智能检索器，支持三种搜索模式的无缝切换
• **🎛️ 权重调节**: 通过alpha参数灵活控制向量搜索和全文搜索的比例

💡 **使用提示**: 点击侧边栏的"📚 学习指南"了解混合搜索的详细原理和调优技巧
""".format(table_name=app_config.get_vector_table_name("hybrid_search")))

# 渲染环境配置状态
env_config, env_file_exists, clickzetta_configured, dashscope_configured = UIComponents.render_env_config_status()

# 会话状态初始化
SessionManager.init_session_state({
    "search_history": [],
    "loaded_documents": [],
    "hybrid_store": None,
    "retriever": None,
    "manager": None
})

# 侧边栏配置
with st.sidebar:
    st.header("🔧 系统配置")

    # 渲染环境配置详情
    UIComponents.render_env_config_sidebar(env_config, env_file_exists)

    # ClickZetta 配置
    clickzetta_config = render_clickzetta_config_form()

    # DashScope 配置
    dashscope_config = render_dashscope_config_form()

    # 配置状态
    render_config_status(clickzetta_config, dashscope_config)

    # 搜索设置
    st.header("🔍 搜索配置")

    search_mode = st.selectbox(
        "搜索模式",
        options=["hybrid", "vector", "fulltext"],
        index=0,
        help="选择搜索方式：混合搜索 (推荐)、仅向量搜索、仅全文搜索"
    )

    if search_mode == "hybrid":
        alpha = st.slider(
            "搜索权重平衡",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="0.0=仅全文搜索，1.0=仅向量搜索，0.7=向量搜索权重70%"
        )
    else:
        alpha = 1.0 if search_mode == "vector" else 0.0

    search_k = st.slider(
        "返回结果数量",
        min_value=1,
        max_value=20,
        value=5,
        help="每次搜索返回的文档数量"
    )

    text_analyzer = st.selectbox(
        "中文分词器",
        options=["ik", "standard", "keyword"],
        index=0,
        help="ik=智能分词 (推荐)，standard=标准分词，keyword=关键词分词"
    )

    # 文档管理
    st.header("📄 文档管理")
    uploaded_file = UIComponents.render_document_upload_area("hybrid_search_upload")

    # 数据管理
    st.header("🗑️ 数据管理")

    # 统计信息
    with st.expander("📊 统计信息"):
        if clickzetta_configured:
            try:
                from langchain_clickzetta import ClickZettaEngine
                from langchain_community.embeddings import DashScopeEmbeddings

                engine = ClickZettaEngine(
                    service=clickzetta_config.service,
                    username=clickzetta_config.username,
                    password=clickzetta_config.password,
                    instance=clickzetta_config.instance,
                    workspace=clickzetta_config.workspace,
                    schema=clickzetta_config.schema,
                    vcluster=clickzetta_config.vcluster if hasattr(clickzetta_config, 'vcluster') else None
                )

                table_name = app_config.get_vector_table_name("hybrid_search")

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
                        else:
                            st.warning("⚠️ 无法获取数据统计")
                    else:
                        st.info("📋 暂无数据表")

                except Exception as e:
                    st.warning(f"⚠️ 无法获取统计信息: {e}")

            except Exception as e:
                st.error(f"❌ 无法获取统计信息: {e}")
        else:
            st.warning("⚠️ 请先配置ClickZetta连接")

    # 清空数据功能
    with st.expander("🗑️ 数据清空"):
        st.write("**清空混合搜索数据**")
        st.caption("删除所有向量数据和搜索历史，重新开始")

        if st.button("🗑️ 清空所有数据", type="secondary", help="删除向量数据和搜索历史"):
            if clickzetta_configured:
                try:
                    from langchain_clickzetta import ClickZettaEngine

                    engine = ClickZettaEngine(
                        service=clickzetta_config.service,
                        username=clickzetta_config.username,
                        password=clickzetta_config.password,
                        instance=clickzetta_config.instance,
                        workspace=clickzetta_config.workspace,
                        schema=clickzetta_config.schema,
                        vcluster=clickzetta_config.vcluster if hasattr(clickzetta_config, 'vcluster') else None
                    )

                    # 清空混合搜索表
                    table_name = app_config.get_vector_table_name("hybrid_search")
                    delete_query = f"DELETE FROM {table_name}"
                    engine.execute_query(delete_query)

                    # 重置session状态
                    st.session_state.hybrid_store = None
                    st.session_state.retriever = None
                    st.session_state.loaded_documents = []
                    st.session_state.search_history = []

                    st.success("✅ 数据已清空，请重新上传文档")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 清空数据失败: {e}")
            else:
                st.warning("⚠️ 请先配置ClickZetta连接")

    # 管理按钮
    if st.button("🗑️ 清空搜索历史"):
        st.session_state.search_history = []
        st.success("搜索历史已清空")

    if st.button("🔄 重新加载文档"):
        st.session_state.loaded_documents = []
        st.session_state.hybrid_store = None
        st.session_state.retriever = None
        st.success("文档状态已重置")

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 混合搜索")

    # 配置验证
    config_valid, errors = ValidationHelper.validate_configs(clickzetta_config, dashscope_config)

    if not config_valid:
        st.error("配置错误：" + "，".join(errors))
        st.stop()

    # 初始化管理器
    if st.session_state.manager is None:
        try:
            st.session_state.manager = ClickZettaManager(clickzetta_config, dashscope_config)

            # 测试连接
            success, message = st.session_state.manager.test_connection()
            if success:
                st.success(message)

                # 自动检测并加载已有的混合搜索数据
                if not st.session_state.hybrid_store and not st.session_state.retriever:
                    try:
                        table_name = app_config.get_vector_table_name("hybrid_search")
                        # 先检查表是否存在
                        show_tables_query = f"SHOW TABLES LIKE '{table_name}'"
                        tables_result, _ = st.session_state.manager.engine.execute_query(show_tables_query)
                        if tables_result and len(tables_result) > 0:
                            # 表存在，检查数据量
                            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                            count_result, _ = st.session_state.manager.engine.execute_query(count_query)
                            if count_result and len(count_result) > 0:
                                doc_count = count_result[0]['count']
                                if doc_count > 0:

                                    # 有数据则自动初始化混合存储
                                    embeddings = DashScopeEmbeddings(
                                        dashscope_api_key=st.session_state.manager.dashscope_config.api_key,
                                        model="text-embedding-v4"
                                    )
                                    st.session_state.hybrid_store = ClickZettaHybridStore(
                                        engine=st.session_state.manager.engine,
                                        embeddings=embeddings,
                                        table_name=table_name,
                                        text_analyzer="ik",
                                        distance_metric="cosine"
                                    )

                                    # 初始化检索器
                                    st.session_state.retriever = ClickZettaUnifiedRetriever(
                                        hybrid_store=st.session_state.hybrid_store,
                                        search_type="hybrid",
                                        alpha=0.7,
                                        k=5
                                    )

                                    st.info(f"🎉 自动加载混合搜索数据成功！已有 {doc_count} 条文档数据，可直接开始搜索")
                                    # 标记为已加载状态，使用模拟文档信息
                                    st.session_state.loaded_documents = [{
                                        "filename": "历史数据",
                                        "info": {
                                            "page_count": "已存在数据",
                                            "total_characters": doc_count,
                                            "file_name": "历史数据"
                                        },
                                        "processed_at": "已存在"
                                    }]
                    except Exception as e:
                        # 表不存在或查询失败，这是正常情况
                        pass
            else:
                st.error(message)
                st.stop()

        except Exception as e:
            st.error(f"初始化失败: {e}")
            st.stop()

    # 文档处理
    if uploaded_file and (not st.session_state.loaded_documents or
                         st.session_state.loaded_documents[0].get("filename") != uploaded_file.name):

        with st.spinner("🔄 正在处理文档..."):
            documents = DocumentProcessor.process_pdf(uploaded_file)

            if documents:
                # 创建混合存储
                table_name = app_config.get_vector_table_name("hybrid_search")

                # 使用text-embedding-v4模型
                embeddings = DashScopeEmbeddings(
                    dashscope_api_key=st.session_state.manager.dashscope_config.api_key,
                    model="text-embedding-v4"
                )

                st.session_state.hybrid_store = ClickZettaHybridStore(
                    engine=st.session_state.manager.engine,
                    embeddings=embeddings,
                    table_name=table_name,
                    text_analyzer=text_analyzer,
                    distance_metric="cosine"
                )

                # 添加文档
                st.session_state.hybrid_store.add_documents(documents)

                # 创建检索器
                st.session_state.retriever = ClickZettaUnifiedRetriever(
                    hybrid_store=st.session_state.hybrid_store,
                    search_type=search_mode,
                    alpha=alpha,
                    k=search_k
                )

                # 保存文档信息
                doc_info = DocumentProcessor.get_document_info(documents)
                st.session_state.loaded_documents = [{
                    "filename": uploaded_file.name,
                    "info": doc_info,
                    "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }]

                st.success(f"✅ 文档处理完成: {doc_info['page_count']} 页，{doc_info['total_characters']:,} 字符")

    # 搜索界面
    if st.session_state.retriever:
        # 搜索输入
        query = st.text_input(
            "🔍 搜索查询",
            placeholder="输入您要搜索的内容...",
            help="支持自然语言查询，会同时进行向量搜索和全文搜索"
        )

        col_search, col_clear = st.columns([3, 1])

        with col_search:
            search_clicked = st.button("🚀 开始搜索", type="primary", use_container_width=True)

        with col_clear:
            if st.button("🧹 清空查询"):
                st.rerun()

        # 执行搜索
        if search_clicked and query.strip():
            with st.spinner("🔍 正在搜索..."):
                try:
                    # 更新检索器参数
                    st.session_state.retriever.search_type = search_mode
                    st.session_state.retriever.alpha = alpha
                    st.session_state.retriever.k = search_k

                    # 执行搜索
                    start_time = datetime.now()
                    results = st.session_state.retriever.invoke(query)
                    end_time = datetime.now()

                    search_time = (end_time - start_time).total_seconds()

                    # 保存搜索历史
                    search_record = {
                        "query": query,
                        "mode": search_mode,
                        "alpha": alpha,
                        "k": search_k,
                        "results_count": len(results),
                        "search_time": search_time,
                        "timestamp": end_time.strftime("%H:%M:%S")
                    }
                    st.session_state.search_history.insert(0, search_record)

                    # 显示搜索结果
                    st.markdown("---")
                    st.subheader(f"📋 搜索结果 ({len(results)} 条)")

                    if results:
                        # 搜索统计
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.metric("搜索模式", search_mode.upper())
                        with col_stats2:
                            st.metric("耗时", f"{search_time:.3f}s")
                        with col_stats3:
                            if search_mode == "hybrid":
                                st.metric("权重比例", f"向量{alpha:.1f}:全文{1-alpha:.1f}")
                            else:
                                st.metric("结果数量", f"{len(results)}")

                        # 结果展示
                        for i, doc in enumerate(results, 1):
                            with st.expander(f"📄 结果 {i}: {doc.page_content[:100]}..."):
                                st.write("**内容:**")
                                st.write(doc.page_content)

                                if doc.metadata:
                                    st.write("**元数据:**")
                                    st.json(doc.metadata)

                        # 生成智能摘要 (可选)
                        if st.button("🧠 生成智能摘要", key="generate_summary"):
                            with st.spinner("🤔 正在生成摘要..."):
                                try:
                                    # 合并搜索结果
                                    combined_content = "\n\n".join([doc.page_content for doc in results])

                                    # 生成摘要
                                    summary_prompt = f"""
                                    基于以下搜索结果，为用户查询"{query}"生成一个简洁的摘要回答：

                                    搜索结果：
                                    {combined_content}

                                    请提供一个准确、简洁的答案：
                                    """

                                    llm = st.session_state.manager.llm
                                    summary = llm.invoke(summary_prompt)

                                    st.success("🎯 智能摘要:")
                                    st.markdown(
                                        f"""
                                        <div style="
                                            background-color: #f0f2f6;
                                            padding: 15px;
                                            border-radius: 8px;
                                            border-left: 4px solid #1f77b4;
                                        ">
                                            {summary}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                except Exception as e:
                                    st.error(f"摘要生成失败: {e}")

                    else:
                        st.warning("🔍 未找到相关结果，请尝试调整搜索关键词或搜索模式")

                except Exception as e:
                    st.error(f"搜索失败: {e}")

        # 搜索历史
        if st.session_state.search_history:
            st.markdown("---")
            st.subheader("📚 搜索历史")

            for i, record in enumerate(st.session_state.search_history[:5]):  # 显示最近5条
                with st.expander(f"🕐 {record['timestamp']} - {record['query'][:50]}..."):
                    col_info1, col_info2 = st.columns(2)

                    with col_info1:
                        st.write(f"**查询**: {record['query']}")
                        st.write(f"**模式**: {record['mode']}")
                        st.write(f"**结果数**: {record['results_count']}")

                    with col_info2:
                        st.write(f"**耗时**: {record['search_time']:.3f}s")
                        if record['mode'] == 'hybrid':
                            st.write(f"**权重**: 向量{record['alpha']:.1f}")
                        st.write(f"**时间**: {record['timestamp']}")

    else:
        st.info("📋 请先上传文档，然后开始搜索")

with col2:
    st.subheader("📊 系统状态")

    # 连接状态
    if st.session_state.manager:
        st.success("🟢 ClickZetta 已连接")
        st.success("🟢 DashScope 已配置")
    else:
        st.error("🔴 系统未初始化")

    # 文档状态
    if st.session_state.loaded_documents:
        doc_info = st.session_state.loaded_documents[0]["info"]
        st.success("🟢 文档已加载")

        st.metric("📄 页数", doc_info.get("page_count", "N/A"))
        st.metric("📝 字符数", f"{doc_info.get('total_characters', 0):,}")
        if "avg_chars_per_page" in doc_info:
            st.metric("📊 平均页长", doc_info["avg_chars_per_page"])
        else:
            # 计算平均页长
            page_count = doc_info.get("page_count", 0)
            total_chars = doc_info.get("total_characters", 0)
            if isinstance(page_count, int) and page_count > 0:
                avg_chars = total_chars // page_count
                st.metric("📊 平均页长", f"{avg_chars} 字符/页")
            else:
                st.metric("📊 平均页长", "N/A")
    else:
        st.error("🔴 未加载文档")

    # 搜索状态
    if st.session_state.retriever:
        st.success("🟢 搜索引擎就绪")
    else:
        st.error("🔴 搜索引擎未就绪")

    # 实时统计
    if st.session_state.search_history:
        st.subheader("📈 搜索统计")

        total_searches = len(st.session_state.search_history)
        avg_time = sum(r["search_time"] for r in st.session_state.search_history) / total_searches

        st.metric("搜索次数", total_searches)
        st.metric("平均耗时", f"{avg_time:.3f}s")

        # 搜索模式分布
        mode_counts = {}
        for record in st.session_state.search_history:
            mode = record["mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        st.write("**搜索模式分布:**")
        for mode, count in mode_counts.items():
            st.write(f"  {mode}: {count} 次")

    # 高级功能
    st.subheader("🚀 高级功能")

    if st.button("📥 导出搜索历史", disabled=not st.session_state.search_history):
        if st.session_state.search_history:
            # 准备导出数据
            export_data = {
                "export_time": datetime.now().isoformat(),
                "total_searches": len(st.session_state.search_history),
                "search_history": st.session_state.search_history
            }

            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

            st.download_button(
                label="📋 下载 JSON 文件",
                data=json_str,
                file_name=f"search_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    if st.button("🔧 检索器配置详情"):
        if st.session_state.retriever:
            st.json({
                "search_type": st.session_state.retriever.search_type,
                "alpha": st.session_state.retriever.alpha,
                "k": st.session_state.retriever.k,
                "table_name": st.session_state.hybrid_store.table_name if st.session_state.hybrid_store else None,
                "text_analyzer": text_analyzer,
                "distance_metric": "cosine"
            })

    if st.button("🗄️ 查看存储表结构", disabled=not st.session_state.manager):
        if st.session_state.manager and st.session_state.hybrid_store:
            try:
                st.subheader("📊 ClickZetta HybridStore 表详情")

                table_name = st.session_state.hybrid_store.table_name
                st.write(f"**⚖️ HybridStore 表**: `{table_name}`")

                try:
                    # Get table schema - try multiple methods
                    schema_query = f"DESCRIBE TABLE EXTENDED {table_name}"
                    schema_result, schema_description = st.session_state.manager.engine.execute_query(schema_query)

                    if schema_result and len(schema_result) > 0:
                        st.write("**📋 表结构信息**:")
                        import pandas as pd

                        # 将结果转换为表格显示
                        try:
                            # 创建 DataFrame
                            df = pd.DataFrame(schema_result)

                            # 过滤掉空行和注释行
                            df = df[
                                (df['column_name'] != '') &
                                (~df['column_name'].str.startswith('#')) &
                                (df['column_name'].notna())
                            ]

                            # 重新排列列的顺序
                            if not df.empty:
                                df = df[['column_name', 'data_type', 'comment']]
                                df.columns = ['列名', '数据类型', '注释']
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.write("表结构数据为空")

                        except Exception as e:
                            # 如果创建表格失败，显示原始数据
                            st.write("表格创建失败，显示原始数据:")
                            for i, row in enumerate(schema_result):
                                if row.get('column_name') and not row.get('column_name').startswith('#'):
                                    st.write(f"**{row}**")

                        # Get record count
                        count_query = f"SELECT count(*) as total_documents FROM {table_name}"
                        count_result, _ = st.session_state.manager.engine.execute_query(count_query)
                        if count_result:
                            total_count = count_result[0]['total_documents']
                            st.metric("📄 存储的文档数", total_count)
                    else:
                        st.warning(f"⚠️ 表 `{table_name}` 不存在或为空。请先使用混合搜索功能添加一些文档。")

                        # Display search capabilities
                        st.markdown("**🔍 搜索能力说明**:")
                        st.markdown("""
                        - **向量搜索**: 使用 `embedding` 字段进行语义相似性检索
                        - **全文搜索**: 使用 `content_fulltext` 字段进行关键词匹配
                        - **混合搜索**: 结合两种方式，通过alpha权重调节比例
                        """)

                except Exception as e:
                    st.warning(f"表结构信息获取失败: {e}")

                st.write("**📖 更多信息**: 访问 [ClickZetta 官方文档](https://www.yunqi.tech/documents/) 了解HybridStore详细功能")

            except Exception as e:
                st.error(f"数据库连接错误: {e}")

    # 性能建议
    st.subheader("💡 性能建议")

    if st.session_state.search_history:
        recent_searches = st.session_state.search_history[:10]
        avg_recent_time = sum(r["search_time"] for r in recent_searches) / len(recent_searches)

        if avg_recent_time > 2.0:
            st.warning("⚠️ 搜索响应较慢，建议减少返回结果数量")
        elif avg_recent_time < 0.5:
            st.success("✅ 搜索性能优秀")
        else:
            st.info("ℹ️ 搜索性能良好")

# 页脚
UIComponents.render_footer()