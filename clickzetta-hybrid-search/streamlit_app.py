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

# 页面配置
UIComponents.render_app_header(
    app_config,
    "基于 ClickZetta 的企业级混合搜索系统，结合向量搜索和全文搜索"
)

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

                st.session_state.hybrid_store = ClickZettaHybridStore(
                    engine=st.session_state.manager.engine,
                    embeddings=st.session_state.manager.embeddings,
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

        st.metric("📄 页数", doc_info["page_count"])
        st.metric("📝 字符数", f"{doc_info['total_characters']:,}")
        st.metric("📊 平均页长", doc_info["avg_chars_per_page"])
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