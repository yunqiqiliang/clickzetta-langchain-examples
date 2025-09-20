import streamlit as st
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import json
import pandas as pd

# 添加父目录到路径以导入通用组件
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_clickzetta import (
    ClickZettaEngine,
    ClickZettaSQLChain,
    ClickZettaChatMessageHistory
)
from langchain_community.llms import Tongyi

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
    UIComponents,
    SessionManager,
    ValidationHelper
)

# 应用配置
app_config = load_app_config("sql_chat")

# 页面配置
UIComponents.render_app_header(
    app_config,
    "基于 ClickZetta 的 SQL 智能问答系统，支持自然语言转 SQL 查询"
)

# 渲染环境配置状态
env_config, env_file_exists, clickzetta_configured, dashscope_configured = UIComponents.render_env_config_status()

# 会话状态初始化
SessionManager.init_session_state({
    "sql_history": [],
    "table_info": None,
    "sql_chain": None,
    "chat_memory": None,
    "manager": None,
    "current_database": None
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

    # SQL 设置
    st.header("💾 数据库设置")

    target_schema = st.text_input(
        "目标模式",
        value=clickzetta_config.schema,
        help="要查询的数据库模式名称"
    )

    include_sample_data = st.checkbox(
        "包含示例数据",
        value=True,
        help="在提示词中包含表的示例数据"
    )

    max_result_rows = st.slider(
        "最大结果行数",
        min_value=10,
        max_value=1000,
        value=100,
        help="SQL查询返回的最大行数"
    )

    # 高级设置
    st.header("⚙️ 高级设置")

    with st.expander("SQL 生成设置"):
        sql_temperature = st.slider(
            "SQL 生成创造性",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            help="控制 SQL 生成的创造性，建议保持较低值"
        )

        return_sql = st.checkbox(
            "返回 SQL 语句",
            value=True,
            help="在回答中显示生成的 SQL 语句"
        )

        use_memory = st.checkbox(
            "启用对话记忆",
            value=True,
            help="记住对话历史，支持上下文查询"
        )

    # 会话管理
    st.header("📝 会话管理")

    if st.button("🔄 新建会话"):
        SessionManager.reset_session()
        st.success("会话已重置")
        st.rerun()

    if st.button("🗑️ 清空 SQL 历史"):
        st.session_state.sql_history = []
        st.success("SQL 历史已清空")

    session_id = SessionManager.get_or_create_session_id()
    st.caption(f"会话ID: {session_id[:8]}...")

# 主界面
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 SQL 智能问答")

    # 配置验证
    config_valid, errors = ValidationHelper.validate_configs(clickzetta_config, dashscope_config)

    if not config_valid:
        st.error("配置错误：" + "，".join(errors))
        st.stop()

    # 初始化管理器和 SQL 链
    if st.session_state.manager is None:
        try:
            st.session_state.manager = ClickZettaManager(clickzetta_config, dashscope_config)

            # 测试连接
            success, message = st.session_state.manager.test_connection()
            if success:
                st.success(message)

                # 创建 SQL 链
                llm = Tongyi(
                    dashscope_api_key=dashscope_config.api_key,
                    model_name=dashscope_config.llm_model,
                    temperature=sql_temperature
                )

                st.session_state.sql_chain = ClickZettaSQLChain.from_engine(
                    engine=st.session_state.manager.engine,
                    llm=llm,
                    return_sql=return_sql,
                    top_k=max_result_rows
                )

                # 创建聊天记忆
                if use_memory:
                    chat_history = ClickZettaChatMessageHistory(
                        engine=st.session_state.manager.engine,
                        session_id=session_id,
                        table_name=app_config.get_chat_table_name("sql_chat")
                    )

                    # 使用简化的记忆管理 (避免弃用警告)
                    st.session_state.chat_memory = chat_history
                    st.session_state.memory_window = 10

                st.session_state.current_database = target_schema

            else:
                st.error(message)
                st.stop()

        except Exception as e:
            st.error(f"初始化失败: {e}")
            st.stop()

    # 数据库信息获取
    if st.session_state.manager and (st.session_state.table_info is None or
                                   st.session_state.current_database != target_schema):

        with st.spinner("📊 正在获取数据库信息..."):
            try:
                # 获取表名列表
                tables = st.session_state.manager.engine.get_table_names(schema=target_schema)

                if tables:
                    # 获取表结构信息
                    table_info = st.session_state.manager.engine.get_table_info(
                        table_names=tables[:20],  # 最多显示20个表
                        schema=target_schema
                    )

                    st.session_state.table_info = {
                        "schema": target_schema,
                        "tables": tables,
                        "table_info": table_info,
                        "updated_at": datetime.now()
                    }

                    st.session_state.current_database = target_schema
                    st.success(f"✅ 数据库信息已加载: {len(tables)} 个表")

                else:
                    st.warning(f"⚠️ 模式 '{target_schema}' 中未找到表")

            except Exception as e:
                st.error(f"❌ 获取数据库信息失败: {e}")

    # SQL 问答界面
    if st.session_state.sql_chain and st.session_state.table_info:

        # 显示数据库概览
        with st.expander("📊 数据库概览", expanded=False):
            db_info = st.session_state.table_info

            col_db1, col_db2, col_db3 = st.columns(3)
            with col_db1:
                st.metric("模式", db_info["schema"])
            with col_db2:
                st.metric("表数量", len(db_info["tables"]))
            with col_db3:
                st.metric("更新时间", db_info["updated_at"].strftime("%H:%M:%S"))

            # 表列表
            st.write("**可用表:**")
            table_cols = st.columns(4)
            for i, table in enumerate(db_info["tables"]):
                with table_cols[i % 4]:
                    st.write(f"• {table}")

        # 快速查询建议
        st.subheader("💡 快速查询建议")

        suggestion_cols = st.columns(2)

        with suggestion_cols[0]:
            if st.button("📋 显示所有表", use_container_width=True):
                st.session_state.suggested_query = "显示数据库中的所有表"

            if st.button("📊 统计表记录数", use_container_width=True):
                st.session_state.suggested_query = "统计每个表的记录数量"

        with suggestion_cols[1]:
            if st.button("🔍 查看表结构", use_container_width=True):
                tables = st.session_state.table_info["tables"]
                if tables:
                    st.session_state.suggested_query = f"描述 {tables[0]} 表的结构"

            if st.button("📈 数据分析查询", use_container_width=True):
                st.session_state.suggested_query = "找出数据量最大的前5个表"

        # 查询输入
        query_input = st.text_area(
            "🔍 请用自然语言描述您的查询需求",
            value=st.session_state.get("suggested_query", ""),
            height=100,
            placeholder="例如：查询销售额最高的前10个产品\n例如：统计每个月的订单数量\n例如：找出最活跃的用户",
            help="支持中文自然语言查询，系统会自动转换为SQL语句"
        )

        col_query1, col_query2 = st.columns([3, 1])

        with col_query1:
            execute_query = st.button("🚀 执行查询", type="primary", use_container_width=True)

        with col_query2:
            if st.button("🧹 清空输入"):
                # 清空建议查询
                if "suggested_query" in st.session_state:
                    del st.session_state.suggested_query
                st.rerun()

        # 执行查询
        if execute_query and query_input.strip():
            # 清除建议查询（只有在用户真正执行时才清除）
            if "suggested_query" in st.session_state:
                del st.session_state.suggested_query

            with st.spinner("🤔 正在分析查询并生成SQL..."):
                try:
                    start_time = datetime.now()

                    # 构建查询输入
                    query_input_dict = {"query": query_input.strip()}

                    # 添加记忆上下文
                    if st.session_state.chat_memory and use_memory:
                        query_input_dict["chat_history"] = st.session_state.chat_memory.buffer

                    # 执行 SQL 链
                    response = st.session_state.sql_chain.invoke(query_input_dict)

                    end_time = datetime.now()
                    execution_time = (end_time - start_time).total_seconds()

                    # 解析响应
                    if isinstance(response, dict):
                        sql_query = response.get("sql_query", "")
                        result = response.get("result", "")
                        answer = response.get("answer", result)
                    else:
                        sql_query = ""
                        answer = str(response)
                        result = answer

                    # 保存到历史记录
                    sql_record = {
                        "query": query_input.strip(),
                        "sql": sql_query,
                        "answer": answer,
                        "execution_time": execution_time,
                        "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "success": True
                    }

                    st.session_state.sql_history.insert(0, sql_record)

                    # 更新聊天记忆
                    if st.session_state.chat_memory and use_memory:
                        st.session_state.chat_memory.save_context(
                            {"input": query_input.strip()},
                            {"output": answer}
                        )

                    # 显示结果
                    st.markdown("---")
                    st.subheader("📋 查询结果")

                    # 执行统计
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("执行时间", f"{execution_time:.3f}s")
                    with col_stat2:
                        st.metric("查询状态", "✅ 成功")
                    with col_stat3:
                        st.metric("结果类型", "SQL 查询" if sql_query else "智能回答")

                    # 显示 SQL 语句
                    if sql_query and return_sql:
                        st.write("**生成的 SQL 语句:**")
                        st.code(sql_query, language="sql")

                    # 显示回答
                    st.write("**查询回答:**")
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f0f2f6;
                            padding: 15px;
                            border-radius: 8px;
                            border-left: 4px solid #1f77b4;
                        ">
                            {answer}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # 尝试解析数据结果
                    if sql_query and "SELECT" in sql_query.upper():
                        try:
                            # 直接执行 SQL 获取结构化数据
                            results, columns = st.session_state.manager.engine.execute_query(sql_query)

                            if results:
                                st.write("**数据结果:**")

                                # 转换为 DataFrame
                                df = pd.DataFrame(results)

                                # 显示数据
                                st.dataframe(df, use_container_width=True)

                                # 提供下载选项
                                if len(results) > 0:
                                    csv_data = df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 下载 CSV",
                                        data=csv_data,
                                        file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )

                                # 数据统计
                                st.caption(f"共 {len(results)} 行 {len(columns)} 列")

                        except Exception as e:
                            st.warning(f"⚠️ 数据解析失败: {e}")

                except Exception as e:
                    # 错误记录
                    error_record = {
                        "query": query_input.strip(),
                        "sql": "",
                        "answer": f"查询失败: {str(e)}",
                        "execution_time": 0,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "success": False
                    }

                    st.session_state.sql_history.insert(0, error_record)
                    st.error(f"❌ 查询执行失败: {e}")

        # SQL 历史记录
        if st.session_state.sql_history:
            st.markdown("---")
            st.subheader("📚 查询历史")

            for i, record in enumerate(st.session_state.sql_history[:5]):  # 显示最近5条
                status_icon = "✅" if record["success"] else "❌"

                with st.expander(f"{status_icon} {record['timestamp']} - {record['query'][:50]}..."):
                    col_hist1, col_hist2 = st.columns([2, 1])

                    with col_hist1:
                        st.write(f"**查询**: {record['query']}")
                        if record['sql']:
                            st.write("**SQL**:")
                            st.code(record['sql'], language="sql")
                        st.write(f"**回答**: {record['answer']}")

                    with col_hist2:
                        st.metric("执行时间", f"{record['execution_time']:.3f}s")
                        st.metric("状态", "成功" if record["success"] else "失败")
                        st.write(f"**时间**: {record['timestamp']}")

    else:
        st.info("📋 请完成配置并连接数据库，然后开始 SQL 问答")

with col2:
    st.subheader("📊 系统状态")

    # 连接状态
    if st.session_state.manager:
        st.success("🟢 ClickZetta 已连接")
        st.success("🟢 DashScope 已配置")
        if st.session_state.sql_chain:
            st.success("🟢 SQL 链已就绪")
        else:
            st.error("🔴 SQL 链未就绪")
    else:
        st.error("🔴 系统未初始化")

    # 数据库状态
    if st.session_state.table_info:
        db_info = st.session_state.table_info
        st.success("🟢 数据库已连接")

        st.metric("当前模式", db_info["schema"])
        st.metric("可用表数", len(db_info["tables"]))
        st.metric("信息更新", db_info["updated_at"].strftime("%H:%M:%S"))
    else:
        st.error("🔴 数据库信息未加载")

    # 记忆状态
    if st.session_state.chat_memory and use_memory:
        st.success("🟢 对话记忆已启用")
    elif use_memory:
        st.warning("⚠️ 对话记忆启用中")
    else:
        st.info("ℹ️ 对话记忆已禁用")

    # 查询统计
    if st.session_state.sql_history:
        st.subheader("📈 查询统计")

        total_queries = len(st.session_state.sql_history)
        successful_queries = sum(1 for r in st.session_state.sql_history if r["success"])
        success_rate = (successful_queries / total_queries) * 100

        st.metric("总查询数", total_queries)
        st.metric("成功率", f"{success_rate:.1f}%")

        if successful_queries > 0:
            avg_time = sum(r["execution_time"] for r in st.session_state.sql_history if r["success"]) / successful_queries
            st.metric("平均响应时间", f"{avg_time:.3f}s")

    # 表信息快速查看
    if st.session_state.table_info:
        st.subheader("📋 数据库表")

        tables = st.session_state.table_info["tables"]
        selected_table = st.selectbox(
            "选择表查看详情",
            options=[""] + tables,
            help="选择表名查看结构信息"
        )

        if selected_table:
            if st.button(f"🔍 查看 {selected_table} 表结构"):
                query = f"描述 {selected_table} 表的结构和字段信息"
                st.session_state.suggested_query = query
                st.rerun()

    # 高级功能
    st.subheader("🚀 高级功能")

    if st.button("📥 导出查询历史", disabled=not st.session_state.sql_history):
        if st.session_state.sql_history:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "session_id": session_id,
                "total_queries": len(st.session_state.sql_history),
                "sql_history": st.session_state.sql_history
            }

            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)

            st.download_button(
                label="📋 下载 JSON",
                data=json_str,
                file_name=f"sql_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    if st.button("🔄 刷新数据库信息"):
        st.session_state.table_info = None
        st.rerun()

    # 性能提示
    st.subheader("💡 使用提示")

    if st.session_state.sql_history:
        recent_failures = [r for r in st.session_state.sql_history[:10] if not r["success"]]

        if len(recent_failures) > 3:
            st.warning("⚠️ 最近查询失败较多，建议检查查询语法")
        elif any(r["execution_time"] > 5.0 for r in st.session_state.sql_history[:5]):
            st.warning("⚠️ 查询响应较慢，建议优化查询条件")
        else:
            st.success("✅ 查询性能良好")

    st.markdown("""
    **查询技巧:**
    - 使用自然语言描述需求
    - 指定具体的表名更准确
    - 避免过于复杂的查询逻辑
    - 可以参考历史成功查询
    """)

# 页脚
UIComponents.render_footer()