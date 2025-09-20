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

# Helper function to show educational help documentation
def show_help_documentation():
    """显示详细的帮助文档"""
    st.markdown("# 📚 ClickZetta SQL智能问答系统 - 学习指南")

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

        **ClickZetta SQL智能问答系统** 是一个革命性的自然语言转SQL查询平台，基于 **SQLChain + ChatMessageHistory** 架构，让数据查询变得像聊天一样简单。

        #### 🔍 主要特点：
        - **🤖 SQLChain**: 智能SQL生成链，将自然语言转换为标准SQL查询
        - **💬 ChatMessageHistory**: 对话记忆功能，支持上下文相关的连续查询
        - **📊 数据库探索**: 自动获取表结构，智能推荐查询建议
        - **📈 结果可视化**: 查询结果的表格展示和CSV导出功能
        - **🔍 实时监控**: 详细的查询统计和性能分析
        """)

        st.markdown("---")

        st.markdown("## 🆚 传统SQL vs 智能问答对比")

        # Traditional vs AI comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 📝 传统SQL查询方式
            **挑战**:
            - 😰 需要熟悉SQL语法和函数
            - 🗃️ 必须了解数据库表结构
            - ⏰ 编写复杂查询耗时较长
            - 🐛 容易出现语法错误
            - 📚 学习成本高，门槛高

            **示例**:
            ```sql
            SELECT p.product_name,
                   SUM(o.quantity * o.price) as revenue
            FROM orders o
            JOIN products p ON o.product_id = p.id
            WHERE o.order_date >= '2023-01-01'
            GROUP BY p.product_name
            ORDER BY revenue DESC
            LIMIT 10;
            ```
            """)

        with col2:
            st.markdown("""
            #### 🤖 智能问答查询方式
            **优势**:
            - 😊 使用自然语言，无需SQL知识
            - 🎯 系统自动理解表结构关系
            - ⚡ 快速生成准确的SQL语句
            - 🛡️ 自动语法检查和优化
            - 📖 零学习成本，人人可用

            **示例**:
            ```
            用户输入: "查询销售额最高的前10个产品"

            系统自动生成:
            ✅ 分析表结构
            ✅ 生成SQL查询
            ✅ 执行并展示结果
            ✅ 记住对话上下文
            ```
            """)

        st.markdown("---")

        st.markdown("## 🏢 企业应用场景")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 📊 业务分析
            - **销售报表**: "统计每个月的销售额变化趋势"
            - **客户分析**: "找出购买最频繁的前20个客户"
            - **产品洞察**: "哪些产品的退货率最高？"
            - **区域统计**: "各个地区的订单分布情况"
            """)

            st.markdown("""
            #### 🔍 运营监控
            - **实时指标**: "今天新增了多少用户？"
            - **异常检测**: "查找异常高价的订单记录"
            - **库存管理**: "库存量低于100的产品有哪些？"
            - **性能分析**: "响应时间最慢的API接口"
            """)

        with col2:
            st.markdown("""
            #### 💼 决策支持
            - **财务分析**: "计算各部门的成本占比"
            - **人力资源**: "统计员工年龄和工作年限分布"
            - **市场调研**: "分析不同价格区间的产品销量"
            - **风险评估**: "识别高风险客户和订单"
            """)

            st.markdown("""
            #### 🎓 非技术人员赋能
            - **管理层**: 无需技术背景即可查询关键指标
            - **业务人员**: 自助式数据分析和报表生成
            - **客服团队**: 快速查询客户订单和历史记录
            - **财务人员**: 灵活的财务数据查询和分析
            """)

    with tab2:
        st.markdown("## 🏗️ 技术架构深度解析")

        # Architecture diagram
        st.markdown("""
        ### 📐 Text-to-SQL 架构图

        ```
        自然语言查询
              ↓
        ┌─────────────────────┐
        │   查询理解与解析     │ ← NLP处理层
        │   (Query Analysis)  │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │ ClickZetta          │ ← 元数据获取层
        │ Schema Inspector    │
        │ 表结构自动发现       │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │ ClickZetta          │ ← 智能SQL生成层
        │ SQLChain            │
        │ AI驱动的SQL生成     │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │   SQL执行引擎        │ ← 查询执行层
        │   (Query Execution) │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │ ClickZetta          │ ← 对话记忆层
        │ ChatMessageHistory  │
        │ 上下文记忆管理       │
        └─────────────────────┘
              ↓
        ┌─────────────────────┐
        │   结果展示与导出     │ ← 可视化层
        │   (Result Display)  │
        └─────────────────────┘
        ```
        """)

        st.markdown("---")

        st.markdown("## 🗄️ ClickZetta 存储组件详解")

        # Dual component explanation
        st.markdown("""
        ### 🤖 SQLChain + 💬 ChatMessageHistory - 智能SQL双引擎

        本应用融合了两个核心ClickZetta组件，实现完整的智能SQL问答体验：
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 🤖 SQLChain (SQL智能生成链)
            **类比**: 像一个**精通SQL的数据库专家**
            - 🧠 理解自然语言查询意图
            - 📊 自动分析数据库表结构
            - ⚡ 生成优化的SQL查询语句
            - 🛡️ 内置SQL安全检查机制
            """)

        with col2:
            st.markdown("""
            #### 💬 ChatMessageHistory (对话记忆)
            **类比**: 像一个**记忆力超群的助手**
            - 💾 记住所有历史对话内容
            - 🔄 支持上下文相关的连续查询
            - 📝 智能关联前后查询关系
            - 🎯 提供个性化查询建议
            """)

        st.markdown("""
        #### 🔧 技术特性对比

        | 特性 | SQLChain | ChatMessageHistory |
        |------|----------|-------------------|
        | **核心功能** | 自然语言→SQL转换 | 对话上下文管理 |
        | **输入处理** | 自然语言查询 | 历史对话记录 |
        | **输出结果** | 标准SQL语句+结果 | 上下文相关建议 |
        | **主要优势** | 智能理解+准确生成 | 连续对话+记忆 |
        | **存储表** | 无独立存储 | `{chat_table}` |
        | **应用场景** | 单次查询转换 | 多轮对话支持 |
        """.format(chat_table=app_config.get_chat_table_name("sql_chat")))

        st.markdown("---")

        st.markdown("## 🔄 Text-to-SQL 工作流程")

        # Text-to-SQL workflow
        st.markdown("""
        ### 🤖 智能SQL生成完整流程

        #### 1️⃣ 自然语言理解阶段
        ```python
        # 用户输入自然语言查询
        user_query = "查询销售额最高的前10个产品"

        # SQLChain分析查询意图
        # - 识别查询类型: SELECT查询
        # - 提取关键信息: 销售额、产品、排序、限制数量
        # - 确定聚合需求: SUM计算、ORDER BY排序
        ```

        #### 2️⃣ 数据库结构分析阶段
        ```python
        # 自动获取相关表结构
        table_info = engine.get_table_info(
            table_names=['products', 'orders', 'order_items'],
            schema=target_schema
        )

        # 分析表关系
        # - products表: 产品信息
        # - orders表: 订单信息
        # - order_items表: 订单详情(数量、价格)
        ```

        #### 3️⃣ SQL生成与优化阶段
        ```python
        # SQLChain智能生成SQL
        sql_chain = ClickZettaSQLChain.from_engine(
            engine=engine,
            llm=tongyi_llm,
            return_sql=True
        )

        # 生成优化的SQL语句
        result = sql_chain.invoke({"query": user_query})
        ```

        #### 4️⃣ 查询执行与结果处理阶段
        ```python
        # 执行生成的SQL
        results = engine.execute_query(generated_sql)

        # 格式化结果为用户友好的回答
        formatted_answer = format_query_results(results)
        ```

        #### 5️⃣ 对话记忆更新阶段
        ```python
        # 保存到对话历史
        chat_memory.save_context(
            {"input": user_query},
            {"output": formatted_answer, "sql": generated_sql}
        )
        ```
        """)

    with tab3:
        st.markdown("## 💡 核心代码示例")

        st.markdown("### 🔧 SQLChain + ChatMessageHistory 初始化")

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

# 2. 通义千问语言模型配置
llm = Tongyi(
    dashscope_api_key="your-api-key",
    model_name="qwen-plus",
    temperature=0.1                    # SQL生成需要低创造性
)

# 3. SQLChain 初始化 (核心组件)
sql_chain = ClickZettaSQLChain.from_engine(
    engine=engine,
    llm=llm,
    return_sql=True,                   # 返回生成的SQL语句
    top_k=100,                         # 限制结果数量
    verbose=True                       # 显示生成过程
)

# 4. ChatMessageHistory 初始化 (对话记忆)
chat_memory = ClickZettaChatMessageHistory(
    engine=engine,
    session_id="user-session-id",
    table_name="sql_chat_history"     # 对话记录表
)
        """, language="python")

        st.markdown("---")

        st.markdown("### 🎯 智能SQL问答流程")

        st.code("""
# 完整的Text-to-SQL查询流程
def intelligent_sql_query(user_question: str) -> dict:
    # 1. 构建查询输入
    query_input = {
        "query": user_question,
        "chat_history": chat_memory.buffer  # 添加对话上下文
    }

    # 2. 执行SQLChain生成SQL
    response = sql_chain.invoke(query_input)

    # 3. 解析响应结果
    generated_sql = response.get("sql_query", "")
    answer = response.get("answer", "")

    # 4. 执行SQL获取数据
    if generated_sql:
        results = engine.execute_query(generated_sql)
        df = pd.DataFrame(results)

    # 5. 更新对话记忆
    chat_memory.save_context(
        {"input": user_question},
        {"output": answer, "sql": generated_sql}
    )

    return {
        "sql": generated_sql,
        "answer": answer,
        "data": df,
        "success": True
    }

# 使用示例
result = intelligent_sql_query("查询销售额最高的前10个产品")
print(f"生成SQL: {result['sql']}")
print(f"AI回答: {result['answer']}")
result['data'].head()  # 显示查询结果
        """, language="python")

        st.markdown("---")

        st.markdown("### 🔍 上下文对话示例")

        st.code("""
# 多轮对话示例，展示ChatMessageHistory的作用

# 第一轮对话
query1 = "查询2023年的销售总额"
result1 = intelligent_sql_query(query1)
# 生成SQL: SELECT SUM(amount) FROM sales WHERE year = 2023
# 记录到对话历史

# 第二轮对话 (基于上下文)
query2 = "按月份分组显示"
result2 = intelligent_sql_query(query2)
# SQLChain理解上下文，知道是对2023年销售额按月分组
# 生成SQL: SELECT MONTH(date), SUM(amount) FROM sales
#          WHERE year = 2023 GROUP BY MONTH(date)

# 第三轮对话 (继续上下文)
query3 = "哪个月最高？"
result3 = intelligent_sql_query(query3)
# 基于前面的月份销售额结果，找出最高的月份
# 生成SQL: SELECT MONTH(date), SUM(amount) as total FROM sales
#          WHERE year = 2023 GROUP BY MONTH(date)
#          ORDER BY total DESC LIMIT 1

# 对话记忆让连续查询变得自然流畅
        """, language="python")

        st.markdown("---")

        st.markdown("### 📊 数据表结构示例")

        st.code("""
-- ChatMessageHistory 表结构 (对话记录)
CREATE TABLE sql_chat_history (
    session_id String,            -- 会话唯一标识
    message_id String,            -- 消息唯一标识
    message_type String,          -- human/ai 消息类型
    content String,               -- 消息内容
    sql_query String,             -- 生成的SQL语句(如果有)
    timestamp DateTime,           -- 消息时间戳
    metadata String               -- 扩展元数据
) ENGINE = ReplicatedMergeTree()
ORDER BY (session_id, timestamp);

-- 业务数据表示例 (SQL查询目标)
CREATE TABLE products (
    id Int32,
    product_name String,
    category String,
    price Decimal(10,2),
    created_at DateTime
) ENGINE = MergeTree()
ORDER BY id;

CREATE TABLE orders (
    id Int32,
    customer_id Int32,
    order_date Date,
    total_amount Decimal(10,2),
    status String
) ENGINE = MergeTree()
ORDER BY id;

-- 常用查询示例
-- 1. 获取对话历史
SELECT message_type, content, sql_query, timestamp
FROM sql_chat_history
WHERE session_id = 'session-123'
ORDER BY timestamp;

-- 2. 业务分析查询 (AI自动生成)
SELECT p.category, SUM(o.total_amount) as revenue
FROM orders o
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE o.order_date >= '2023-01-01'
GROUP BY p.category
ORDER BY revenue DESC;
        """, language="sql")

    with tab4:
        st.markdown("## 🔧 最佳实践与优化建议")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### ⚡ 查询优化技巧

            #### 🎯 自然语言查询建议
            - **具体明确**: "查询2023年1月的销售数据"
              优于 "查询销售数据"
            - **指定表名**: "用户表中的活跃用户数量"
              优于 "活跃用户数量"
            - **明确条件**: "价格大于100元的产品"
              优于 "贵的产品"
            - **指定排序**: "按销量降序排列的前10个产品"

            #### 🧠 上下文对话技巧
            - **渐进式查询**: 先查大范围，再细化条件
            - **引用前面结果**: "在刚才的结果中找出..."
            - **保持话题连贯**: 避免频繁切换查询主题
            - **适时重置**: 新话题开始时重置会话
            """)

        with col2:
            st.markdown("""
            ### 🛡️ 安全与性能

            #### 🔐 SQL安全防护
            - **自动注入防护**: SQLChain内置SQL注入检测
            - **权限控制**: 基于数据库用户权限限制
            - **查询限制**: 自动添加LIMIT防止大数据查询
            - **敏感数据**: 避免在查询中暴露敏感信息

            #### 📊 性能优化策略
            - **索引优化**: 确保查询字段有适当索引
            - **结果限制**: 大表查询自动添加TOP N限制
            - **查询缓存**: 相同查询结果智能缓存
            - **并发控制**: 控制同时执行的查询数量

            #### 🔧 系统配置建议
            - **温度设置**: SQL生成使用低温度(0.1)
            - **上下文长度**: 保持适当的对话历史长度
            - **超时设置**: 配置合理的查询超时时间
            """)

        st.markdown("---")

        st.markdown("## 🎓 学习建议")

        st.markdown("""
        ### 📚 循序渐进的学习路径

        #### 🟢 初级阶段 (掌握基础查询)
        1. **简单查询**: 练习基础的数据查找和统计
        2. **表结构理解**: 熟悉数据库中的表关系
        3. **自然语言技巧**: 学会用清晰的语言表达查询需求

        #### 🟡 中级阶段 (复杂查询技能)
        1. **多表关联**: 掌握跨表查询和数据聚合
        2. **上下文对话**: 利用对话记忆进行连续查询
        3. **查询优化**: 学习如何让查询更准确高效

        #### 🔴 高级阶段 (企业级应用)
        1. **业务建模**: 将复杂业务需求转化为查询需求
        2. **性能调优**: 优化大数据量查询的性能
        3. **权限管理**: 设计多用户的安全访问策略

        ### 📖 相关资源
        - **[ClickZetta 官方文档](https://www.yunqi.tech/documents/)**: 获取最新的平台功能和最佳实践
        - **[SQL教程](https://www.w3schools.com/sql/)**: 深入了解SQL语言基础
        - **[LangChain SQL指南](https://docs.langchain.com/docs/use-cases/sql)**: Text-to-SQL最佳实践
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
        ["🚀 SQL问答", "📚 学习指南"],
        key="sql_chat_page_selection"
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
    "基于 ClickZetta SQLChain + ChatMessageHistory + 通义千问AI 的智能SQL问答系统"
)

# Add educational info banner
st.info("""
🎯 **系统特色**:
• **🤖 SQLChain**: 智能将自然语言转换为标准SQL查询，无需SQL知识
• **💬 ChatMessageHistory**: 使用 `{chat_table}` 表记住对话上下文，支持连续查询
• **📊 智能分析**: 自动获取数据库结构，生成优化的查询语句和结果展示

💡 **使用提示**: 点击侧边栏的"📚 学习指南"了解Text-to-SQL技术和对话式查询的详细原理
""".format(chat_table=app_config.get_chat_table_name("sql_chat")))

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

    if st.button("🗄️ 查看数据库表结构", disabled=not st.session_state.manager):
        if st.session_state.manager and st.session_state.table_info:
            try:
                st.subheader("📊 ClickZetta 数据库详情")

                schema_name = st.session_state.table_info["schema"]
                tables = st.session_state.table_info["tables"]

                st.write(f"**📋 数据库模式**: `{schema_name}`")
                st.write(f"**📊 可用表数量**: {len(tables)}")

                # Show tables with details
                for table in tables[:10]:  # Show first 10 tables
                    try:
                        st.write(f"**📋 表**: `{table}`")

                        # Get table schema
                        schema_query = f"DESCRIBE TABLE {schema_name}.{table}"
                        schema_result = st.session_state.manager.engine.execute_query(schema_query)

                        if schema_result:
                            schema_df = pd.DataFrame(schema_result.fetchall(),
                                                   columns=[desc[0] for desc in schema_result.description])
                            st.dataframe(schema_df, use_container_width=True)

                            # Get record count
                            try:
                                count_query = f"SELECT count(*) as total_records FROM {schema_name}.{table}"
                                count_result = st.session_state.manager.engine.execute_query(count_query)
                                if count_result:
                                    total_count = count_result.fetchone()[0]
                                    st.metric(f"📊 {table} 记录数", total_count)
                            except:
                                st.caption("无法获取记录数")

                        st.markdown("---")

                    except Exception as e:
                        st.warning(f"表 {table} 信息获取失败: {e}")

                st.markdown("**🔍 SQL功能说明**:")
                st.markdown("""
                - **SQLChain**: 自动分析这些表结构，生成准确的SQL查询
                - **智能关联**: 理解表之间的关系，支持多表联查
                - **上下文记忆**: ChatMessageHistory记录查询历史，支持连续对话
                """)

                st.write("**📖 更多信息**: 访问 [ClickZetta 官方文档](https://www.yunqi.tech/documents/) 了解SQLChain详细功能")

            except Exception as e:
                st.error(f"数据库连接错误: {e}")

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