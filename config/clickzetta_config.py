"""
ClickZetta LangChain Examples 统一配置管理
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import streamlit as st


@dataclass
class ClickZettaConfig:
    """ClickZetta 连接配置"""
    service: str
    instance: str
    workspace: str
    schema: str
    username: str
    password: str
    vcluster: str
    connection_timeout: int = 60
    query_timeout: int = 1800
    hints: Optional[Dict[str, Any]] = None

    @classmethod
    def from_env(cls) -> 'ClickZettaConfig':
        """从环境变量加载配置"""
        return cls(
            service=os.getenv("CLICKZETTA_SERVICE", ""),
            instance=os.getenv("CLICKZETTA_INSTANCE", ""),
            workspace=os.getenv("CLICKZETTA_WORKSPACE", ""),
            schema=os.getenv("CLICKZETTA_SCHEMA", ""),
            username=os.getenv("CLICKZETTA_USERNAME", ""),
            password=os.getenv("CLICKZETTA_PASSWORD", ""),
            vcluster=os.getenv("CLICKZETTA_VCLUSTER", ""),
            connection_timeout=int(os.getenv("CLICKZETTA_CONNECTION_TIMEOUT", "60")),
            query_timeout=int(os.getenv("CLICKZETTA_QUERY_TIMEOUT", "1800"))
        )

    @classmethod
    def from_streamlit(cls) -> 'ClickZettaConfig':
        """从 Streamlit 界面获取配置"""
        return cls(
            service=st.session_state.get("clickzetta_service", ""),
            instance=st.session_state.get("clickzetta_instance", ""),
            workspace=st.session_state.get("clickzetta_workspace", ""),
            schema=st.session_state.get("clickzetta_schema", ""),
            username=st.session_state.get("clickzetta_username", ""),
            password=st.session_state.get("clickzetta_password", ""),
            vcluster=st.session_state.get("clickzetta_vcluster", ""),
            connection_timeout=st.session_state.get("clickzetta_connection_timeout", 60),
            query_timeout=st.session_state.get("clickzetta_query_timeout", 1800)
        )

    def is_complete(self) -> bool:
        """检查配置是否完整"""
        return all([
            self.service, self.instance, self.workspace,
            self.schema, self.username, self.password, self.vcluster
        ])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        config_dict = {
            "service": self.service,
            "instance": self.instance,
            "workspace": self.workspace,
            "schema": self.schema,
            "username": self.username,
            "password": self.password,
            "vcluster": self.vcluster,
            "connection_timeout": self.connection_timeout,
            "query_timeout": self.query_timeout
        }
        if self.hints:
            config_dict["hints"] = self.hints
        return config_dict


@dataclass
class DashScopeConfig:
    """DashScope 模型配置"""
    api_key: str
    embedding_model: str = "text-embedding-v4"
    llm_model: str = "qwen-plus"
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> 'DashScopeConfig':
        """从环境变量加载配置"""
        return cls(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            embedding_model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4"),
            llm_model=os.getenv("DASHSCOPE_LLM_MODEL", "qwen-plus"),
            temperature=float(os.getenv("DASHSCOPE_TEMPERATURE", "0.1"))
        )

    @classmethod
    def from_streamlit(cls) -> 'DashScopeConfig':
        """从 Streamlit 界面获取配置"""
        return cls(
            api_key=st.session_state.get("dashscope_api_key", ""),
            embedding_model=st.session_state.get("dashscope_embedding_model", "text-embedding-v4"),
            llm_model=st.session_state.get("dashscope_llm_model", "qwen-plus"),
            temperature=st.session_state.get("dashscope_temperature", 0.1)
        )

    def is_valid(self) -> bool:
        """检查配置是否有效"""
        return bool(self.api_key.strip())


@dataclass
class AppConfig:
    """应用级配置"""
    app_name: str
    page_title: str
    page_icon: str
    layout: str = "wide"
    table_prefix: str = "langchain_"

    # 向量存储配置
    vector_table_suffix: str = "vectors"
    chat_history_table_suffix: str = "chat_history"
    search_k: int = 5
    distance_metric: str = "cosine"

    # 记忆配置
    memory_window: int = 10
    max_history_length: int = 100

    def get_vector_table_name(self, app_suffix: str) -> str:
        """获取向量表名"""
        return f"{self.table_prefix}{app_suffix}_{self.vector_table_suffix}"

    def get_chat_table_name(self, app_suffix: str) -> str:
        """获取聊天历史表名"""
        return f"{self.table_prefix}{app_suffix}_{self.chat_history_table_suffix}"


# 预定义的应用配置
APP_CONFIGS = {
    "summary": AppConfig(
        app_name="ClickZetta 文档智能摘要",
        page_title="ClickZetta Document Summary",
        page_icon="📄"
    ),
    "qa": AppConfig(
        app_name="ClickZetta 智能问答系统",
        page_title="ClickZetta Intelligent Q&A",
        page_icon="🤖"
    ),
    "hybrid_search": AppConfig(
        app_name="ClickZetta 混合搜索",
        page_title="ClickZetta Hybrid Search",
        page_icon="🔍"
    ),
    "sql_chat": AppConfig(
        app_name="ClickZetta SQL 智能问答",
        page_title="ClickZetta SQL Chat",
        page_icon="💬"
    )
}


def render_clickzetta_config_form(config: Optional[ClickZettaConfig] = None) -> ClickZettaConfig:
    """渲染 ClickZetta 配置表单"""
    if config is None:
        config = ClickZettaConfig.from_env()

    st.subheader("🔧 ClickZetta 连接配置")

    with st.expander("基础连接设置", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            service = st.text_input(
                "Service *",
                value=config.service,
                help="ClickZetta 服务地址",
                key="clickzetta_service"
            )
            workspace = st.text_input(
                "Workspace *",
                value=config.workspace,
                help="工作空间名称",
                key="clickzetta_workspace"
            )
            username = st.text_input(
                "Username *",
                value=config.username,
                help="登录用户名",
                key="clickzetta_username"
            )

        with col2:
            instance = st.text_input(
                "Instance *",
                value=config.instance,
                help="实例名称",
                key="clickzetta_instance"
            )
            schema = st.text_input(
                "Schema *",
                value=config.schema,
                help="数据库模式",
                key="clickzetta_schema"
            )
            password = st.text_input(
                "Password *",
                value=config.password,
                type="password",
                help="登录密码",
                key="clickzetta_password"
            )

        vcluster = st.text_input(
            "VCluster *",
            value=config.vcluster,
            help="虚拟集群名称",
            key="clickzetta_vcluster"
        )

    with st.expander("高级设置"):
        col1, col2 = st.columns(2)
        with col1:
            connection_timeout = st.number_input(
                "连接超时 (秒)",
                min_value=10,
                max_value=300,
                value=config.connection_timeout,
                key="clickzetta_connection_timeout"
            )
        with col2:
            query_timeout = st.number_input(
                "查询超时 (秒)",
                min_value=60,
                max_value=3600,
                value=config.query_timeout,
                key="clickzetta_query_timeout"
            )

    return ClickZettaConfig(
        service=service,
        instance=instance,
        workspace=workspace,
        schema=schema,
        username=username,
        password=password,
        vcluster=vcluster,
        connection_timeout=connection_timeout,
        query_timeout=query_timeout
    )


def render_dashscope_config_form(config: Optional[DashScopeConfig] = None) -> DashScopeConfig:
    """渲染 DashScope 配置表单"""
    if config is None:
        config = DashScopeConfig.from_env()

    st.subheader("🧠 DashScope 模型配置")

    api_key = st.text_input(
        "DashScope API Key *",
        value=config.api_key,
        type="password",
        help="从阿里云 DashScope 获取的 API 密钥",
        key="dashscope_api_key"
    )

    col1, col2 = st.columns(2)
    with col1:
        embedding_model = st.selectbox(
            "嵌入模型",
            options=["text-embedding-v4", "text-embedding-v3", "text-embedding-v2"],
            index=0 if config.embedding_model == "text-embedding-v4" else 1,
            help="用于文档向量化的嵌入模型",
            key="dashscope_embedding_model"
        )

    with col2:
        llm_model = st.selectbox(
            "语言模型",
            options=["qwen-plus", "qwen-turbo", "qwen-max", "qwen-long"],
            index=0 if config.llm_model == "qwen-plus" else (
                1 if config.llm_model == "qwen-turbo" else 2
            ),
            help="用于生成回答的大语言模型",
            key="dashscope_llm_model"
        )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=config.temperature,
        step=0.1,
        help="控制生成文本的随机性，0.0 最保守，1.0 最有创造性",
        key="dashscope_temperature"
    )

    return DashScopeConfig(
        api_key=api_key,
        embedding_model=embedding_model,
        llm_model=llm_model,
        temperature=temperature
    )


def render_config_status(clickzetta_config: ClickZettaConfig, dashscope_config: DashScopeConfig):
    """渲染配置状态"""
    st.subheader("📊 配置状态")

    col1, col2 = st.columns(2)

    with col1:
        if clickzetta_config.is_complete():
            st.success("✅ ClickZetta 配置完整")
        else:
            st.error("❌ ClickZetta 配置不完整")

    with col2:
        if dashscope_config.is_valid():
            st.success("✅ DashScope 配置有效")
        else:
            st.error("❌ DashScope 配置无效")

    # 详细状态
    with st.expander("详细状态信息"):
        st.write("**ClickZetta 连接状态:**")
        required_fields = ["service", "instance", "workspace", "schema", "username", "password", "vcluster"]
        for field in required_fields:
            value = getattr(clickzetta_config, field)
            if value:
                st.write(f"  ✅ {field}: 已配置")
            else:
                st.write(f"  ❌ {field}: 未配置")

        st.write("**DashScope 配置状态:**")
        if dashscope_config.api_key:
            st.write(f"  ✅ API Key: 已配置 (***{dashscope_config.api_key[-4:]})")
            st.write(f"  ✅ 嵌入模型: {dashscope_config.embedding_model}")
            st.write(f"  ✅ 语言模型: {dashscope_config.llm_model}")
            st.write(f"  ✅ Temperature: {dashscope_config.temperature}")
        else:
            st.write("  ❌ API Key: 未配置")


def load_app_config(app_type: str) -> AppConfig:
    """加载应用配置"""
    return APP_CONFIGS.get(app_type, APP_CONFIGS["summary"])


def save_config_to_env_file(clickzetta_config: ClickZettaConfig, dashscope_config: DashScopeConfig, file_path: str = ".env"):
    """保存配置到 .env 文件"""
    env_content = f"""# ClickZetta 配置
CLICKZETTA_SERVICE={clickzetta_config.service}
CLICKZETTA_INSTANCE={clickzetta_config.instance}
CLICKZETTA_WORKSPACE={clickzetta_config.workspace}
CLICKZETTA_SCHEMA={clickzetta_config.schema}
CLICKZETTA_USERNAME={clickzetta_config.username}
CLICKZETTA_PASSWORD={clickzetta_config.password}
CLICKZETTA_VCLUSTER={clickzetta_config.vcluster}
CLICKZETTA_CONNECTION_TIMEOUT={clickzetta_config.connection_timeout}
CLICKZETTA_QUERY_TIMEOUT={clickzetta_config.query_timeout}

# DashScope 配置
DASHSCOPE_API_KEY={dashscope_config.api_key}
DASHSCOPE_EMBEDDING_MODEL={dashscope_config.embedding_model}
DASHSCOPE_LLM_MODEL={dashscope_config.llm_model}
DASHSCOPE_TEMPERATURE={dashscope_config.temperature}
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(env_content)


def test_clickzetta_connection(config: ClickZettaConfig) -> tuple[bool, str]:
    """测试 ClickZetta 连接"""
    try:
        from langchain_clickzetta import ClickZettaEngine

        engine = ClickZettaEngine(**config.to_dict())
        results, columns = engine.execute_query("SELECT 1 as test")

        if results and results[0].get("test") == 1:
            return True, "✅ 连接测试成功"
        else:
            return False, "❌ 查询结果异常"

    except Exception as e:
        return False, f"❌ 连接失败: {str(e)}"


def test_dashscope_connection(config: DashScopeConfig) -> tuple[bool, str]:
    """测试 DashScope 连接"""
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        from langchain_community.llms import Tongyi

        # 测试嵌入模型
        embeddings = DashScopeEmbeddings(
            dashscope_api_key=config.api_key,
            model=config.embedding_model
        )

        # 简单测试
        test_text = ["测试文本"]
        embeddings.embed_documents(test_text)

        # 测试语言模型
        llm = Tongyi(
            dashscope_api_key=config.api_key,
            model_name=config.llm_model,
            temperature=config.temperature
        )

        response = llm.invoke("你好")

        if response:
            return True, "✅ DashScope 连接测试成功"
        else:
            return False, "❌ 模型响应异常"

    except Exception as e:
        return False, f"❌ DashScope 连接失败: {str(e)}"