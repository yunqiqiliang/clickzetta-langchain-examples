"""
ClickZetta LangChain Examples 通用组件库
"""

import streamlit as st
import tempfile
import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import uuid
from dotenv import load_dotenv

from langchain_clickzetta import (
    ClickZettaEngine,
    ClickZettaVectorStore,
    ClickZettaChatMessageHistory
)
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from config.clickzetta_config import ClickZettaConfig, DashScopeConfig, AppConfig


class ClickZettaManager:
    """ClickZetta 连接和组件管理器"""

    def __init__(self, clickzetta_config: ClickZettaConfig, dashscope_config: DashScopeConfig):
        self.clickzetta_config = clickzetta_config
        self.dashscope_config = dashscope_config
        self._engine = None
        self._embeddings = None
        self._llm = None

    @property
    def engine(self) -> ClickZettaEngine:
        """获取 ClickZetta 引擎实例"""
        if self._engine is None:
            self._engine = ClickZettaEngine(**self.clickzetta_config.to_dict())
        return self._engine

    @property
    def embeddings(self) -> DashScopeEmbeddings:
        """获取嵌入模型实例"""
        if self._embeddings is None:
            self._embeddings = DashScopeEmbeddings(
                dashscope_api_key=self.dashscope_config.api_key,
                model=self.dashscope_config.embedding_model
            )
        return self._embeddings

    @property
    def llm(self) -> Tongyi:
        """获取语言模型实例"""
        if self._llm is None:
            self._llm = Tongyi(
                dashscope_api_key=self.dashscope_config.api_key,
                model_name=self.dashscope_config.llm_model,
                temperature=self.dashscope_config.temperature
            )
        return self._llm

    def create_vector_store(self, table_name: str, distance_metric: str = "cosine") -> ClickZettaVectorStore:
        """创建向量存储"""
        return ClickZettaVectorStore(
            engine=self.engine,
            embeddings=self.embeddings,
            table_name=table_name,
            distance_metric=distance_metric
        )

    def create_chat_history(self, session_id: str, table_name: str) -> ClickZettaChatMessageHistory:
        """创建聊天历史"""
        return ClickZettaChatMessageHistory(
            engine=self.engine,
            session_id=session_id,
            table_name=table_name
        )

    def test_connection(self) -> Tuple[bool, str]:
        """测试连接"""
        try:
            results, columns = self.engine.execute_query("SELECT 1 as test")
            if results and results[0].get("test") == 1:
                return True, "✅ ClickZetta 连接成功"
            else:
                return False, "❌ 查询结果异常"
        except Exception as e:
            return False, f"❌ 连接失败: {str(e)}"


class DocumentProcessor:
    """文档处理器"""

    @staticmethod
    def process_pdf(uploaded_file) -> List[Document]:
        """处理上传的 PDF 文件"""
        if uploaded_file is None:
            return []

        try:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # 加载文档
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load_and_split()

            # 清理临时文件
            os.remove(tmp_file_path)

            return documents

        except Exception as e:
            st.error(f"❌ 文档处理失败: {str(e)}")
            return []

    @staticmethod
    def get_document_info(documents: List[Document]) -> Dict[str, Any]:
        """获取文档信息"""
        if not documents:
            return {}

        total_chars = sum(len(doc.page_content) for doc in documents)
        total_pages = len(documents)

        return {
            "page_count": total_pages,
            "total_characters": total_chars,
            "avg_chars_per_page": total_chars // total_pages if total_pages > 0 else 0,
            "first_page_preview": documents[0].page_content[:200] + "..." if documents else ""
        }


class UIComponents:
    """UI 组件库"""

    @staticmethod
    def load_env_config() -> Dict[str, str]:
        """从环境变量加载配置"""
        # 尝试从当前目录和父目录加载 .env 文件
        load_dotenv()  # 当前目录
        load_dotenv('../.env')  # 父目录
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

    @staticmethod
    def render_env_config_status():
        """渲染环境配置状态横幅"""
        env_config = UIComponents.load_env_config()

        # 检查配置完整性
        clickzetta_configured = all([
            env_config['clickzetta_service'], env_config['clickzetta_instance'],
            env_config['clickzetta_workspace'], env_config['clickzetta_schema'],
            env_config['clickzetta_username'], env_config['clickzetta_password'],
            env_config['clickzetta_vcluster']
        ])
        dashscope_configured = bool(env_config['dashscope_api_key'])

        # 检查 .env 文件是否存在（检查当前目录和父目录）
        env_file_exists = os.path.exists('.env') or os.path.exists('../.env')

        # 状态横幅
        col1, col2, col3 = st.columns(3)

        with col1:
            if env_file_exists:
                st.success("✅ 配置文件已加载")
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
                # 显示遮掩的 API Key
                api_key = env_config['dashscope_api_key']
                masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else api_key[:4] + "*" * 4
                st.caption(f"API Key: {masked_key}")
            else:
                st.warning("⚠️ DashScope API 未配置")

        st.markdown("---")
        return env_config, env_file_exists, clickzetta_configured, dashscope_configured

    @staticmethod
    def render_env_config_sidebar(env_config: Dict[str, str], env_file_exists: bool):
        """在侧边栏渲染环境配置详情"""
        st.subheader("📋 环境配置状态")

        # 显示配置文件状态和路径
        if env_file_exists:
            st.success("✅ 已加载 .env 配置文件")
            # 显示实际找到的 .env 文件路径
            if os.path.exists('.env'):
                st.caption("📁 位置: ./.env")
            elif os.path.exists('../.env'):
                st.caption("📁 位置: ../.env")
        else:
            st.warning("⚠️ 未找到 .env 配置文件")
            st.caption("💡 请在项目根目录创建 .env 文件")

        # 配置详情展开器
        with st.expander("📋 查看配置详情"):
            st.write("**ClickZetta 配置:**")
            st.write(f"• Service: `{env_config['clickzetta_service'] or '未配置'}`")
            st.write(f"• Instance: `{env_config['clickzetta_instance'] or '未配置'}`")
            st.write(f"• Workspace: `{env_config['clickzetta_workspace'] or '未配置'}`")
            st.write(f"• Schema: `{env_config['clickzetta_schema'] or '未配置'}`")
            st.write(f"• Username: `{env_config['clickzetta_username'] or '未配置'}`")
            st.write(f"• VCluster: `{env_config['clickzetta_vcluster'] or '未配置'}`")

            st.write("**DashScope 配置:**")
            st.write(f"• 嵌入模型: `{env_config['embedding_model']}`")
            st.write(f"• 语言模型: `{env_config['llm_model']}`")
            if env_config['dashscope_api_key']:
                masked_key = env_config['dashscope_api_key'][:8] + "****"
                st.write(f"• API Key: `{masked_key}`")
            else:
                st.write("• API Key: `未配置`")

    @staticmethod
    def render_app_header(app_config: AppConfig, subtitle: str = ""):
        """渲染应用标题"""
        st.set_page_config(
            page_title=app_config.page_title,
            page_icon=app_config.page_icon,
            layout=app_config.layout
        )

        st.title(f"{app_config.page_icon} {app_config.app_name}")
        if subtitle:
            st.markdown(f"*{subtitle}*")

    @staticmethod
    def render_connection_status(manager: ClickZettaManager) -> bool:
        """渲染连接状态并返回是否连接成功"""
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                success, message = manager.test_connection()
                if success:
                    st.success(message)
                else:
                    st.error(message)

            with col2:
                if st.button("🔄 重新测试", key="test_connection"):
                    st.rerun()

            return success

    @staticmethod
    def render_document_upload_area(key: str = "document_upload"):
        """渲染文档上传区域"""
        st.subheader("📄 文档管理")

        uploaded_file = st.file_uploader(
            "上传 PDF 文档",
            type=["pdf"],
            help="支持 PDF 格式，建议文件大小不超过 10MB",
            key=key
        )

        if uploaded_file:
            col1, col2 = st.columns([2, 1])

            with col1:
                st.info(f"📋 文件: {uploaded_file.name}")

            with col2:
                file_size_mb = uploaded_file.size / 1024 / 1024
                st.metric("文件大小", f"{file_size_mb:.1f} MB")

        return uploaded_file

    @staticmethod
    def render_processing_status(status_text: str, is_processing: bool = False):
        """渲染处理状态"""
        if is_processing:
            with st.spinner(status_text):
                st.empty()
        else:
            st.info(status_text)

    @staticmethod
    def render_chat_interface(messages: List[Tuple[str, str, str]], key: str = "chat"):
        """渲染聊天界面

        Args:
            messages: List of (role, message, timestamp) tuples
            key: Unique key for the chat input
        """
        # 显示历史消息
        if messages:
            st.subheader("💬 对话历史")
            for role, message, timestamp in messages:
                if role == "user":
                    st.chat_message("user").write(f"**{timestamp}**\n{message}")
                else:
                    st.chat_message("assistant").write(f"**{timestamp}**\n{message}")

        # 输入框
        return st.chat_input("请输入您的问题...", key=key)

    @staticmethod
    def render_statistics_panel(stats: Dict[str, Any]):
        """渲染统计面板"""
        st.subheader("📊 统计信息")

        # 根据统计数据动态生成列
        if not stats:
            st.info("暂无统计数据")
            return

        # 创建适当数量的列
        num_stats = len(stats)
        cols = st.columns(min(num_stats, 4))  # 最多4列

        for i, (key, value) in enumerate(stats.items()):
            with cols[i % len(cols)]:
                st.metric(key, value)

    @staticmethod
    def render_model_settings(dashscope_config: DashScopeConfig, key_prefix: str = ""):
        """渲染模型设置面板"""
        st.subheader("🧠 模型设置")

        col1, col2 = st.columns(2)

        with col1:
            embedding_model = st.selectbox(
                "嵌入模型",
                options=["text-embedding-v4", "text-embedding-v3"],
                index=0 if dashscope_config.embedding_model == "text-embedding-v4" else 1,
                key=f"{key_prefix}embedding_model"
            )

        with col2:
            llm_model = st.selectbox(
                "语言模型",
                options=["qwen-plus", "qwen-turbo", "qwen-max"],
                index=["qwen-plus", "qwen-turbo", "qwen-max"].index(dashscope_config.llm_model),
                key=f"{key_prefix}llm_model"
            )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=dashscope_config.temperature,
            step=0.1,
            help="控制回答的创造性，0.0 最保守，1.0 最有创造性",
            key=f"{key_prefix}temperature"
        )

        return embedding_model, llm_model, temperature

    @staticmethod
    def render_footer():
        """渲染页脚"""
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666;">
                <p>🚀 Powered by <strong>ClickZetta</strong> + <strong>DashScope</strong> + <strong>LangChain</strong></p>
                <p>💡 企业级向量数据库 + 智能语言模型 = 无限可能</p>
            </div>
            """,
            unsafe_allow_html=True
        )


class SessionManager:
    """会话管理器"""

    @staticmethod
    def init_session_state(defaults: Dict[str, Any]):
        """初始化会话状态"""
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def get_or_create_session_id() -> str:
        """获取或创建会话ID"""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id

    @staticmethod
    def reset_session():
        """重置会话"""
        keys_to_keep = ["clickzetta_config", "dashscope_config"]
        keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]

        for key in keys_to_remove:
            del st.session_state[key]

        st.session_state.session_id = str(uuid.uuid4())

    @staticmethod
    def add_chat_message(role: str, message: str) -> str:
        """添加聊天消息到会话状态"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append((role, message, timestamp))

        return timestamp


class ValidationHelper:
    """验证助手"""

    @staticmethod
    def validate_configs(clickzetta_config: ClickZettaConfig, dashscope_config: DashScopeConfig) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []

        if not clickzetta_config.is_complete():
            errors.append("ClickZetta 配置不完整")

        if not dashscope_config.is_valid():
            errors.append("DashScope API Key 未配置")

        return len(errors) == 0, errors

    @staticmethod
    def validate_file_upload(uploaded_file, max_size_mb: int = 10) -> Tuple[bool, str]:
        """验证文件上传"""
        if uploaded_file is None:
            return False, "请上传文件"

        file_size_mb = uploaded_file.size / 1024 / 1024
        if file_size_mb > max_size_mb:
            return False, f"文件大小超出限制 ({max_size_mb}MB)"

        if not uploaded_file.name.lower().endswith('.pdf'):
            return False, "仅支持 PDF 格式文件"

        return True, "文件验证通过"


# 辅助函数
def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_current_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."