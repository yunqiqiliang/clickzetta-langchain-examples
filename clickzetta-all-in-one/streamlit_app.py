"""
ClickZetta LangChain All-in-One Demo

集成展示所有 ClickZetta 存储服务和 LangChain 功能的综合演示平台。
一个页面体验所有功能：文档摘要、智能问答、混合搜索、SQL问答、网络爬虫。
"""

import streamlit as st
import os
import hashlib
import time
from datetime import datetime
import re
import json

# 辅助函数：清理控制字符
def clean_message_content(content):
    """清理消息内容中的JSON控制字符"""
    if not content:
        return content

    if isinstance(content, str):
        # 移除或替换控制字符，保留换行符和制表符
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        return cleaned
    return content

def safe_get_chat_history(chat_memory):
    """安全获取聊天历史，处理JSON控制字符错误"""
    try:
        return chat_memory.buffer if chat_memory else []
    except Exception as e:
        print(f"Failed to retrieve messages: {e}")  # 这是日志中看到的错误
        return []  # 返回空历史而不是崩溃

# 加载环境变量
from dotenv import load_dotenv

# 首先尝试加载当前目录的 .env，然后尝试父目录的 .env
env_loaded = load_dotenv('.env')
if not env_loaded:
    env_loaded = load_dotenv('../.env')
if not env_loaded:
    st.warning("⚠️ 未找到 .env 文件，请确保环境变量已正确配置")

try:
    import requests
    from bs4 import BeautifulSoup
    import html2text
    import validators
    import pandas as pd
    import plotly.express as px
    from langchain_core.documents import Document
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_clickzetta import (
        ClickZettaEngine,
        ClickZettaStore,
        ClickZettaDocumentStore,
        ClickZettaFileStore,
        ClickZettaVectorStore,
        ClickZettaHybridStore,
        ClickZettaChatMessageHistory
    )
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import Tongyi

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    st.error(f"缺少依赖: {e}")
    st.stop()

# Streamlit 页面配置
st.set_page_config(
    page_title="ClickZetta LangChain All-in-One Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 样式定义
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown("""
<div class="main-header">
    <h1>🚀 ClickZetta LangChain All-in-One Demo</h1>
    <p>一站式体验所有 ClickZetta 存储服务、检索服务与 LangChain 的集成</p>
</div>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'demo_data' not in st.session_state:
    st.session_state.demo_data = {}

def init_clickzetta_engine():
    """初始化 ClickZetta 引擎"""
    try:
        required_env_vars = [
            'CLICKZETTA_SERVICE',
            'CLICKZETTA_INSTANCE',
            'CLICKZETTA_WORKSPACE',
            'CLICKZETTA_SCHEMA',
            'CLICKZETTA_USERNAME',
            'CLICKZETTA_PASSWORD',
            'CLICKZETTA_VCLUSTER'
        ]

        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            st.error(f"缺少环境变量: {', '.join(missing_vars)}")
            st.info("请在 .env 文件中配置这些变量")
            return None

        engine = ClickZettaEngine(
            service=os.getenv('CLICKZETTA_SERVICE'),
            instance=os.getenv('CLICKZETTA_INSTANCE'),
            workspace=os.getenv('CLICKZETTA_WORKSPACE'),
            schema=os.getenv('CLICKZETTA_SCHEMA'),
            username=os.getenv('CLICKZETTA_USERNAME'),
            password=os.getenv('CLICKZETTA_PASSWORD'),
            vcluster=os.getenv('CLICKZETTA_VCLUSTER')
        )

        return engine

    except Exception as e:
        st.error(f"ClickZetta 引擎初始化失败: {e}")
        return None

def init_ai_services():
    """初始化 AI 服务"""
    try:
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            st.error("请设置 DASHSCOPE_API_KEY 环境变量")
            return None, None

        embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=api_key
        )

        llm = Tongyi(
            model_name="qwen-turbo",
            dashscope_api_key=api_key,
            temperature=0.7
        )

        return embeddings, llm

    except Exception as e:
        st.error(f"AI 服务初始化失败: {e}")
        return None, None


def chunk_text(text, max_length=2000):
    """将长文本分块，确保每块不超过指定长度"""
    if len(text) <= max_length:
        return [text]

    chunks = []
    # 按段落分割
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for paragraph in paragraphs:
        # 如果当前段落本身就超长，需要进一步分割
        if len(paragraph) > max_length:
            # 先处理当前积累的内容
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # 按句子分割长段落
            sentences = re.split(r'[。！？.!?]', paragraph)
            for sentence in sentences:
                if not sentence.strip():
                    continue
                sentence = sentence.strip() + '。'

                # 如果单个句子就超长，需要强制分割
                if len(sentence) > max_length:
                    # 强制按字符分割
                    while len(sentence) > max_length:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                        chunks.append(sentence[:max_length])
                        sentence = sentence[max_length:]
                    if sentence:
                        current_chunk = sentence
                elif len(current_chunk + sentence) <= max_length:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        else:
            # 正常段落处理
            if len(paragraph) > max_length:
                # 段落本身超长，需要特殊处理
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # 按句子分割长段落
                sentences = re.split(r'[。！？.!?]', paragraph)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    sentence = sentence.strip() + '。'

                    if len(sentence) > max_length:
                        # 句子还是太长，强制分割
                        while len(sentence) > max_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                            chunks.append(sentence[:max_length])
                            sentence = sentence[max_length:]
                        if sentence:
                            current_chunk = sentence
                    elif len(current_chunk + sentence) <= max_length:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
            elif len(current_chunk + paragraph + '\n\n') <= max_length:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def sanitize_content_for_sql(content):
    """
    对文档内容进行清理，避免 SQL 注入和解析错误
    """
    if not content:
        return content

    # 移除或替换可能导致SQL解析问题的字符
    sanitized = content

    # 替换空字节
    sanitized = sanitized.replace('\x00', '')

    # 处理引号 - 转义单引号和双引号
    sanitized = sanitized.replace("'", "''")
    sanitized = sanitized.replace('"', '""')

    # 处理反斜杠
    sanitized = sanitized.replace('\\', '\\\\')

    # 处理控制字符，但保留常见的换行符
    control_chars = ['\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08',
                    '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12', '\x13',
                    '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', '\x1b',
                    '\x1c', '\x1d', '\x1e', '\x1f']

    for char in control_chars:
        sanitized = sanitized.replace(char, '')

    # 限制内容长度，避免过长的文本导致SQL性能问题
    max_length = 100000  # 100KB 文本限制
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "...[内容被截断]"

    return sanitized


def process_long_text(text, llm, operation="摘要"):
    """处理长文本，支持分块处理"""
    chunks = chunk_text(text)

    if len(chunks) == 1:
        # 文本长度合适，直接处理
        st.write("📝 正在生成摘要...")
        if operation == "摘要":
            prompt = PromptTemplate(
                input_variables=["content"],
                template="""请对以下内容生成一个简洁准确的摘要，突出关键信息：

内容：
{content}

摘要："""
            )
        else:
            prompt = PromptTemplate(
                input_variables=["content"],
                template="""请分析以下内容：

内容：
{content}

分析："""
            )

        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(content=text)

    else:
        # 文本过长，需要分块处理
        st.write(f"📋 文本已分为 {len(chunks)} 个部分")

        # 优化策略提示
        if len(chunks) > 10:
            st.warning(f"⚠️ 检测到大型文档（{len(chunks)}个分块）。处理时间约需 {len(chunks) * 3} 秒。")

            # 提供快速摘要选项
            if st.checkbox("🚀 使用快速模式（合并相似段落）", value=True):
                # 智能合并相似长度的分块
                optimized_chunks = []
                current_batch = ""

                for chunk in chunks:
                    if len(current_batch + chunk) <= 2000:  # 确保不超过API限制
                        current_batch += "\n\n" + chunk if current_batch else chunk
                    else:
                        if current_batch:
                            optimized_chunks.append(current_batch)
                        current_batch = chunk

                if current_batch:
                    optimized_chunks.append(current_batch)

                chunks = optimized_chunks
                st.info(f"📊 优化后分为 {len(chunks)} 个部分（减少 {((len(chunk_text(text)) - len(chunks)) / len(chunk_text(text)) * 100):.0f}% 处理时间）")

        st.write(f"🔄 开始处理 {len(chunks)} 个部分...")
        chunk_results = []

        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        for i, chunk in enumerate(chunks):
            # 更新进度和时间估算
            progress = (i + 1) / len(chunks)
            progress_bar.progress(progress)

            if i > 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / i) * (len(chunks) - i)
                status_text.write(f"🔄 正在处理第 {i+1}/{len(chunks)} 部分 ({len(chunk)} 字符) - 预计剩余 {remaining:.0f} 秒")
            else:
                status_text.write(f"🔄 正在处理第 {i+1}/{len(chunks)} 部分 ({len(chunk)} 字符)...")

            if operation == "摘要":
                prompt = PromptTemplate(
                    input_variables=["content", "part"],
                    template="""请对以下内容的第{part}部分生成简洁摘要：

内容：
{content}

简洁摘要："""
                )
            else:
                prompt = PromptTemplate(
                    input_variables=["content", "part"],
                    template="""请简要分析以下内容的第{part}部分：

内容：
{content}

简要分析："""
                )

            # 安全检查：确保分块不超过API限制
            if len(chunk) > 2048:
                st.error(f"⚠️ 第{i+1}部分长度({len(chunk)}字符)超过API限制，跳过处理")
                chunk_results.append(f"[第{i+1}部分内容过长，已跳过]")
                continue

            chain = LLMChain(llm=llm, prompt=prompt)
            try:
                result = chain.run(content=chunk, part=f"{i+1}")
                chunk_results.append(result)
            except Exception as e:
                st.error(f"处理第{i+1}部分时出错: {e}")
                chunk_results.append(f"[第{i+1}部分处理失败: {str(e)}]")

        # 合并阶段
        progress_bar.progress(1.0)
        status_text.write("🔗 正在合并各部分结果...")

        # 合并各部分结果
        combined_content = "\n\n".join([f"第{i+1}部分{operation}：{result}" for i, result in enumerate(chunk_results)])

        if operation == "摘要":
            final_prompt = PromptTemplate(
                input_variables=["summaries"],
                template="""请将以下各部分摘要合并成一个完整的综合摘要：

{summaries}

综合摘要："""
            )
        else:
            final_prompt = PromptTemplate(
                input_variables=["analyses"],
                template="""请将以下各部分分析合并成一个完整的综合分析：

{analyses}

综合分析："""
            )

        final_chain = LLMChain(llm=llm, prompt=final_prompt)
        final_result = final_chain.run(summaries=combined_content) if operation == "摘要" else final_chain.run(analyses=combined_content)

        # 清理进度显示
        progress_bar.empty()
        status_text.write("✅ 处理完成！")

        return final_result


class ClickZettaAllInOneDemo:
    """ClickZetta All-in-One 演示类"""

    def __init__(self):
        self.engine = None
        self.embeddings = None
        self.llm = None
        self.doc_store = None
        self.cache_store = None
        self.file_store = None
        self.vector_store = None
        self.hybrid_store = None
        self.chat_history = None

    def initialize(self):
        """初始化所有组件"""
        try:
            # 初始化引擎
            self.engine = init_clickzetta_engine()
            if not self.engine:
                return False

            # 初始化 AI 服务
            self.embeddings, self.llm = init_ai_services()
            if not self.embeddings or not self.llm:
                return False

            # 初始化存储服务
            self.doc_store = ClickZettaDocumentStore(
                engine=self.engine,
                table_name="allinone_documents"
            )

            self.cache_store = ClickZettaStore(
                engine=self.engine,
                table_name="allinone_cache"
            )

            self.file_store = ClickZettaFileStore(
                engine=self.engine,
                volume_type="user",
                subdirectory="allinone_files"
            )

            self.vector_store = ClickZettaVectorStore(
                engine=self.engine,
                embeddings=self.embeddings,
                table_name="allinone_vectors"
            )

            self.hybrid_store = ClickZettaHybridStore(
                engine=self.engine,
                embeddings=self.embeddings,
                table_name="allinone_hybrid"
            )

            self.chat_history = ClickZettaChatMessageHistory(
                engine=self.engine,
                session_id="allinone_session",
                table_name="allinone_chat_history"
            )

            return True

        except Exception as e:
            st.error(f"组件初始化失败: {e}")
            return False

    def get_all_stats(self):
        """获取所有存储服务的统计信息"""
        stats = {}

        try:
            # 文档统计
            try:
                # 添加调试信息：先检查表是否存在
                table_check_query = f"SHOW TABLES LIKE '%{self.doc_store.table_name.split('.')[-1]}%'"
                try:
                    table_exists = self.engine.execute_query(table_check_query)
                    stats["documents_debug"] = {
                        "table_name": self.doc_store.table_name,
                        "table_exists": len(table_exists) > 0 if table_exists else False,
                        "table_check_result": table_exists
                    }
                except Exception as te:
                    stats["documents_debug"] = {
                        "table_name": self.doc_store.table_name,
                        "table_check_error": str(te)
                    }

                doc_result = self.engine.execute_query(f"""
                    SELECT
                        COUNT(*) as doc_count,
                        AVG(LENGTH(doc_content)) as avg_length
                    FROM {self.doc_store.table_name}
                """)
                if doc_result and len(doc_result) > 0:
                    row = doc_result[0]
                    if isinstance(row, dict):
                        stats["documents"] = row
                        stats["documents_debug"]["query_result"] = row
                    else:
                        # 如果不是字典，可能是列表或元组
                        result_dict = {"doc_count": row[0] if len(row) > 0 else 0, "avg_length": row[1] if len(row) > 1 else 0}
                        stats["documents"] = result_dict
                        stats["documents_debug"]["query_result"] = result_dict
                else:
                    stats["documents"] = {"doc_count": 0, "avg_length": 0}
                    stats["documents_debug"]["query_result"] = "No results returned"
            except Exception as e:
                stats["documents"] = {"doc_count": 0, "avg_length": 0}
                stats["documents_debug"]["error"] = str(e)

            # 缓存统计
            try:
                cache_result = self.engine.execute_query(f"""
                    SELECT COUNT(*) as cache_count
                    FROM {self.cache_store.table_name}
                """)
                if cache_result and len(cache_result) > 0:
                    row = cache_result[0]
                    if isinstance(row, dict):
                        stats["cache"] = row
                    else:
                        stats["cache"] = {"cache_count": row[0] if len(row) > 0 else 0}
                else:
                    stats["cache"] = {"cache_count": 0}
            except Exception as e:
                stats["cache"] = {"cache_count": 0}

            # 文件统计
            try:
                files = self.file_store.list_files()
                total_size = sum(file_info[1] for file_info in files if len(file_info) >= 2)
                stats["files"] = {"file_count": len(files), "total_size": total_size}
            except:
                stats["files"] = {"file_count": 0, "total_size": 0}

            # 向量统计
            try:
                vector_result = self.engine.execute_query(f"""
                    SELECT COUNT(*) as vector_count
                    FROM {self.vector_store.table_name}
                """)
                if vector_result and len(vector_result) > 0:
                    row = vector_result[0]
                    if isinstance(row, dict):
                        stats["vectors"] = row
                    else:
                        stats["vectors"] = {"vector_count": row[0] if len(row) > 0 else 0}
                else:
                    stats["vectors"] = {"vector_count": 0}
            except Exception as e:
                stats["vectors"] = {"vector_count": 0}

            # 混合存储统计
            try:
                hybrid_result = self.engine.execute_query(f"""
                    SELECT COUNT(*) as hybrid_count
                    FROM {self.hybrid_store.table_name}
                """)
                if hybrid_result and len(hybrid_result) > 0:
                    row = hybrid_result[0]
                    if isinstance(row, dict):
                        stats["hybrid"] = row
                    else:
                        stats["hybrid"] = {"hybrid_count": row[0] if len(row) > 0 else 0}
                else:
                    stats["hybrid"] = {"hybrid_count": 0}
            except Exception as e:
                stats["hybrid"] = {"hybrid_count": 0}

            # 聊天历史统计
            try:
                chat_result = self.engine.execute_query(f"""
                    SELECT COUNT(*) as message_count
                    FROM {self.chat_history.table_name}
                """)
                if chat_result and len(chat_result) > 0:
                    row = chat_result[0]
                    if isinstance(row, dict):
                        stats["chat"] = row
                    else:
                        stats["chat"] = {"message_count": row[0] if len(row) > 0 else 0}
                else:
                    stats["chat"] = {"message_count": 0}
            except Exception as e:
                stats["chat"] = {"message_count": 0}

        except Exception as e:
            st.warning(f"统计信息获取失败: {e}")
            stats = {
                "documents": {"doc_count": 0, "avg_length": 0},
                "cache": {"cache_count": 0},
                "files": {"file_count": 0, "total_size": 0},
                "vectors": {"vector_count": 0},
                "hybrid": {"hybrid_count": 0},
                "chat": {"message_count": 0}
            }

        return stats

def safe_get_metric_value(stats_dict, category, key, default=0):
    """安全地从统计字典中提取指标值，处理嵌套结构"""
    try:
        category_data = stats_dict.get(category, {})
        if isinstance(category_data, dict):
            value = category_data.get(key, default)

            # 处理嵌套结构的情况，如 {"doc_count": {"doc_count": 6, "avg_length": 123}}
            if isinstance(value, dict) and key in value:
                # 如果value是字典且包含同名key，则提取内部值
                nested_value = value.get(key, default)
                if isinstance(nested_value, (int, float)):
                    return nested_value
                elif isinstance(nested_value, str):
                    try:
                        return float(nested_value)
                    except ValueError:
                        return nested_value
                else:
                    return default

            # 确保返回值是 st.metric 接受的类型
            if value is None:
                return default
            # 确保数值类型
            if isinstance(value, (int, float)):
                return value
            elif isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return value
            else:
                return default
        else:
            return default
    except Exception as e:
        # 在调试模式下显示错误
        return default

def show_overview():
    """显示系统概览"""
    st.header("📊 系统概览")

    if not st.session_state.get('initialized', False):
        st.warning("请先在侧边栏中初始化系统")
        return

    demo = st.session_state.demo
    try:
        stats = demo.get_all_stats()

        # 确保 stats 是字典类型
        if not isinstance(stats, dict):
            st.error("统计数据格式异常")
            return

        # 创建指标卡片
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📄 文档总数",
                safe_get_metric_value(stats, "documents", "doc_count", 0),
                help="存储在 DocumentStore 中的文档数量"
            )

        with col2:
            st.metric(
                "💾 缓存条目",
                safe_get_metric_value(stats, "cache", "cache_count", 0),
                help="键值存储中的缓存条目数"
            )

        with col3:
            file_count = safe_get_metric_value(stats, "files", "file_count", 0)
            st.metric(
                "📁 文件数量",
                file_count,
                help="Volume 存储中的文件数量"
            )

        with col4:
            st.metric(
                "🔍 向量记录",
                safe_get_metric_value(stats, "vectors", "vector_count", 0),
                help="向量存储中的记录数"
            )
    except Exception as e:
        st.error(f"获取统计信息失败: {e}")
        return

    # 存储类型分布图表
    if any(stats.values()):
        st.subheader("存储类型分布")

        storage_data = {
            "文档存储": safe_get_metric_value(stats, "documents", "doc_count", 0),
            "键值存储": safe_get_metric_value(stats, "cache", "cache_count", 0),
            "文件存储": safe_get_metric_value(stats, "files", "file_count", 0),
            "向量存储": safe_get_metric_value(stats, "vectors", "vector_count", 0),
            "混合存储": safe_get_metric_value(stats, "hybrid", "hybrid_count", 0),
            "聊天历史": safe_get_metric_value(stats, "chat", "message_count", 0)
        }

        # 确保所有数值都是有效的
        clean_storage_data = {}
        for key, value in storage_data.items():
            if isinstance(value, (int, float)) and value > 0:
                clean_storage_data[key] = value
            elif isinstance(value, str) and value.isdigit() and int(value) > 0:
                clean_storage_data[key] = int(value)

        if clean_storage_data:
            df = pd.DataFrame(list(clean_storage_data.items()), columns=["存储类型", "数量"])
        else:
            df = pd.DataFrame()  # 空的 DataFrame

        if not df.empty:
            col1, col2 = st.columns(2)

            with col1:
                fig_pie = px.pie(df, values="数量", names="存储类型", title="存储类型分布")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_bar = px.bar(df, x="存储类型", y="数量", title="各存储类型数据量")
                st.plotly_chart(fig_bar, use_container_width=True)

def show_document_storage():
    """文档存储功能"""
    st.header("📝 文档存储管理")

    if not st.session_state.get('initialized', False):
        st.warning("请先在侧边栏中初始化系统")
        return

    demo = st.session_state.demo

    st.markdown("""
    <div class="feature-card">
        <h4>💡 功能说明</h4>
        <p><strong>📝 文档存储</strong> - 将文档内容同时存储到三个ClickZetta存储组件：</p>
        <p>• <strong>📄 DocumentStore</strong> (<code>allinone_documents</code>表) - 存储结构化文档内容和元数据</p>
        <p>• <strong>🔍 VectorStore</strong> (<code>allinone_vectors</code>表) - 自动生成文档向量，支持语义搜索</p>
        <p>• <strong>⚡ HybridStore</strong> (<code>allinone_hybrid</code>表) - 支持混合搜索（语义+关键词）</p>
        <p><strong>🎯 用途：</strong>企业知识库构建，为后续智能问答和搜索提供数据基础</p>
    </div>
    """, unsafe_allow_html=True)

    # 输入方式选择
    input_method = st.radio("选择输入方式:", ["文本输入", "文件上传"], horizontal=True, key="doc_storage_input_method")

    content = ""
    if input_method == "文本输入":
        content = st.text_area(
            "输入文档内容:",
            height=300,
            placeholder="请输入要存储的文档内容...",
            key="doc_storage_text_input"
        )
    else:
        uploaded_file = st.file_uploader(
            "选择文件:",
            type=['txt', 'md'],
            help="支持 .txt 和 .md 文件"
        )
        if uploaded_file:
            content = str(uploaded_file.read(), "utf-8")
            st.text_area("文件内容预览:", content, height=150, disabled=True, key="doc_storage_file_preview")

    # 文档标题输入
    doc_title = st.text_input("文档标题 (可选):", placeholder="输入文档标题")

    if st.button("💾 存储文档", type="primary") and content.strip():
        with st.spinner("正在存储文档..."):
            try:
                st.info(f"📊 文档长度: {len(content)} 字符")

                doc_id = hashlib.md5(content.encode()).hexdigest()
                metadata = {
                    "type": "document",
                    "title": doc_title or "未命名文档",
                    "created_at": datetime.now().isoformat(),
                    "word_count": len(content.split()),
                    "char_count": len(content)
                }

                # 步骤1：存储到 DocumentStore
                with st.expander("💾 存储进度", expanded=True):
                    st.write("🗄️ 正在存储到 DocumentStore...")
                    # 对内容进行清理以避免SQL解析错误
                    sanitized_content = sanitize_content_for_sql(content)
                    demo.doc_store.store_document(
                        doc_id=doc_id,
                        content=sanitized_content,
                        metadata=metadata
                    )
                    st.write("✅ DocumentStore 存储完成")

                    # 存储到 VectorStore 以支持搜索和问答
                    st.write("🔍 正在存储到 VectorStore...")

                    # 如果内容太长，进行分块存储
                    if len(content) > 2000:
                        st.write("📝 内容较长，进行分块存储...")
                        chunks = chunk_text(content)
                        documents = []
                        for i, chunk in enumerate(chunks):
                            chunk_metadata = {**metadata, "chunk_id": i, "total_chunks": len(chunks)}
                            doc = Document(page_content=chunk, metadata=chunk_metadata)
                            documents.append(doc)
                        demo.vector_store.add_documents(documents)
                        st.write(f"✅ VectorStore 存储完成 ({len(chunks)} 个分块)")
                    else:
                        doc = Document(page_content=content, metadata=metadata)
                        demo.vector_store.add_documents([doc])
                        st.write("✅ VectorStore 存储完成")

                    # 如果有 HybridStore，也存储到混合存储
                    if demo.hybrid_store:
                        st.write("🔗 正在存储到 HybridStore...")
                        if len(content) > 2000:
                            demo.hybrid_store.add_documents(documents)
                        else:
                            demo.hybrid_store.add_documents([doc])
                        st.write("✅ HybridStore 存储完成")

                    st.write("🎉 文档存储完成！")

                # 显示结果
                st.success("✅ 文档已成功存储到所有存储服务中！")

                with st.expander("📋 存储详情"):
                    st.json({
                        "文档ID": doc_id,
                        "标题": metadata["title"],
                        "字符数": metadata["char_count"],
                        "词数": metadata["word_count"],
                        "存储时间": metadata["created_at"],
                        "分块数": len(chunks) if len(content) > 2000 else 1
                    })

                # 刷新统计数据
                if 'stats_refresh_counter' not in st.session_state:
                    st.session_state.stats_refresh_counter = 0
                st.session_state.stats_refresh_counter += 1
                st.rerun()

            except Exception as e:
                st.error(f"存储失败: {e}")


def show_intelligent_summary():
    """智能摘要功能"""
    st.header("📄 智能文档摘要")

    if not st.session_state.get('initialized', False):
        st.warning("请先在侧边栏中初始化系统")
        return

    demo = st.session_state.demo

    st.markdown("""
    <div class="feature-card">
        <h4>💡 功能说明</h4>
        <p><strong>📄 智能摘要</strong> - 结合AI能力的文档处理和存储：</p>
        <p>• <strong>🤖 通义千问 AI</strong> - 自动生成文档摘要（智能内容提取）</p>
        <p>• <strong>📄 DocumentStore</strong> (<code>allinone_documents</code>表) - 同时存储原文和AI摘要</p>
        <p>• <strong>🔍 VectorStore</strong> (<code>allinone_vectors</code>表) - 为摘要生成向量表示</p>
        <p>• <strong>⚡ HybridStore</strong> (<code>allinone_hybrid</code>表) - 支持摘要的混合搜索</p>
        <p><strong>🎯 用途：</strong>智能内容管理，长文档自动摘要，提升信息获取效率</p>
    </div>
    """, unsafe_allow_html=True)

    # 输入方式选择
    input_method = st.radio("选择输入方式:", ["文本输入", "文件上传"], horizontal=True, key="summary_input_method")

    content = ""
    if input_method == "文本输入":
        content = st.text_area(
            "请输入文档内容:",
            height=200,
            placeholder="在此处粘贴您要摘要的文档内容...",
            key="summary_text_input"
        )
    else:
        uploaded_file = st.file_uploader(
            "上传文档文件:",
            type=['txt', 'md'],
            help="支持 .txt 和 .md 文件"
        )
        if uploaded_file:
            content = str(uploaded_file.read(), "utf-8")
            st.text_area("文件内容预览:", content, height=150, disabled=True, key="summary_file_preview")

    if st.button("🎯 处理文档", type="primary") and content.strip():
        with st.spinner("正在处理文档..."):
            try:
                # 生成摘要，支持长文本处理
                st.info(f"📊 文本长度: {len(content)} 字符")
                if len(content) > 2000:
                    st.info("📝 检测到长文本，将采用分块处理...")

                # 步骤1：生成摘要
                with st.expander("🔍 摘要生成进度", expanded=True):
                    summary = process_long_text(content, demo.llm, operation="摘要")

                # 步骤2：存储到多个存储服务
                with st.expander("💾 存储进度", expanded=True):
                    doc_id = hashlib.md5(content.encode()).hexdigest()
                    metadata = {
                        "type": "document_summary",
                        "summary": summary,
                        "created_at": datetime.now().isoformat(),
                        "word_count": len(content.split())
                    }

                    # 存储到 DocumentStore
                    st.write("🗄️ 正在存储到 DocumentStore...")
                    # 对内容进行清理以避免SQL解析错误
                    sanitized_content = sanitize_content_for_sql(content)
                    demo.doc_store.store_document(
                        doc_id=doc_id,
                        content=sanitized_content,
                        metadata=metadata
                    )
                    st.write("✅ DocumentStore 存储完成")

                    # 同时存储到 VectorStore 以支持搜索和问答
                    st.write("🔍 正在存储到 VectorStore...")

                    # 如果内容太长，存储摘要而不是原文
                    if len(content) > 2000:
                        vector_content = summary[:2000]  # 使用摘要作为向量内容
                        st.write("📝 内容过长，使用摘要进行向量化...")
                    else:
                        vector_content = content

                    doc = Document(
                        page_content=vector_content,
                        metadata={**metadata, "original_length": len(content)}
                    )
                    demo.vector_store.add_documents([doc])
                    st.write("✅ VectorStore 存储完成")

                    # 如果有 HybridStore，也存储到混合存储
                    if demo.hybrid_store:
                        st.write("🔗 正在存储到 HybridStore...")
                        demo.hybrid_store.add_documents([doc])
                        st.write("✅ HybridStore 存储完成")

                    st.write("🎉 所有存储服务同步完成！")

                # 显示结果
                st.success("✅ 摘要生成并存储成功！")

                st.subheader("📝 生成的摘要")
                st.write(summary)

                st.subheader("📊 文档信息")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("文档ID", doc_id[:8] + "...")
                with col2:
                    st.metric("字数", len(content.split()))
                with col3:
                    st.metric("存储时间", datetime.now().strftime("%H:%M:%S"))

                # 刷新统计数据
                if 'stats_refresh_counter' not in st.session_state:
                    st.session_state.stats_refresh_counter = 0
                st.session_state.stats_refresh_counter += 1
                st.rerun()

            except Exception as e:
                st.error(f"摘要生成失败: {e}")


def show_qa_system():
    """智能问答系统"""
    st.header("💬 智能问答系统")

    if not st.session_state.get('initialized', False):
        st.warning("请先在侧边栏中初始化系统")
        return

    demo = st.session_state.demo

    st.markdown("""
    <div class="feature-card">
        <h4>💡 功能说明</h4>
        <p><strong>💬 智能问答</strong> - 基于RAG(检索增强生成)的对话系统：</p>
        <p>• <strong>🔍 VectorStore</strong> (<code>allinone_vectors</code>表) - 语义检索相关文档</p>
        <p>• <strong>🤖 通义千问 AI</strong> - 基于检索内容生成回答</p>
        <p>• <strong>💬 ChatMessageHistory</strong> (<code>allinone_chat_history</code>表) - 保存对话历史</p>
        <p><strong>🎯 用途：</strong>智能客服系统，企业知识问答，AI助手应用</p>
    </div>
    """, unsafe_allow_html=True)

    # 显示聊天历史 - 添加错误处理
    try:
        if hasattr(demo.chat_history, 'messages') and demo.chat_history.messages:
            st.subheader("💭 聊天历史")
            for message in demo.chat_history.messages[-5:]:  # 显示最近5条
                if message.type == "human":
                    st.chat_message("user").write(message.content)
                else:
                    st.chat_message("assistant").write(message.content)
    except Exception as e:
        # 静默处理聊天历史错误，不影响主要功能
        if st.session_state.get('debug_mode', False):
            st.warning(f"聊天历史加载失败: {e}")
        pass

    # 问答输入
    question = st.text_input("请输入您的问题:", placeholder="例如：请总结一下刚才摘要的文档内容")

    if st.button("🤔 获取答案", type="primary") and question.strip():
        with st.spinner("正在思考答案..."):
            try:
                # 检索相关文档
                docs = demo.vector_store.similarity_search(question, k=3)

                if docs:
                    context = "\n".join([doc.page_content for doc in docs])

                    prompt = PromptTemplate(
                        input_variables=["context", "question"],
                        template="""基于以下上下文信息回答问题：

上下文：
{context}

问题：{question}

答案："""
                    )

                    chain = LLMChain(llm=demo.llm, prompt=prompt)
                    answer = chain.run(context=context, question=question)
                else:
                    answer = "抱歉，我在已存储的文档中没有找到相关信息来回答您的问题。"

                # 保存到聊天历史
                demo.chat_history.add_user_message(question)
                demo.chat_history.add_ai_message(answer)

                # 显示答案
                st.chat_message("user").write(question)
                st.chat_message("assistant").write(answer)

                if docs:
                    with st.expander("📚 参考文档"):
                        for i, doc in enumerate(docs, 1):
                            st.write(f"**文档 {i}:**")
                            st.write(doc.page_content[:200] + "...")

            except Exception as e:
                st.error(f"问答处理失败: {e}")

def show_search_system():
    """搜索系统"""
    st.header("🔍 混合搜索系统")

    if not st.session_state.get('initialized', False):
        st.warning("请先在侧边栏中初始化系统")
        return

    demo = st.session_state.demo

    st.markdown("""
    <div class="feature-card">
        <h4>💡 功能说明</h4>
        <p><strong>🔍 混合搜索</strong> - 最强大的搜索体验：</p>
        <p>• <strong>⚡ HybridStore</strong> (<code>allinone_hybrid</code>表) - 核心混合搜索引擎</p>
        <p>• <strong>🧠 语义搜索</strong> - 理解查询意图，找到相关概念</p>
        <p>• <strong>🔤 关键词搜索</strong> - 精确匹配特定词汇</p>
        <p>• <strong>⚖️ 智能融合</strong> - 可调权重，平衡两种搜索结果</p>
        <p><strong>🎯 用途：</strong>企业内部搜索，电商商品搜索，文档检索系统</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("搜索关键词:", placeholder="输入您要搜索的内容...")
    with col2:
        search_type = st.selectbox("搜索类型:", ["混合搜索", "语义搜索", "关键词搜索"])

    if st.button("🔍 开始搜索", type="primary") and search_query.strip():
        with st.spinner("搜索中..."):
            try:
                if search_type == "语义搜索":
                    results = demo.vector_store.similarity_search(search_query, k=5)
                elif search_type == "关键词搜索":
                    # 简单的关键词搜索（实际项目中可以使用全文搜索）
                    docs = demo.vector_store.similarity_search(search_query, k=10)
                    results = [doc for doc in docs if search_query.lower() in doc.page_content.lower()]
                else:  # 混合搜索
                    results = demo.hybrid_store.similarity_search(search_query, k=5)

                if results:
                    st.success(f"找到 {len(results)} 个相关结果")

                    for i, result in enumerate(results, 1):
                        with st.expander(f"结果 {i}: {result.metadata.get('type', '文档')}"):
                            st.write("**内容:**")
                            st.write(result.page_content[:300] + ("..." if len(result.page_content) > 300 else ""))

                            if result.metadata:
                                st.write("**元数据:**")
                                for key, value in result.metadata.items():
                                    st.write(f"- {key}: {value}")
                else:
                    st.info("没有找到相关结果")

            except Exception as e:
                st.error(f"搜索失败: {e}")

def show_web_crawler():
    """网络爬虫功能"""
    st.header("🕷️ 网络爬虫演示")

    if not st.session_state.get('initialized', False):
        st.warning("请先在侧边栏中初始化系统")
        return

    demo = st.session_state.demo

    st.markdown("""
    <div class="feature-card">
        <h4>💡 功能说明</h4>
        <p><strong>🕷️ 网络爬虫</strong> - 多存储协同的数据采集：</p>
        <p>• <strong>📄 DocumentStore</strong> (<code>allinone_documents</code>表) - 存储提取的文本内容</p>
        <p>• <strong>📁 FileStore</strong> (<code>allinone_files</code> Volume) - 存储原始HTML文件</p>
        <p>• <strong>🔍 VectorStore</strong> (<code>allinone_vectors</code>表) - 生成内容向量</p>
        <p>• <strong>🔑 Store</strong> (<code>allinone_cache</code>表) - 缓存爬取状态</p>
        <p><strong>🎯 用途：</strong>网页数据采集，内容聚合平台，竞品信息监控</p>
    </div>
    """, unsafe_allow_html=True)

    url = st.text_input(
        "网页URL:",
        value="https://www.yunqi.tech",
        placeholder="输入要爬取的网页URL..."
    )

    if st.button("🚀 开始爬取", type="primary") and url.strip():
        if not validators.url(url):
            st.error("请输入有效的URL")
            return

        with st.spinner("正在爬取网页..."):
            try:
                # 爬取网页
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                # 解析内容
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find('title').get_text() if soup.find('title') else "无标题"

                # 提取文本内容
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                text_content = h.handle(response.text)

                # 生成ID
                url_hash = hashlib.md5(url.encode()).hexdigest()

                # 存储到 DocumentStore
                # 对网页内容进行清理以避免SQL解析错误
                sanitized_text_content = sanitize_content_for_sql(text_content)
                demo.doc_store.store_document(
                    doc_id=url_hash,
                    content=sanitized_text_content,
                    metadata={
                        "url": url,
                        "title": title,
                        "crawled_at": datetime.now().isoformat(),
                        "type": "web_page",
                        "word_count": len(text_content.split())
                    }
                )

                # 存储到 FileStore (原始HTML)
                demo.file_store.store_file(
                    file_path=f"{url_hash}.html",
                    content=response.content,
                    mime_type="text/html"
                )

                # 存储到 VectorStore
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "url": url,
                        "title": title,
                        "type": "web_page"
                    }
                )
                demo.vector_store.add_documents([doc])

                # 缓存爬取状态
                demo.cache_store.mset([(f"crawl_status:{url_hash}", b"completed")])

                # 显示结果
                st.markdown("""
                <div class="success-box">
                    <h4>✅ 网页爬取成功！</h4>
                    <p>内容已存储到多个 ClickZetta 存储服务</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 爬取信息")
                    st.metric("网页标题", title)
                    st.metric("内容长度", f"{len(text_content)} 字符")
                    st.metric("字数统计", f"{len(text_content.split())} 词")

                with col2:
                    st.subheader("🗂️ 存储分布")
                    st.write("✅ DocumentStore - 文本内容")
                    st.write("✅ FileStore - 原始HTML")
                    st.write("✅ VectorStore - 向量索引")
                    st.write("✅ Cache - 爬取状态")

                # 内容预览
                with st.expander("📄 内容预览"):
                    st.write(text_content[:500] + ("..." if len(text_content) > 500 else ""))

                # 刷新统计数据
                if 'stats_refresh_counter' not in st.session_state:
                    st.session_state.stats_refresh_counter = 0
                st.session_state.stats_refresh_counter += 1
                st.rerun()

            except Exception as e:
                st.error(f"爬取失败: {e}")

def show_sql_chat():
    """SQL 聊天功能"""
    st.header("💾 SQL 智能问答")

    if not st.session_state.get('initialized', False):
        st.warning("请先在侧边栏中初始化系统")
        return

    demo = st.session_state.demo

    st.markdown("""
    <div class="feature-card">
        <h4>💡 功能说明</h4>
        <p><strong>💾 SQL智能问答</strong> - 直接访问底层数据的两种方式：</p>
        <p>• <strong>🤖 自然语言转SQL</strong> - AI将问题转换为SQL查询</p>
        <p>• <strong>📝 直接SQL执行</strong> - 手写SQL直接查询数据库</p>
        <p>• <strong>🔧 ClickZetta Engine</strong> - 直接执行SQL，访问所有表</p>
        <p>• <strong>📊 结果可视化</strong> - 表格形式展示查询结果</p>
        <p><strong>🎯 用途：</strong>数据分析，系统调试，表结构查看，统计报表</p>
    </div>
    """, unsafe_allow_html=True)

    # 显示可用的表
    st.subheader("📋 可用数据表")
    tables = [
        f"{demo.doc_store.table_name} - 文档存储表",
        f"{demo.cache_store.table_name} - 键值缓存表",
        f"{demo.vector_store.table_name} - 向量存储表",
        f"{demo.chat_history.table_name} - 聊天历史表"
    ]

    for table in tables:
        st.write(f"• {table}")

    # SQL 查询输入
    query_type = st.radio("查询方式:", ["自然语言", "直接SQL"], horizontal=True)

    if query_type == "自然语言":
        nl_query = st.text_input(
            "自然语言查询:",
            placeholder="例如：显示所有文档的数量和标题"
        )

        if st.button("🔍 执行查询", type="primary") and nl_query.strip():
            with st.spinner("正在生成SQL..."):
                try:
                    # 生成 SQL
                    prompt = PromptTemplate(
                        input_variables=["query", "tables"],
                        template="""根据用户的自然语言查询，生成对应的SQL语句。请只返回纯SQL语句，不要包含任何解释或额外文本。

可用表：
{tables}

用户查询：{query}

请生成标准的SQL查询语句："""
                    )

                    chain = LLMChain(llm=demo.llm, prompt=prompt)
                    # 确保查询文本编码正确
                    clean_query = nl_query.encode('utf-8', errors='ignore').decode('utf-8')
                    sql_query = chain.run(
                        query=clean_query,
                        tables="\n".join(tables)
                    )

                    # 清理生成的SQL中的无效字符和格式问题
                    sql_query = sql_query.strip().encode('utf-8', errors='ignore').decode('utf-8')

                    # 移除常见的格式问题
                    if sql_query.startswith('```sql'):
                        sql_query = sql_query[6:]
                    if sql_query.endswith('```'):
                        sql_query = sql_query[:-3]

                    # 移除多余的空行和空格
                    sql_query = '\n'.join(line.strip() for line in sql_query.split('\n') if line.strip())

                    # 确保以分号结尾（如果不是以分号结尾）
                    if not sql_query.endswith(';'):
                        sql_query += ';'

                    st.code(sql_query, language="sql")

                    # 执行 SQL（这里需要安全检查）
                    if st.button("▶️ 执行生成的SQL"):
                        try:
                            # 确保SQL语句编码正确
                            clean_sql_exec = sql_query.strip().encode('utf-8', errors='ignore').decode('utf-8')
                            result = demo.engine.execute_query(clean_sql_exec)
                            if result:
                                df = pd.DataFrame(result)
                                st.dataframe(df)
                            else:
                                st.info("查询执行成功，但没有返回数据")
                        except Exception as e:
                            st.error(f"SQL执行失败: {e}")

                except Exception as e:
                    st.error(f"SQL生成失败: {e}")
    else:
        sql_query = st.text_area(
            "SQL查询:",
            height=100,
            placeholder="SELECT * FROM table_name LIMIT 10",
            key="sql_query_input"
        )

        if st.button("▶️ 执行SQL", type="primary") and sql_query.strip():
            try:
                # 清理SQL查询中的无效字符
                clean_sql = sql_query.strip().encode('utf-8', errors='ignore').decode('utf-8')
                result = demo.engine.execute_query(clean_sql)
                if result:
                    df = pd.DataFrame(result)
                    st.dataframe(df)

                    # 显示统计信息
                    st.write(f"返回 {len(result)} 行数据")
                else:
                    st.info("查询执行成功，但没有返回数据")
            except Exception as e:
                st.error(f"SQL执行失败: {e}")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统设置")

    # 调试模式开关
    debug_mode = st.checkbox("🐛 调试模式", value=st.session_state.get('debug_mode', False))
    st.session_state.debug_mode = debug_mode

    if not st.session_state.get('initialized', False):
        if st.button("🚀 初始化系统", type="primary"):
            with st.spinner("正在初始化..."):
                demo = ClickZettaAllInOneDemo()
                if demo.initialize():
                    st.session_state.demo = demo
                    st.session_state.initialized = True
                    st.success("系统初始化成功！")
                    st.rerun()
                else:
                    st.error("系统初始化失败")
    else:
        st.success("✅ 系统已就绪")

        if st.button("🔄 重新初始化"):
            st.session_state.initialized = False
            st.rerun()

    st.divider()

    # 环境配置信息
    st.header("🌍 环境信息")

    # 显示环境配置状态
    def show_env_config():
        """显示环境配置状态"""
        # ClickZetta 配置
        clickzetta_configs = [
            ("CLICKZETTA_SERVICE", "服务地址"),
            ("CLICKZETTA_INSTANCE", "实例名称"),
            ("CLICKZETTA_WORKSPACE", "工作空间"),
            ("CLICKZETTA_SCHEMA", "模式名称"),
            ("CLICKZETTA_USERNAME", "用户名"),
            ("CLICKZETTA_VCLUSTER", "虚拟集群")
        ]

        # DashScope 配置
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")

        # ClickZetta 配置状态
        with st.expander("📊 ClickZetta 配置", expanded=False):
            for env_var, display_name in clickzetta_configs:
                value = os.getenv(env_var, "")
                if value:
                    # 对于敏感信息，只显示前几位和后几位
                    if env_var in ["CLICKZETTA_USERNAME"]:
                        masked_value = f"{value[:3]}{'*' * (len(value) - 6)}{value[-3:]}" if len(value) > 6 else "*" * len(value)
                        st.write(f"✅ **{display_name}**: `{masked_value}`")
                    else:
                        st.write(f"✅ **{display_name}**: `{value}`")
                else:
                    st.write(f"❌ **{display_name}**: 未配置")

        # DashScope 配置状态
        with st.expander("🤖 DashScope AI 配置", expanded=False):
            if dashscope_api_key:
                # 隐藏API Key的大部分内容
                masked_key = f"{dashscope_api_key[:8]}{'*' * (len(dashscope_api_key) - 16)}{dashscope_api_key[-8:]}" if len(dashscope_api_key) > 16 else "*" * len(dashscope_api_key)
                st.write(f"✅ **API Key**: `{masked_key}`")
                st.write("✅ **状态**: 已配置，支持AI功能")
            else:
                st.write("❌ **API Key**: 未配置")
                st.write("⚠️ **状态**: AI功能不可用（摘要、问答、向量搜索）")

        # 系统状态总览
        clickzetta_configured = all(os.getenv(env_var) for env_var, _ in clickzetta_configs)
        dashscope_configured = bool(dashscope_api_key)

        st.markdown("**📋 配置状态总览:**")
        if clickzetta_configured:
            st.success("✅ ClickZetta: 完全配置")
        else:
            st.error("❌ ClickZetta: 配置不完整")

        if dashscope_configured:
            st.success("✅ DashScope: 已配置")
        else:
            st.warning("⚠️ DashScope: 未配置")

    show_env_config()

    st.divider()

    st.header("📈 快速统计")
    if st.session_state.get('initialized', False):
        demo = st.session_state.demo

        # 添加刷新按钮
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 刷新", key="refresh_stats"):
                st.rerun()

        try:
            stats = demo.get_all_stats()
            # 确保 stats 是字典类型
            if isinstance(stats, dict):
                # 添加调试信息（可选）
                if st.session_state.get('debug_mode', False):
                    st.json(stats)

                # 处理可能的嵌套结构，从调试信息来看需要特别处理
                doc_count = safe_get_metric_value(stats, "documents", "doc_count", 0)
                cache_count = safe_get_metric_value(stats, "cache", "cache_count", 0)
                file_count = safe_get_metric_value(stats, "files", "file_count", 0)
                vector_count = safe_get_metric_value(stats, "vectors", "vector_count", 0)

                st.metric("📄 文档", doc_count)
                st.metric("💾 缓存", cache_count)
                st.metric("📁 文件", file_count)
                st.metric("🔍 向量", vector_count)
            else:
                st.warning("统计数据格式异常")
                if st.session_state.get('debug_mode', False):
                    st.write("Stats type:", type(stats))
                    st.write("Stats content:", stats)
        except Exception as e:
            st.error(f"获取统计失败: {e}")
            if st.session_state.get('debug_mode', False):
                import traceback
                st.text(traceback.format_exc())


def show_help_documentation():
    """显示详细的帮助文档和教育内容"""
    st.header("💡 ClickZetta LangChain 学习指南")

    st.markdown("""
    <div class="feature-card">
        <h4>🎯 学习目标</h4>
        <p>通过这个综合演示，您将深入了解 ClickZetta LangChain 的完整技术栈，
        包括存储服务的具体实现、表结构设计、最佳实践等。</p>
    </div>
    """, unsafe_allow_html=True)

    # 创建帮助文档的子标签页
    help_tab1, help_tab2, help_tab3, help_tab4 = st.tabs([
        "🏗️ 架构原理", "🗄️ 存储服务详解", "📝 代码示例", "🚀 最佳实践"
    ])

    with help_tab1:
        show_architecture_guide()

    with help_tab2:
        show_storage_services_guide()

    with help_tab3:
        show_code_examples()

    with help_tab4:
        show_best_practices()


def show_architecture_guide():
    """显示架构原理指南"""
    st.subheader("🏗️ ClickZetta LangChain 架构原理")

    # 整体架构图
    st.markdown("""
    ### 🌟 整体架构

    ClickZetta LangChain 提供了完整的企业级 AI 应用开发栈：

    ```
    ┌─────────────────────────────────────────────────────┐
    │                  应用层 (Application Layer)           │
    │  • Streamlit UI  • FastAPI  • Jupyter Notebook     │
    └─────────────────────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────┐
    │              LangChain 集成层 (Integration Layer)     │
    │  • Document Processing  • Chain Management         │
    │  • Memory Management   • Agent Framework           │
    └─────────────────────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────┐
    │            ClickZetta 存储服务层 (Storage Layer)      │
    │  • DocumentStore  • VectorStore  • HybridStore     │
    │  • FileStore      • Store        • ChatHistory     │
    └─────────────────────────────────────────────────────┘
                               ↓
    ┌─────────────────────────────────────────────────────┐
    │              ClickZetta Lakehouse 引擎              │
    │  • SQL Engine  • Vector Engine  • Storage Engine   │
    └─────────────────────────────────────────────────────┘
    ```
    """)

    # 数据流转示例
    st.markdown("### 🔄 数据流转示例")

    data_flows = {
        "文档处理流程": [
            "用户上传文档 → 解析提取文本",
            "文本清洗 → 元数据提取",
            "存储到 DocumentStore → 生成向量表示",
            "存储到 VectorStore → 建立索引",
            "同步到 HybridStore → 启用搜索"
        ],
        "智能问答流程": [
            "用户提问 → 问题向量化",
            "VectorStore 检索 → 找到相关文档",
            "上下文构建 → LLM 生成答案",
            "存储到 ChatHistory → 更新对话状态"
        ],
        "混合搜索流程": [
            "搜索查询 → 关键词提取",
            "并行执行 → 向量搜索 + 关键词搜索",
            "结果融合 → 相关性排序",
            "返回最终结果 → 用户展示"
        ]
    }

    for flow_name, steps in data_flows.items():
        with st.expander(f"🔄 {flow_name}", expanded=False):
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")


def show_storage_services_guide():
    """显示存储服务详解"""
    st.subheader("🗄️ ClickZetta 存储服务详解")

    # 显示当前系统的具体配置
    if st.session_state.get('initialized', False) and hasattr(st.session_state, 'engine_manager'):
        manager = st.session_state.engine_manager

        st.markdown("### 📋 当前系统配置")

        # 引擎配置信息
        with st.expander("🔧 引擎配置信息", expanded=True):
            engine_info = {
                "ClickZetta Service": os.getenv("CLICKZETTA_SERVICE", "未配置"),
                "Instance": os.getenv("CLICKZETTA_INSTANCE", "未配置"),
                "Workspace": os.getenv("CLICKZETTA_WORKSPACE", "未配置"),
                "Schema": os.getenv("CLICKZETTA_SCHEMA", "未配置"),
                "VCluster": os.getenv("CLICKZETTA_VCLUSTER", "未配置")
            }

            for key, value in engine_info.items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**{key}:**")
                with col2:
                    st.code(value)

        # 存储服务详情
        storage_services = [
            {
                "name": "DocumentStore",
                "emoji": "📄",
                "description": "结构化文档存储",
                "table_name": "allinone_documents",
                "store_obj": manager.document_store,
                "key_features": [
                    "自动文档解析和元数据提取",
                    "支持全文搜索和条件查询",
                    "版本管理和文档历史",
                    "批量导入和导出功能"
                ]
            },
            {
                "name": "VectorStore",
                "emoji": "🔍",
                "description": "向量化存储和语义搜索",
                "table_name": "allinone_vectors",
                "store_obj": manager.vector_store,
                "key_features": [
                    "自动向量化处理",
                    "高性能相似度搜索",
                    "支持多种嵌入模型",
                    "实时索引更新"
                ]
            },
            {
                "name": "HybridStore",
                "emoji": "⚡",
                "description": "混合搜索引擎",
                "table_name": "allinone_hybrid",
                "store_obj": manager.hybrid_store,
                "key_features": [
                    "结合向量和关键词搜索",
                    "智能结果融合算法",
                    "灵活的权重调整",
                    "多模态搜索支持"
                ]
            },
            {
                "name": "FileStore (Volume)",
                "emoji": "📁",
                "description": "文件存储和管理",
                "table_name": "allinone_files (Volume)",
                "store_obj": manager.file_store,
                "key_features": [
                    "大文件存储优化",
                    "文件版本控制",
                    "访问权限管理",
                    "自动备份和恢复"
                ]
            },
            {
                "name": "Store (Cache)",
                "emoji": "🔑",
                "description": "键值存储和缓存",
                "table_name": "allinone_cache",
                "store_obj": manager.cache_store,
                "key_features": [
                    "高性能键值操作",
                    "TTL 过期管理",
                    "分布式缓存",
                    "原子操作支持"
                ]
            },
            {
                "name": "ChatMessageHistory",
                "emoji": "💬",
                "description": "对话历史管理",
                "table_name": "allinone_chat_history",
                "store_obj": manager.chat_memory,
                "key_features": [
                    "结构化消息存储",
                    "对话上下文管理",
                    "多轮对话支持",
                    "用户会话隔离"
                ]
            }
        ]

        for service in storage_services:
            with st.expander(f"{service['emoji']} {service['name']} - {service['description']}", expanded=False):
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**📊 服务信息:**")
                    st.markdown(f"• **表名/Volume:** `{service['table_name']}`")

                    # 获取统计信息
                    try:
                        if hasattr(service['store_obj'], 'get_stats'):
                            stats = service['store_obj'].get_stats()
                            if stats:
                                if isinstance(stats, dict):
                                    for key, value in stats.items():
                                        if isinstance(value, (int, float)):
                                            st.markdown(f"• **{key}:** {value}")
                                        elif isinstance(value, dict) and len(value) == 1:
                                            # 处理嵌套结构
                                            nested_key = list(value.keys())[0]
                                            nested_value = value[nested_key]
                                            if isinstance(nested_value, (int, float)):
                                                st.markdown(f"• **{key}:** {nested_value}")
                                else:
                                    st.markdown(f"• **记录数:** {stats}")
                            else:
                                st.markdown("• **状态:** 暂无数据")
                    except Exception as e:
                        st.markdown(f"• **状态:** 获取统计失败 ({str(e)[:50]}...)")

                with col2:
                    st.markdown("**🎯 核心特性:**")
                    for feature in service['key_features']:
                        st.markdown(f"• {feature}")

                # 表结构查看功能
                if st.button(f"🔍 查看 {service['name']} 表结构", key=f"schema_{service['name']}"):
                    try:
                        if service['table_name'].endswith('(Volume)'):
                            st.info(f"📁 {service['name']} 使用 ClickZetta Volume 存储，为二进制文件存储，无固定表结构")
                        else:
                            # 尝试获取表结构
                            with st.spinner(f"正在获取 {service['table_name']} 表结构..."):
                                try:
                                    schema_sql = f"DESCRIBE {service['table_name']}"
                                    result = manager.engine.execute_sql(schema_sql)

                                    if result:
                                        st.success(f"✅ {service['table_name']} 表结构:")
                                        if isinstance(result, list) and len(result) > 0:
                                            # 创建表格显示结构
                                            if isinstance(result[0], dict):
                                                schema_df = pd.DataFrame(result)
                                                st.dataframe(schema_df, use_container_width=True)
                                            else:
                                                for row in result:
                                                    st.write(row)

                                        # 显示示例查询
                                        st.markdown("**💡 示例查询:**")
                                        sample_queries = {
                                            "allinone_documents": [
                                                f"SELECT COUNT(*) FROM {service['table_name']};",
                                                f"SELECT title, created_at FROM {service['table_name']} LIMIT 5;",
                                                f"SELECT title, LENGTH(content) as content_length FROM {service['table_name']} ORDER BY created_at DESC LIMIT 3;"
                                            ],
                                            "allinone_vectors": [
                                                f"SELECT COUNT(*) FROM {service['table_name']};",
                                                f"SELECT id, document_id FROM {service['table_name']} LIMIT 5;"
                                            ],
                                            "allinone_cache": [
                                                f"SELECT COUNT(*) FROM {service['table_name']};",
                                                f"SELECT key, created_at FROM {service['table_name']} LIMIT 5;"
                                            ]
                                        }

                                        if service['table_name'] in sample_queries:
                                            for query in sample_queries[service['table_name']]:
                                                st.code(query, language="sql")
                                    else:
                                        st.warning(f"⚠️ 表 {service['table_name']} 可能尚未创建或无数据")

                                except Exception as sql_error:
                                    st.error(f"❌ 获取表结构失败: {sql_error}")
                                    st.markdown("**💡 可能的原因:**")
                                    st.markdown("• 表尚未创建（请先使用相应功能生成数据）")
                                    st.markdown("• 数据库连接问题")
                                    st.markdown("• 权限不足")
                    except Exception as e:
                        st.error(f"操作失败: {e}")
    else:
        st.warning("⚠️ 请先在侧边栏点击 '🚀 初始化系统' 按钮来查看具体配置信息")

        # 显示通用的存储服务介绍
        st.markdown("### 📚 存储服务类型介绍")

        st.markdown("""
        <div class="feature-card">
            <h4>🎯 6种存储服务的形象类比</h4>
            <p>为了帮助理解不同存储服务的作用，我们用生活中熟悉的场景来类比：</p>
        </div>
        """, unsafe_allow_html=True)

        # 添加类比说明
        st.markdown("#### 🏠 生活场景类比")

        analogy_col1, analogy_col2, analogy_col3 = st.columns(3)

        with analogy_col1:
            st.markdown("""
            **📄 DocumentStore = 📚 图书馆**
            - 像图书馆的书架，每本书都有：
            - 📖 书名（标题）
            - 📝 内容（正文）
            - 🏷️ 标签（元数据）
            - 📅 入库时间
            """)

            st.markdown("""
            **🔍 VectorStore = 🧠 大脑记忆**
            - 像人脑记住概念的"感觉"
            - 🤔 "机器学习"和"AI"很相似
            - 🔗 通过"语义相似度"找内容
            - 💡 理解意思，不只是关键词
            """)

        with analogy_col2:
            st.markdown("""
            **🔑 Store (Cache) = 🗂️ 便签纸**
            - 像贴在冰箱上的便签
            - 📝 简单的键值对
            - ⏰ 可以设置过期时间
            - 🏃 存取速度很快
            """)

            st.markdown("""
            **📁 FileStore = 📦 仓库**
            - 像仓储中心的大箱子
            - 📦 存放各种文件（图片、视频、PDF）
            - 🏷️ 每个箱子有标签
            - 💾 适合大文件存储
            """)

        with analogy_col3:
            st.markdown("""
            **⚡ HybridStore = 🔍 智能搜索**
            - 像百度/谷歌的搜索引擎
            - 🎯 既能理解意思（语义）
            - 🔤 又能匹配关键词
            - ⚖️ 两种方式结合，结果更准
            """)

            st.markdown("""
            **💬 ChatHistory = 📞 通话记录**
            - 像手机的聊天记录
            - 👤 记住谁说了什么
            - ⏰ 按时间顺序排列
            - 🔄 支持多轮对话
            """)

        st.divider()

        generic_services = {
            "DocumentStore": {
                "emoji": "📄",
                "description": "结构化文档存储 - 像图书馆管理系统",
                "real_world_analogy": "📚 图书馆书架：每本书都有标题、内容、分类、入库日期",
                "when_to_use": "当你需要存储和管理大量文档时（如企业知识库、文章管理）",
                "use_cases": ["知识库管理", "文档归档", "内容管理", "企业文档中心"],
                "typical_schema": ["id", "title", "content", "metadata", "created_at", "updated_at"],
                "example": "存储公司的技术文档、政策文件、产品说明书等"
            },
            "VectorStore": {
                "emoji": "🔍",
                "description": "向量化存储 - 像大脑的概念记忆",
                "real_world_analogy": "🧠 大脑记忆：理解概念的'感觉'，找到意思相近的内容",
                "when_to_use": "当你需要根据意思而不是精确词汇来搜索时（如智能问答、推荐系统）",
                "use_cases": ["语义搜索", "推荐系统", "智能问答", "内容发现"],
                "typical_schema": ["id", "document_id", "embedding", "metadata", "created_at"],
                "example": "搜索'机器学习'时，也能找到含有'人工智能'、'深度学习'的文档"
            },
            "HybridStore": {
                "emoji": "⚡",
                "description": "混合搜索引擎 - 像智能搜索引擎",
                "real_world_analogy": "🔍 百度/谷歌：既能理解意思，又能精确匹配关键词",
                "when_to_use": "当你需要最准确的搜索结果时（结合了精确匹配和语义理解）",
                "use_cases": ["企业搜索", "电商搜索", "学术检索", "多媒体搜索"],
                "typical_schema": ["id", "content", "embedding", "keywords", "metadata"],
                "example": "搜索'苹果手机'既能找到精确包含这些词的文档，也能找到相关的'iPhone'文档"
            },
            "FileStore": {
                "emoji": "📁",
                "description": "文件存储系统 - 像云盘仓库",
                "real_world_analogy": "📦 云盘/仓库：存放各种文件（图片、视频、PDF），每个都有标签",
                "when_to_use": "当你需要存储二进制文件（图片、视频、文档等）时",
                "use_cases": ["多媒体资源", "数据湖", "备份归档", "内容分发"],
                "typical_schema": ["Volume based", "Binary storage", "File metadata"],
                "example": "存储产品图片、用户上传的文档、视频文件等"
            },
            "Store": {
                "emoji": "🔑",
                "description": "键值存储和缓存 - 像便签纸",
                "real_world_analogy": "🗂️ 便签纸：简单的键值对，贴在冰箱上的备忘录",
                "when_to_use": "当你需要快速存取简单数据时（如用户会话、配置参数）",
                "use_cases": ["会话管理", "配置存储", "临时缓存", "分布式锁"],
                "typical_schema": ["key", "value", "expires_at", "created_at"],
                "example": "记住用户登录状态、保存应用配置、临时缓存计算结果"
            },
            "ChatMessageHistory": {
                "emoji": "💬",
                "description": "对话历史管理 - 像聊天记录",
                "real_world_analogy": "📞 微信聊天记录：按时间记录谁说了什么，支持翻看历史",
                "when_to_use": "当你需要构建对话系统时（如智能客服、聊天机器人）",
                "use_cases": ["智能客服", "聊天机器人", "AI助手", "多轮问答"],
                "typical_schema": ["session_id", "message_type", "content", "timestamp"],
                "example": "客服系统记录用户对话，AI助手记住前面的聊天内容"
            }
        }

        for name, info in generic_services.items():
            with st.expander(f"{info['emoji']} {name} - {info['description']}", expanded=False):
                # 添加生活化类比
                st.markdown(f"**🏠 生活类比:** {info['real_world_analogy']}")
                st.markdown(f"**🎯 使用场景:** {info['when_to_use']}")
                st.markdown(f"**💡 实际例子:** {info['example']}")

                st.divider()

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**💼 应用场景:**")
                    for use_case in info['use_cases']:
                        st.markdown(f"• {use_case}")

                with col2:
                    st.markdown("**🗃️ 典型表结构:**")
                    for field in info['typical_schema']:
                        st.markdown(f"• `{field}`")


def show_code_examples():
    """显示代码示例"""
    st.subheader("📝 代码示例和实现模板")

    # 基础使用示例
    st.markdown("### 🚀 快速开始")

    basic_example = '''
# 1. 初始化 ClickZetta 引擎
from langchain_clickzetta import ClickZettaEngine

engine = ClickZettaEngine(
    service=os.getenv("CLICKZETTA_SERVICE"),
    instance=os.getenv("CLICKZETTA_INSTANCE"),
    workspace=os.getenv("CLICKZETTA_WORKSPACE"),
    schema=os.getenv("CLICKZETTA_SCHEMA"),
    username=os.getenv("CLICKZETTA_USERNAME"),
    password=os.getenv("CLICKZETTA_PASSWORD"),
    vcluster=os.getenv("CLICKZETTA_VCLUSTER")
)

# 2. 创建存储服务
from langchain_clickzetta import ClickZettaDocumentStore

document_store = ClickZettaDocumentStore(
    engine=engine,
    table_name="my_documents"
)

# 3. 存储文档
from langchain_core.documents import Document

document = Document(
    page_content="这是一个示例文档内容",
    metadata={"title": "示例文档", "category": "演示"}
)

document_store.add_document(document)

# 4. 搜索文档
results = document_store.search("示例", k=5)
for doc in results:
    print(f"标题: {doc.metadata['title']}")
    print(f"内容: {doc.page_content[:100]}...")
'''

    st.code(basic_example, language="python")

    # 高级功能示例
    st.markdown("### ⚡ 高级功能示例")

    advanced_tabs = st.tabs(["向量搜索", "混合搜索", "对话历史", "文件存储", "键值缓存"])

    with advanced_tabs[0]:
        st.markdown("**向量搜索和语义检索:**")
        vector_example = '''
from langchain_clickzetta import ClickZettaVectorStore
from langchain_community.embeddings import DashScopeEmbeddings

# 初始化向量存储
embeddings = DashScopeEmbeddings(model="text-embedding-v1")
vector_store = ClickZettaVectorStore(
    engine=engine,
    embedding=embeddings,
    table_name="my_vectors"
)

# 添加文档（自动向量化）
vector_store.add_document(document)

# 语义搜索
results = vector_store.similarity_search(
    "人工智能和机器学习",
    k=3,
    score_threshold=0.7
)

# 带分数的搜索
results_with_scores = vector_store.similarity_search_with_score(
    "深度学习技术",
    k=5
)

for doc, score in results_with_scores:
    print(f"相似度: {score:.3f}")
    print(f"内容: {doc.page_content[:100]}...")
'''
        st.code(vector_example, language="python")

    with advanced_tabs[1]:
        st.markdown("**混合搜索（向量+关键词）:**")
        hybrid_example = '''
from langchain_clickzetta import ClickZettaHybridStore

# 初始化混合存储
hybrid_store = ClickZettaHybridStore(
    engine=engine,
    embedding=embeddings,
    table_name="my_hybrid"
)

# 添加文档
hybrid_store.add_document(document)

# 混合搜索（可调权重）
results = hybrid_store.search(
    query="机器学习算法",
    search_type="hybrid",  # "semantic", "keyword", "hybrid"
    k=5,
    alpha=0.5  # 0.0=纯关键词, 1.0=纯向量, 0.5=平衡
)

# 语义搜索
semantic_results = hybrid_store.search(
    query="人工智能发展趋势",
    search_type="semantic",
    k=3
)

# 关键词搜索
keyword_results = hybrid_store.search(
    query="Python 编程 教程",
    search_type="keyword",
    k=3
)
'''
        st.code(hybrid_example, language="python")

    with advanced_tabs[2]:
        st.markdown("**对话历史管理:**")
        chat_example = '''
from langchain_clickzetta import ClickZettaChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# 初始化对话历史
chat_history = ClickZettaChatMessageHistory(
    engine=engine,
    session_id="user_123",
    table_name="chat_sessions"
)

# 添加消息
chat_history.add_message(HumanMessage(content="你好，请介绍一下机器学习"))
chat_history.add_message(AIMessage(content="机器学习是人工智能的一个分支..."))

# 获取历史消息
messages = chat_history.get_messages()
for msg in messages:
    role = "用户" if isinstance(msg, HumanMessage) else "AI"
    print(f"{role}: {msg.content}")

# 清除历史
chat_history.clear()

# 获取最近的 N 条消息
recent_messages = chat_history.get_recent_messages(n=10)
'''
        st.code(chat_example, language="python")

    with advanced_tabs[3]:
        st.markdown("**文件存储和管理:**")
        file_example = '''
from langchain_clickzetta import ClickZettaFileStore

# 初始化文件存储
file_store = ClickZettaFileStore(
    engine=engine,
    volume_name="my_files"
)

# 存储文件
with open("document.pdf", "rb") as f:
    file_content = f.read()

file_store.store(
    key="docs/important_document.pdf",
    content=file_content,
    metadata={
        "content_type": "application/pdf",
        "size": len(file_content),
        "uploaded_by": "user_123"
    }
)

# 读取文件
retrieved_content = file_store.retrieve("docs/important_document.pdf")

# 列出文件
files = file_store.list_files(prefix="docs/")
for file_info in files:
    print(f"文件: {file_info['key']}")
    print(f"大小: {file_info['size']} bytes")

# 删除文件
file_store.delete("docs/old_document.pdf")
'''
        st.code(file_example, language="python")

    with advanced_tabs[4]:
        st.markdown("**键值存储和缓存:**")
        cache_example = '''
from langchain_clickzetta import ClickZettaStore
import json
from datetime import datetime, timedelta

# 初始化键值存储
cache_store = ClickZettaStore(
    engine=engine,
    table_name="my_cache"
)

# 存储简单值
cache_store.set("user_session", "active")
cache_store.set("login_count", 42)

# 存储复杂对象
user_profile = {
    "user_id": "123",
    "name": "张三",
    "preferences": ["AI", "机器学习"],
    "last_login": datetime.now().isoformat()
}
cache_store.set("user_profile_123", json.dumps(user_profile))

# 带过期时间的存储
expiry_time = datetime.now() + timedelta(hours=1)
cache_store.set("temp_token", "abc123", expires_at=expiry_time)

# 读取值
session_status = cache_store.get("user_session")
user_data = json.loads(cache_store.get("user_profile_123"))

# 批量操作
cache_store.batch_set({
    "config_theme": "dark",
    "config_language": "zh-CN",
    "config_notifications": True
})

batch_data = cache_store.batch_get(["config_theme", "config_language"])

# 删除
cache_store.delete("temp_token")

# 检查存在性
if cache_store.exists("user_session"):
    print("用户会话存在")
'''
        st.code(cache_example, language="python")


def show_best_practices():
    """显示最佳实践指南"""
    st.subheader("🚀 最佳实践和开发建议")

    # 性能优化
    st.markdown("### ⚡ 性能优化")

    perf_tips = {
        "连接管理": [
            "复用 ClickZettaEngine 实例，避免频繁创建连接",
            "使用连接池管理并发访问",
            "合理设置连接超时参数",
            "定期清理空闲连接"
        ],
        "存储优化": [
            "选择合适的表名和索引策略",
            "批量操作时使用 batch_add 等方法",
            "大文件存储优先使用 FileStore",
            "定期清理过期数据和缓存"
        ],
        "向量搜索": [
            "选择合适的嵌入模型和维度",
            "设置合理的相似度阈值",
            "使用分页查询处理大结果集",
            "缓存常用查询的向量表示"
        ],
        "内存管理": [
            "及时释放大对象和文件句柄",
            "使用生成器处理大数据集",
            "监控内存使用情况",
            "合理设置批处理大小"
        ]
    }

    for category, tips in perf_tips.items():
        with st.expander(f"🔧 {category}", expanded=False):
            for tip in tips:
                st.markdown(f"• {tip}")

    # 安全最佳实践
    st.markdown("### 🔒 安全最佳实践")

    security_code = '''
# 1. 环境变量管理
import os
from dotenv import load_dotenv

# 永远不要硬编码敏感信息
# ❌ 错误做法
# engine = ClickZettaEngine(
#     service="my-service",
#     username="admin",
#     password="password123"  # 危险！
# )

# ✅ 正确做法
load_dotenv()
engine = ClickZettaEngine(
    service=os.getenv("CLICKZETTA_SERVICE"),
    username=os.getenv("CLICKZETTA_USERNAME"),
    password=os.getenv("CLICKZETTA_PASSWORD")
)

# 2. 输入验证和清理
def sanitize_input(content):
    """清理用户输入，防止注入攻击"""
    if not content:
        return content

    # 移除危险字符
    dangerous_chars = ['<', '>', '"', "'", '&', '\\x00']
    for char in dangerous_chars:
        content = content.replace(char, '')

    # 限制长度
    max_length = 10000
    if len(content) > max_length:
        content = content[:max_length] + "...[截断]"

    return content

# 3. 错误处理（不泄露敏感信息）
try:
    result = engine.execute_sql(query)
except Exception as e:
    # ❌ 不要直接暴露详细错误
    # print(f"Database error: {str(e)}")

    # ✅ 记录详细错误，返回通用消息
    logger.error(f"Database operation failed: {str(e)}")
    return {"error": "操作失败，请稍后重试"}
'''

    st.code(security_code, language="python")

    # 学习资源
    st.markdown("### 📚 学习资源和社区")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📖 官方文档:**")
        st.markdown("• [ClickZetta 官方文档](https://www.yunqi.tech/documents/)")
        st.markdown("• [LangChain 官方文档](https://python.langchain.com)")
        st.markdown("• [通义千问 API 文档](https://help.aliyun.com/product/2400395.html)")
        st.markdown("• [Streamlit 开发指南](https://docs.streamlit.io)")

    with col2:
        st.markdown("**🛠️ 开发工具:**")
        st.markdown("• Jupyter Notebook 交互式开发")
        st.markdown("• VS Code + Python 扩展")
        st.markdown("• Docker Desktop 容器化")
        st.markdown("• Git 版本控制")

    st.markdown("**💡 开发建议:**")
    st.info("""
    1. **从简单开始**: 先掌握单个存储服务，再尝试复杂的组合使用
    2. **多实践**: 通过实际项目加深对 ClickZetta LangChain 的理解
    3. **关注性能**: 在开发过程中关注查询性能和资源使用
    4. **社区参与**: 积极参与开源社区，分享经验和最佳实践
    5. **持续学习**: 跟上 AI 和数据技术的最新发展趋势
    """)


# 主内容区域
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 系统概览",
    "📝 文档存储",
    "📄 智能摘要",
    "💬 智能问答",
    "🔍 混合搜索",
    "🕷️ 网络爬虫",
    "💾 SQL问答",
    "💡 帮助文档"
])

with tab1:
    show_overview()

with tab2:
    show_document_storage()

with tab3:
    show_intelligent_summary()

with tab4:
    show_qa_system()

with tab5:
    show_search_system()

with tab6:
    show_web_crawler()

with tab7:
    show_sql_chat()

with tab8:
    show_help_documentation()

# 页脚
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🚀 ClickZetta LangChain All-in-One Demo |
    展示企业级 AI 应用的完整技术栈</p>
</div>
""", unsafe_allow_html=True)