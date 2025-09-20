"""
ClickZetta Web Crawler & Storage Demo

展示LangChain网站爬取功能与ClickZetta存储服务的完整集成。
演示文档存储、键值存储、文件存储和向量存储的实际应用。
"""

import streamlit as st
import sys
import os
import hashlib
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse
import re

try:
    import requests
    from bs4 import BeautifulSoup
    import html2text
    import validators
    from langchain_core.documents import Document
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_clickzetta import (
        ClickZettaEngine,
        ClickZettaStore,
        ClickZettaDocumentStore,
        ClickZettaFileStore,
        ClickZettaVectorStore
    )
    from dotenv import load_dotenv
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    import_error = str(e)

# 页面配置
st.set_page_config(
    page_title="ClickZetta Web Crawler & Storage Demo",
    page_icon="🕷️",
    layout="wide"
)

def extract_text_content(html_content: str) -> str:
    """从HTML中提取纯文本内容"""
    if not DEPENDENCIES_AVAILABLE:
        return "依赖未安装"

    try:
        # 使用html2text转换HTML为Markdown格式文本
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        text_content = h.handle(html_content)

        # 清理多余的空行
        lines = [line.strip() for line in text_content.split('\n')]
        cleaned_lines = []
        for line in lines:
            if line or (cleaned_lines and cleaned_lines[-1]):
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)
    except Exception as e:
        st.warning(f"文本提取失败: {e}")
        # fallback to BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()

def extract_metadata(soup, url: str) -> Dict:
    """提取网页元数据"""
    metadata = {
        "url": url,
        "crawled_at": datetime.now().isoformat(),
        "title": "",
        "description": "",
        "author": "",
        "keywords": [],
        "language": "zh",
        "content_type": "webpage"
    }

    try:
        # 提取标题
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()

        # 提取描述
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag:
            metadata["description"] = desc_tag.get('content', '').strip()

        # 提取作者
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag:
            metadata["author"] = author_tag.get('content', '').strip()

        # 提取关键词
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            keywords = keywords_tag.get('content', '').strip()
            metadata["keywords"] = [k.strip() for k in keywords.split(',') if k.strip()]

        # 检测语言
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata["language"] = html_tag.get('lang')

    except Exception as e:
        st.warning(f"元数据提取失败: {e}")

    return metadata

def url_to_key(url: str) -> str:
    """将URL转换为存储键"""
    return hashlib.md5(url.encode()).hexdigest()

def load_env_config():
    """加载环境配置"""
    load_dotenv()
    return {
        'clickzetta_service': os.getenv('CLICKZETTA_SERVICE', ''),
        'clickzetta_instance': os.getenv('CLICKZETTA_INSTANCE', ''),
        'clickzetta_workspace': os.getenv('CLICKZETTA_WORKSPACE', ''),
        'clickzetta_schema': os.getenv('CLICKZETTA_SCHEMA', ''),
        'clickzetta_username': os.getenv('CLICKZETTA_USERNAME', ''),
        'clickzetta_password': os.getenv('CLICKZETTA_PASSWORD', ''),
        'clickzetta_vcluster': os.getenv('CLICKZETTA_VCLUSTER', ''),
        'dashscope_api_key': os.getenv('DASHSCOPE_API_KEY', ''),
    }

def check_environment():
    """检查环境配置"""
    config = load_env_config()

    clickzetta_available = all([
        config['clickzetta_service'], config['clickzetta_instance'],
        config['clickzetta_workspace'], config['clickzetta_schema'],
        config['clickzetta_username'], config['clickzetta_password'],
        config['clickzetta_vcluster']
    ])

    dashscope_available = bool(config['dashscope_api_key'])

    return {
        "clickzetta_available": clickzetta_available,
        "dashscope_available": dashscope_available,
        "config": config
    }

def display_environment_status(env_status):
    """显示环境状态"""
    col1, col2 = st.columns(2)

    with col1:
        if env_status["clickzetta_available"]:
            st.success("✅ ClickZetta 配置完整")
        else:
            st.error("❌ ClickZetta 配置不完整")

    with col2:
        if env_status["dashscope_available"]:
            st.success("✅ DashScope API 配置完整")
            st.session_state['dashscope_available'] = True
        else:
            st.warning("⚠️ DashScope API 未配置")
            st.session_state['dashscope_available'] = False

def create_sidebar_info():
    """创建侧边栏信息"""
    st.sidebar.header("ℹ️ 应用信息")
    st.sidebar.markdown("""
    ### 🕷️ 网络爬虫演示

    **功能特色:**
    - 网页内容爬取
    - 多种存储服务
    - 语义搜索
    - 统计分析

    **存储服务:**
    - DocumentStore: 文档存储
    - Store: 键值存储
    - FileStore: 文件存储
    - VectorStore: 向量存储
    """)

    # 数据管理功能
    st.sidebar.header("🗑️ 数据管理")

    # 统计信息
    with st.sidebar.expander("📊 统计信息"):
        env_status = check_environment()
        if env_status["clickzetta_available"]:
            try:
                engine = get_clickzetta_engine()

                # 检查各种存储的数据量
                doc_count = 0
                cache_count = 0
                vector_count = 0
                file_count = 0

                try:
                    # DocumentStore数据
                    doc_result, _ = engine.execute_query("SELECT COUNT(*) as count FROM web_crawler_documents")
                    doc_count = doc_result[0]['count'] if doc_result else 0
                except:
                    pass

                try:
                    # Cache数据
                    cache_result, _ = engine.execute_query("SELECT COUNT(*) as count FROM web_crawler_cache")
                    cache_count = cache_result[0]['count'] if cache_result else 0
                except:
                    pass

                try:
                    # Vector数据
                    vector_result, _ = engine.execute_query("SELECT COUNT(*) as count FROM web_crawler_vectors")
                    vector_count = vector_result[0]['count'] if vector_result else 0
                except:
                    pass

                st.metric("📚 文档数据", f"{doc_count} 条")
                st.metric("🗂️ 缓存数据", f"{cache_count} 条")
                st.metric("🧠 向量数据", f"{vector_count} 条")

                total_data = doc_count + cache_count + vector_count
                if total_data > 0:
                    st.success(f"💡 检测到已有 {total_data} 条数据")

            except Exception as e:
                st.warning(f"⚠️ 无法获取统计信息: {e}")
        else:
            st.warning("⚠️ 请先配置ClickZetta连接")

    # 清空数据功能
    with st.sidebar.expander("🗑️ 数据清空"):
        st.write("**清空爬虫数据**")
        st.caption("删除所有爬取的网页数据和文件，重新开始")

        if st.button("🗑️ 清空所有数据", type="secondary", help="删除所有数据"):
            env_status = check_environment()
            if env_status["clickzetta_available"]:
                st.session_state.clear_data_requested = True
                st.info("数据清空请求已提交，正在处理...")
                st.rerun()
            else:
                st.warning("⚠️ 请先配置ClickZetta连接")

def clear_file_storage(crawler_instance):
    """清空文件存储"""
    try:
        # 获取所有文件
        files = crawler_instance.file_store.list_files()
        if not files:
            st.info("没有找到文件需要删除")
            return 0

        st.info(f"找到 {len(files)} 个文件，开始删除...")

        # 收集文件路径
        file_paths = []
        for file_info in files:
            if isinstance(file_info, tuple) and len(file_info) >= 1:
                file_paths.append(file_info[0])

        if not file_paths:
            st.info("没有找到有效的文件路径")
            return 0

        # 删除文件

        # 构建正确的键名（基于 store_file 的存储方式）
        keys_to_delete = []
        for key in file_paths:
            keys_to_delete.append(key)                    # 主文件键
            keys_to_delete.append(f"_metadata_{key}")     # 元数据键

        # 使用 volume_store.mdelete 删除
        crawler_instance.file_store.volume_store.mdelete(keys_to_delete)

        # 验证删除结果
        remaining_files = crawler_instance.file_store.list_files()
        remaining_count = len(remaining_files) if remaining_files else 0

        if remaining_count == 0:
            st.success("🎉 所有文件已清空")
            return len(file_paths)
        else:
            st.warning(f"⚠️ 仍有 {remaining_count} 个文件残留")
            return len(file_paths) - remaining_count

    except Exception as e:
        st.error(f"文件存储清空失败: {e}")
        return 0

def get_clickzetta_engine():
    """获取ClickZetta引擎"""
    config = load_env_config()
    return ClickZettaEngine(
        service=config['clickzetta_service'],
        instance=config['clickzetta_instance'],
        workspace=config['clickzetta_workspace'],
        schema=config['clickzetta_schema'],
        username=config['clickzetta_username'],
        password=config['clickzetta_password'],
        vcluster=config['clickzetta_vcluster']
    )

class WebCrawlerDemo:
    """网络爬虫演示类"""

    def __init__(self, engine):
        self.engine = engine
        self.setup_storage_services()

    def setup_storage_services(self):
        """初始化所有存储服务"""
        try:
            # 文档存储 - 存储网页内容和元数据
            self.doc_store = ClickZettaDocumentStore(
                engine=self.engine,
                table_name="web_crawler_documents"
            )

            # 键值存储 - 存储爬取状态和缓存
            self.cache_store = ClickZettaStore(
                engine=self.engine,
                table_name="web_crawler_cache"
            )

            # 文件存储 - 存储原始HTML文件
            self.file_store = ClickZettaFileStore(
                engine=self.engine,
                volume_type="user",
                subdirectory="web_crawler_files"
            )

            # 向量存储 - 语义搜索
            if st.session_state.get('dashscope_available'):
                embeddings = DashScopeEmbeddings(
                    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
                    model="text-embedding-v4"
                )
                self.vector_store = ClickZettaVectorStore(
                    engine=self.engine,
                    embeddings=embeddings,
                    table_name="web_crawler_vectors"
                )
            else:
                self.vector_store = None

        except Exception as e:
            st.error(f"存储服务初始化失败: {e}")

    def crawl_url(self, url: str, progress_callback=None) -> Dict:
        """爬取单个URL"""
        result = {
            "success": False,
            "url": url,
            "error": None,
            "content": "",
            "metadata": {},
            "storage_status": {}
        }

        try:
            # 验证URL
            if not validators.url(url):
                result["error"] = "无效的URL格式"
                return result

            if progress_callback:
                progress_callback("正在发送请求...")

            # 发送HTTP请求
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            if progress_callback:
                progress_callback("正在解析HTML...")

            # 解析HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # 提取内容和元数据
            text_content = extract_text_content(response.text)
            metadata = extract_metadata(soup, url)

            # 计算URL键
            url_key = url_to_key(url)

            if progress_callback:
                progress_callback("正在存储到ClickZetta...")

            # 存储到各种存储服务
            storage_status = self.store_crawled_data(
                url_key, url, text_content, metadata, response.text
            )

            result.update({
                "success": True,
                "content": text_content,
                "metadata": metadata,
                "storage_status": storage_status
            })

        except requests.RequestException as e:
            result["error"] = f"网络请求失败: {e}"
        except Exception as e:
            result["error"] = f"爬取失败: {e}"

        return result

    def store_crawled_data(self, url_key: str, url: str, content: str,
                          metadata: Dict, raw_html: str) -> Dict:
        """将爬取的数据存储到各种存储服务"""
        storage_status = {}

        try:
            # 1. 文档存储 - 存储处理后的内容和元数据
            self.doc_store.store_document(
                doc_id=url_key,
                content=content,
                metadata=metadata
            )
            storage_status["document_store"] = "✅ 成功"

            # 2. 键值存储 - 存储爬取状态
            cache_data = [
                (f"crawl_status:{url_key}", b"completed"),
                (f"url_mapping:{url_key}", url.encode()),
                (f"crawl_time:{url_key}", datetime.now().isoformat().encode()),
                (f"content_hash:{url_key}", hashlib.md5(content.encode()).hexdigest().encode())
            ]
            self.cache_store.mset(cache_data)
            storage_status["cache_store"] = "✅ 成功"

            # 3. 文件存储 - 存储原始HTML
            self.file_store.store_file(
                file_path=f"{url_key}.html",
                content=raw_html.encode(),
                mime_type="text/html"
            )
            storage_status["file_store"] = "✅ 成功"

            # 4. 向量存储 - 语义搜索（如果可用）
            if self.vector_store and content.strip():
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                self.vector_store.add_documents([doc])
                storage_status["vector_store"] = "✅ 成功"
            else:
                storage_status["vector_store"] = "⚠️ 跳过 (DashScope不可用或内容为空)"

        except Exception as e:
            storage_status["error"] = f"❌ 存储失败: {e}"

        return storage_status

    def search_documents(self, query: str, search_type: str = "semantic") -> List[Dict]:
        """搜索已爬取的文档"""
        results = []

        try:
            if search_type == "semantic" and self.vector_store:
                # 语义搜索
                docs = self.vector_store.similarity_search(query, k=5)
                for doc in docs:
                    results.append({
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "metadata": doc.metadata,
                        "search_type": "语义搜索"
                    })

            elif search_type == "keyword":
                # 关键词搜索 (简单实现)
                # 这里可以使用ClickZetta的全文搜索功能
                query_result, _ = self.engine.execute_query(f"""
                    SELECT doc_id, doc_content, metadata
                    FROM {self.doc_store.table_name}
                    WHERE doc_content LIKE '%{query}%'
                    LIMIT 5
                """)

                for row in query_result:
                    try:
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    except:
                        metadata = {}

                    results.append({
                        "content": row['doc_content'][:500] + "..." if len(row['doc_content']) > 500 else row['doc_content'],
                        "metadata": metadata,
                        "search_type": "关键词搜索"
                    })

        except Exception as e:
            st.error(f"搜索失败: {e}")

        return results

    def get_storage_stats(self) -> Dict:
        """获取存储统计信息"""
        stats = {}

        try:
            # 文档统计
            doc_result, _ = self.engine.execute_query(f"""
                SELECT COUNT(*) as doc_count,
                       AVG(LENGTH(doc_content)) as avg_content_length
                FROM {self.doc_store.table_name}
            """)
            stats["documents"] = doc_result[0] if doc_result else {"doc_count": 0, "avg_content_length": 0}

            # 缓存统计
            cache_result, _ = self.engine.execute_query(f"""
                SELECT COUNT(*) as cache_count
                FROM {self.cache_store.table_name}
            """)
            stats["cache"] = cache_result[0] if cache_result else {"cache_count": 0}

            # 文件统计 (通过 Volume Store API)
            try:
                # list_files() 返回 list[tuple[str, int, str]] 格式: (文件路径, 文件大小, 内容类型)
                files = self.file_store.list_files()
                file_count = len(files) if files else 0
                total_size = 0
                if files:
                    for file_info in files:
                        if isinstance(file_info, tuple) and len(file_info) >= 2:
                            # file_info[1] 是文件大小
                            file_size = file_info[1]
                            total_size += file_size

                stats["files"] = {
                    "file_count": file_count,
                    "total_size": total_size,
                    "total_size_kb": round(total_size / 1024, 2) if total_size > 0 else 0
                }
            except Exception as e:
                stats["files"] = {"file_count": f"获取失败: {e}", "total_size": 0}

        except Exception as e:
            st.warning(f"统计信息获取失败: {e}")
            stats = {"error": str(e)}

        return stats

def show_help_documentation():
    """显示帮助文档"""
    st.markdown("# 🕷️ ClickZetta 网络爬虫与存储系统 - 学习指南")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 功能概述",
        "🔧 技术架构",
        "💡 实际应用",
        "📚 学习资源"
    ])

    with tab1:
        st.markdown("## 🎯 功能概述")

        st.markdown("""
        ### 🌟 什么是ClickZetta网络爬虫与存储系统？

        这是一个完整的网络数据采集和存储解决方案，展示了如何使用ClickZetta的**四大存储服务**来构建现代化的数据pipeline：

        **🔄 数据流程**：网页抓取 → 内容解析 → 多存储协同 → 智能检索
        """)

        # 核心功能展示
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **📥 数据采集功能**
            - 🕷️ 智能网页爬取
            - 📄 HTML内容解析
            - 🏷️ 元数据提取
            - 🔍 内容清理优化
            """)

        with col2:
            st.markdown("""
            **💾 存储与检索**
            - 📚 结构化文档存储
            - 🗃️ 高速键值缓存
            - 📁 原始文件保存
            - 🔍 AI语义搜索
            """)

        st.markdown("""
        ### 🏗️ 四大存储服务协同工作

        本系统独特地展示了ClickZetta四种存储服务如何协同工作，就像一个完整的图书馆系统：
        """)

        storage_services = [
            {
                "name": "📚 ClickZettaDocumentStore",
                "description": "结构化文档库",
                "analogy": "就像图书馆的主要书架，存储书籍内容和详细信息",
                "data": "网页正文、标题、作者、发布时间等结构化信息",
                "table": "web_crawler_documents表"
            },
            {
                "name": "🗃️ ClickZettaStore",
                "description": "键值缓存系统",
                "analogy": "就像图书馆的索引卡片，快速查找书籍状态和位置",
                "data": "爬取状态、URL映射、内容哈希、更新时间",
                "table": "web_crawler_cache表"
            },
            {
                "name": "📁 ClickZettaFileStore",
                "description": "原始文件仓库",
                "analogy": "就像图书馆的档案室，保存原始文档和手稿",
                "data": "完整HTML源码、CSS、JavaScript等原始文件",
                "table": "Volume存储（文件系统）"
            },
            {
                "name": "🔍 ClickZettaVectorStore",
                "description": "AI语义搜索",
                "analogy": "就像图书馆的智能推荐系统，根据内容相似性推荐相关书籍",
                "data": "文本向量化表示，支持语义相似性搜索",
                "table": "web_crawler_vectors表"
            }
        ]

        for service in storage_services:
            with st.expander(f"{service['name']} - {service['description']}"):
                st.write(f"**生活化理解**: {service['analogy']}")
                st.write(f"**存储内容**: {service['data']}")
                st.write(f"**存储位置**: {service['table']}")

        st.success("💡 **核心优势**: 四种存储服务各司其职，既保证了数据的完整性，又优化了不同场景下的查询性能！")

    with tab2:
        st.markdown("## 🔧 技术架构")

        st.markdown("### 🛠️ 核心技术栈")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🌐 网络爬取技术**
            - `requests`: HTTP请求处理
            - `BeautifulSoup`: HTML解析
            - `html2text`: 内容清理
            - `validators`: URL验证
            """)

        with col2:
            st.markdown("""
            **🤖 AI与存储技术**
            - `ClickZetta`: 统一数据存储平台
            - `LangChain`: AI应用框架
            - `DashScope`: 阿里云向量化服务
            - `Streamlit`: 可视化界面
            """)

        st.markdown("### 🏗️ 系统架构图")

        st.markdown("""
        ```
        🌐 网页输入
            ↓
        🕷️ 网络爬虫 (requests + BeautifulSoup)
            ↓
        📄 内容解析 (html2text + metadata extraction)
            ↓
        ┌─────────────────────────────────────────┐
        │           ClickZetta存储层               │
        ├─────────────────────────────────────────┤
        │ 📚 DocumentStore │ 🗃️ Store (Cache)     │
        │ (结构化文档)      │ (状态与映射)          │
        │                  │                      │
        │ 📁 FileStore     │ 🔍 VectorStore       │
        │ (原始文件)        │ (语义搜索)            │
        └─────────────────────────────────────────┘
            ↓
        🔍 智能检索 (关键词 + 语义搜索)
            ↓
        📊 可视化展示 (Streamlit)
        ```
        """)

        st.markdown("### 🔄 数据处理流程")

        process_steps = [
            {
                "step": "1️⃣ 网页抓取",
                "description": "使用requests获取HTML内容，BeautifulSoup解析DOM结构",
                "code": """
# 发送HTTP请求
response = requests.get(url, headers=headers, timeout=30)

# 解析HTML内容
soup = BeautifulSoup(response.content, 'html.parser')
"""
            },
            {
                "step": "2️⃣ 内容提取",
                "description": "提取正文内容和元数据信息",
                "code": """
# 提取正文
text_content = extract_text_content(response.text)

# 提取元数据
metadata = extract_metadata(soup, url)
"""
            },
            {
                "step": "3️⃣ 多存储写入",
                "description": "同时写入四种存储服务，确保数据完整性",
                "code": """
# 文档存储
doc_store.store_document(url_key, content, metadata)

# 键值存储
cache_store.mset([(status_key, status), (url_key, url)])

# 文件存储
file_store.store_file(f"{url_key}.html", raw_html)

# 向量存储
vector_store.add_documents([Document(content, metadata)])
"""
            },
            {
                "step": "4️⃣ 智能检索",
                "description": "支持关键词和语义两种搜索方式",
                "code": """
# 语义搜索
docs = vector_store.similarity_search(query, k=5)

# 关键词搜索
results = engine.execute_query(
    f"SELECT * FROM documents WHERE content LIKE '%{query}%'"
)
"""
            }
        ]

        for step_info in process_steps:
            with st.expander(step_info["step"]):
                st.write(step_info["description"])
                st.code(step_info["code"], language="python")

        st.markdown("### 🔍 表结构详情")

        if st.button("🗄️ 查看存储表结构", key="crawler_table_structure"):
            st.markdown("""
            **📚 web_crawler_documents表 (DocumentStore)**
            ```sql
            - doc_id: VARCHAR      # URL的MD5哈希值
            - doc_content: TEXT    # 提取的网页正文
            - metadata: JSON       # 标题、作者、发布时间等元数据
            - created_at: TIMESTAMP
            ```

            **🗃️ web_crawler_cache表 (Store)**
            ```sql
            - key: VARCHAR         # 缓存键 (如: crawl_status:xxxxx)
            - value: BYTEA         # 缓存值 (状态、URL映射等)
            - created_at: TIMESTAMP
            ```

            **📁 FileStore (Volume存储)**
            ```
            - 文件路径: web_crawler_files/{url_hash}.html
            - 内容类型: text/html
            - 文件大小: 自动计算
            ```

            **🔍 web_crawler_vectors表 (VectorStore)**
            ```sql
            - id: VARCHAR          # 文档ID
            - embedding: FLOAT[]   # 1536维向量 (text-embedding-v4)
            - document: TEXT       # 原始文档内容
            - metadata: JSON       # 文档元数据
            ```
            """)

    with tab3:
        st.markdown("## 💡 实际应用场景")

        st.markdown("### 🌟 企业级应用场景")

        use_cases = [
            {
                "title": "📰 媒体内容聚合",
                "description": "新闻机构可以使用此系统监控多个新闻源，自动采集、分析和分发新闻内容",
                "benefits": ["实时新闻更新", "重复内容去重", "智能内容推荐", "历史数据查询"],
                "example": "每日自动爬取100+新闻网站，智能分类和推荐相关新闻"
            },
            {
                "title": "🏢 企业知识管理",
                "description": "企业可以爬取内外部文档、政策、流程，构建智能知识库系统",
                "benefits": ["文档自动收集", "智能搜索引擎", "版本变更追踪", "知识图谱构建"],
                "example": "爬取公司内网文档，员工可通过自然语言查询相关政策"
            },
            {
                "title": "🛍️ 电商价格监控",
                "description": "电商平台可以监控竞品价格、库存、评价等信息，制定动态定价策略",
                "benefits": ["实时价格追踪", "库存状态监控", "用户评价分析", "市场趋势预测"],
                "example": "监控1000+竞品商品，自动调整价格策略"
            },
            {
                "title": "🔬 学术研究助手",
                "description": "研究人员可以自动收集论文、报告，进行文献综述和研究趋势分析",
                "benefits": ["论文自动收集", "研究趋势分析", "引用关系挖掘", "重复研究避免"],
                "example": "爬取arXiv、IEEE等平台，为AI研究提供文献支持"
            },
            {
                "title": "🏛️ 政府公开信息监控",
                "description": "监控政府网站的政策更新、公告发布，及时响应政策变化",
                "benefits": ["政策实时监控", "法规变更提醒", "公开信息归档", "影响评估分析"],
                "example": "监控政府官网，自动提取政策文件并分析对企业的影响"
            }
        ]

        for i, case in enumerate(use_cases):
            with st.expander(f"{case['title']}", expanded=i==0):
                st.write(f"**应用描述**: {case['description']}")
                st.write(f"**实际案例**: {case['example']}")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**核心优势**:")
                    for benefit in case['benefits']:
                        st.write(f"- {benefit}")

                with col2:
                    if i == 0:  # 媒体内容
                        st.code("""
# 新闻爬取示例
urls = [
    "https://news.sina.com.cn",
    "https://news.163.com",
    "https://www.thepaper.cn"
]

for url in urls:
    result = crawler.crawl_url(url)
    # 自动分类和去重
    classify_news(result)
""", language="python")
                    elif i == 1:  # 知识管理
                        st.code("""
# 知识库构建
knowledge_sources = [
    "内部文档系统",
    "政策制度网站",
    "行业标准文档"
]

# 智能问答
query = "公司差旅报销政策"
results = search_documents(query, "semantic")
""", language="python")
                    elif i == 2:  # 电商监控
                        st.code("""
# 价格监控
products = ["iPhone 15", "华为Mate60"]

for product in products:
    price_data = crawl_ecommerce_data(product)
    if price_changed(price_data):
        update_pricing_strategy(product)
""", language="python")
                    elif i == 3:  # 学术研究
                        st.code("""
# 论文收集
keywords = ["machine learning", "deep learning"]

papers = crawl_academic_papers(keywords)
trends = analyze_research_trends(papers)
generate_literature_review(trends)
""", language="python")
                    else:  # 政府监控
                        st.code("""
# 政策监控
gov_sites = [
    "工信部官网",
    "发改委网站",
    "央行官网"
]

policies = monitor_policy_changes(gov_sites)
analyze_business_impact(policies)
""", language="python")

        st.markdown("### ⚡ 性能优势")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🚀 ClickZetta存储优势**
            - **10倍性能提升**: 云原生架构
            - **无缝扩展**: GB到PB级数据
            - **ACID事务**: 确保数据一致性
            - **SQL兼容**: 熟悉的查询语法
            """)

        with col2:
            st.markdown("""
            **🤖 AI能力集成**
            - **语义搜索**: 理解内容含义
            - **智能推荐**: 基于相似性匹配
            - **自动分类**: 内容智能标签
            - **趋势分析**: 数据洞察挖掘
            """)

    with tab4:
        st.markdown("## 📚 学习资源")

        st.markdown("### 📖 官方文档")

        doc_links = [
            {
                "title": "ClickZetta 官方文档",
                "url": "https://www.yunqi.tech/documents/",
                "description": "完整的ClickZetta平台使用指南"
            },
            {
                "title": "LangChain 文档",
                "url": "https://python.langchain.com/",
                "description": "LangChain框架完整文档"
            },
            {
                "title": "DashScope API文档",
                "url": "https://help.aliyun.com/zh/dashscope/",
                "description": "阿里云大模型服务API文档"
            },
            {
                "title": "Streamlit 文档",
                "url": "https://docs.streamlit.io/",
                "description": "Streamlit应用开发文档"
            }
        ]

        for doc in doc_links:
            st.markdown(f"- **[{doc['title']}]({doc['url']})**: {doc['description']}")

        st.markdown("### 🛠️ 快速开始")

        st.markdown("""
        #### 1️⃣ 环境准备
        ```bash
        # 安装依赖
        pip install -r requirements.txt

        # 配置环境变量
        cp .env.example .env
        # 编辑.env文件，填入ClickZetta和DashScope配置
        ```

        #### 2️⃣ 运行应用
        ```bash
        # 启动应用
        streamlit run streamlit_app.py

        # 打开浏览器访问 http://localhost:8501
        ```

        #### 3️⃣ 开始爬取
        1. 在"🕷️ 网页爬取"标签页输入URL
        2. 点击"🚀 开始爬取"按钮
        3. 在"🔍 内容搜索"标签页测试搜索功能
        4. 在"📊 存储统计"标签页查看数据状态

        #### 4️⃣ 文件管理操作 ⭐
        **重要提示：可靠的文件删除方法**

        当需要清空存储数据时，请使用以下操作：
        1. 点击侧边栏的"🗑️ 清空所有数据"按钮
        2. 系统会自动使用**优化的删除策略**：
           - 识别所有相关文件键（主键 + 元数据键）
           - 一次性批量删除所有相关数据
           - 确保文件完全从存储中移除

        **技术说明**：
        - 🔑 **关键发现**: 每个文件有两个键（主键 + `_metadata_`键），必须同时删除
        - ✅ **有效方法**: 系统已优化为直接使用 `volume_store.mdelete()`
        - 🎯 **删除策略**: 自动识别并删除所有相关键，确保完全清理

        ```python
        # 系统实现的删除方式
        keys_to_delete = []
        for file_path in file_paths:
            keys_to_delete.append(file_path)              # 主文件键
            keys_to_delete.append(f"_metadata_{file_path}") # 元数据键

        # 一次性删除所有相关键
        crawler.file_store.volume_store.mdelete(keys_to_delete)
        ```
        """)

        st.markdown("### 🔧 自定义开发")

        st.markdown("""
        #### 扩展爬虫功能
        ```python
        # 自定义爬虫类
        class CustomWebCrawler(WebCrawlerDemo):
            def custom_parse_content(self, soup):
                # 添加自定义解析逻辑
                pass

            def custom_metadata_extraction(self, soup):
                # 添加自定义元数据提取
                pass
        ```

        #### 扩展存储功能
        ```python
        # 添加新的存储服务
        custom_store = ClickZettaCustomStore(
            engine=engine,
            table_name="custom_table"
        )

        # 自定义数据处理流程
        def custom_data_pipeline(data):
            # 添加数据预处理逻辑
            processed_data = preprocess(data)

            # 存储到自定义表
            custom_store.store(processed_data)
        ```
        """)

        st.markdown("### 💡 最佳实践")

        best_practices = [
            {
                "category": "🕷️ 爬虫优化",
                "tips": [
                    "设置合理的请求间隔，避免被反爬虫系统拦截",
                    "使用代理池和User-Agent轮换提高成功率",
                    "针对不同网站设计专门的解析策略",
                    "实施断点续爬功能，提高大批量任务的稳定性"
                ]
            },
            {
                "category": "💾 存储策略",
                "tips": [
                    "根据查询模式选择合适的存储服务组合",
                    "设计合理的数据分区策略，提高查询性能",
                    "定期清理过期数据，控制存储成本",
                    "建立数据备份和恢复机制"
                ]
            },
            {
                "category": "🔍 搜索优化",
                "tips": [
                    "为不同类型内容设计专门的向量化策略",
                    "结合关键词和语义搜索，提供更全面的结果",
                    "建立搜索结果相关性评分机制",
                    "实施搜索日志分析，持续优化搜索质量"
                ]
            },
            {
                "category": "🚀 性能优化",
                "tips": [
                    "使用异步并发爬取，提高数据采集效率",
                    "实施智能缓存策略，减少重复计算",
                    "优化数据库查询，使用索引和查询优化",
                    "监控系统性能，及时发现和解决瓶颈"
                ]
            }
        ]

        for practice in best_practices:
            with st.expander(practice["category"]):
                for tip in practice["tips"]:
                    st.write(f"- {tip}")

        st.markdown("### 🤝 社区支持")

        st.markdown("""
        - **技术交流**: 加入ClickZetta技术交流群
        - **问题反馈**: 通过GitHub Issues报告问题
        - **功能建议**: 提交功能需求和改进建议
        - **案例分享**: 分享你的应用案例和最佳实践
        """)

        st.success("🎉 恭喜！你已经掌握了ClickZetta网络爬虫与存储系统的核心知识。现在就开始构建你自己的智能数据采集系统吧！")

def main():
    """主函数"""
    # 页面导航
    page_selection = st.selectbox(
        "选择功能页面",
        ["🕷️ 网络爬虫", "📚 学习指南"],
        key="crawler_page_selection"
    )

    if page_selection == "📚 学习指南":
        show_help_documentation()
        return

    # 原有的主要功能界面
    st.title("🕷️ ClickZetta Web Crawler & Storage Demo")
    st.markdown("### 展示LangChain网站爬取与ClickZetta存储服务的完整集成")

    # 初始化session state
    if 'crawl_results' not in st.session_state:
        st.session_state.crawl_results = []
    if 'show_content' not in st.session_state:
        st.session_state.show_content = {}

    # 检查依赖
    if not DEPENDENCIES_AVAILABLE:
        st.error(f"缺少必要依赖: {import_error}")
        st.info("请运行: pip install -r requirements.txt")
        return

    # 检查环境配置
    env_status = check_environment()
    display_environment_status(env_status)

    if not env_status["clickzetta_available"]:
        st.error("ClickZetta配置不完整，无法使用存储服务")
        return

    # 创建侧边栏信息
    create_sidebar_info()

    # 初始化ClickZetta引擎
    try:
        engine = get_clickzetta_engine()
        crawler = WebCrawlerDemo(engine)
        st.success("✅ ClickZetta存储服务初始化成功")

        # 处理数据清空请求
        if st.session_state.get('clear_data_requested', False):
            st.info("正在清空数据...")

            try:
                # 清空各种存储表
                tables_to_clear = [
                    "web_crawler_documents",
                    "web_crawler_cache",
                    "web_crawler_vectors"
                ]

                for table in tables_to_clear:
                    try:
                        delete_query = f"DELETE FROM {table}"
                        engine.execute_query(delete_query)
                    except:
                        # 表不存在是正常的
                        pass

                # 清空文件存储（使用crawler实例）
                deleted_count = clear_file_storage(crawler)

                # 验证文件是否真的被删除
                verification_files = crawler.file_store.list_files()
                remaining_files = len(verification_files) if verification_files else 0

                if remaining_files > 0:
                    st.warning(f"⚠️ 注意：删除了 {deleted_count} 个文件，但仍有 {remaining_files} 个文件残留")

                    # 分析残留文件是否与删除的文件相同
                    remaining_paths = set()
                    if verification_files:
                        for file_info in verification_files:
                            if isinstance(file_info, tuple) and len(file_info) >= 1:
                                remaining_paths.add(file_info[0])

                    with st.expander(f"查看残留文件详情 ({len(verification_files)}个)"):
                        for file_info in verification_files:
                            if isinstance(file_info, tuple) and len(file_info) >= 1:
                                st.write(f"• {file_info[0]}")

                    # 尝试再次删除残留文件
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 尝试再次删除残留文件"):
                            try:
                                remaining_file_paths = [f[0] for f in verification_files if isinstance(f, tuple) and len(f) >= 1]
                                if remaining_file_paths:
                                    st.info(f"尝试删除 {len(remaining_file_paths)} 个残留文件...")
                                    # 包含主键和元数据键
                                    all_remaining_keys = []
                                    for key in remaining_file_paths:
                                        all_remaining_keys.append(key)
                                        all_remaining_keys.append(f"_metadata_{key}")
                                    try:
                                        crawler.file_store.mdelete(all_remaining_keys)
                                    except Exception as fs_err:
                                        st.warning(f"FileStore删除失败，尝试底层删除: {fs_err}")
                                        crawler.file_store.volume_store.mdelete(all_remaining_keys)

                                    # 再次验证
                                    final_check = crawler.file_store.list_files()
                                    final_count = len(final_check) if final_check else 0

                                    if final_count == 0:
                                        st.success("✅ 残留文件删除成功!")
                                    else:
                                        st.error(f"❌ 仍有 {final_count} 个文件无法删除，可能存在权限问题或API限制")
                            except Exception as retry_e:
                                st.error(f"❌ 再次删除失败: {retry_e}")

                else:
                    st.success("✅ 文件存储已完全清空")

                # 重置session状态
                if 'crawl_results' in st.session_state:
                    st.session_state.crawl_results = []

                # 清除清空请求标志
                st.session_state.clear_data_requested = False

                # 设置标记以便下次访问统计页面时自动刷新
                st.session_state.force_stats_refresh = True

                st.success("✅ 所有数据已清空，请重新开始爬取")
                st.rerun()

            except Exception as e:
                st.error(f"❌ 清空数据失败: {e}")
                st.session_state.clear_data_requested = False

        # 自动检测并显示已有数据统计
        if st.session_state.get('dashscope_available', False):
            try:
                # 检查各个存储服务的数据量
                storage_stats = {}

                # 检查DocumentStore数据
                try:
                    doc_count_query = f"SELECT COUNT(*) as count FROM {crawler.doc_store.table_name}"
                    doc_result, _ = engine.execute_query(doc_count_query)
                    storage_stats['documents'] = doc_result[0]['count'] if doc_result else 0
                except:
                    storage_stats['documents'] = 0

                # 检查Cache数据
                try:
                    cache_count_query = f"SELECT COUNT(*) as count FROM {crawler.cache_store.table_name}"
                    cache_result, _ = engine.execute_query(cache_count_query)
                    storage_stats['cache'] = cache_result[0]['count'] if cache_result else 0
                except:
                    storage_stats['cache'] = 0

                # 检查VectorStore数据
                try:
                    if crawler.vector_store:
                        vector_count_query = f"SELECT COUNT(*) as count FROM {crawler.vector_store.table_name}"
                        vector_result, _ = engine.execute_query(vector_count_query)
                        storage_stats['vectors'] = vector_result[0]['count'] if vector_result else 0
                    else:
                        storage_stats['vectors'] = 0
                except:
                    storage_stats['vectors'] = 0

                # 显示数据统计
                total_data = sum(storage_stats.values())
                if total_data > 0:
                    st.info(f"🎉 检测到已存在数据: 📚文档{storage_stats['documents']}条 | 🗂️缓存{storage_stats['cache']}条 | 🧠向量{storage_stats['vectors']}条，可直接使用搜索功能")
            except Exception as e:
                # 数据检测失败，不影响主要功能
                pass

    except Exception as e:
        st.error(f"❌ ClickZetta连接失败: {e}")
        return

    # 主界面标签页
    tab1, tab2, tab3, tab4 = st.tabs(["🕷️ 网页爬取", "🔍 内容搜索", "📊 存储统计", "💡 功能演示"])

    with tab1:
        st.header("网页爬取与存储")

        # URL输入
        url_input = st.text_input(
            "请输入要爬取的网页URL:",
            value="https://www.yunqi.tech",
            placeholder="https://www.yunqi.tech",
            help="支持大多数网站，建议使用新闻、博客等内容丰富的页面"
        )

        # 批量URL输入
        with st.expander("批量爬取 (每行一个URL)"):
            batch_urls = st.text_area(
                "批量URL列表:",
                placeholder="https://www.yunqi.tech\nhttps://www.yunqi.tech/products\nhttps://www.yunqi.tech/solutions",
                height=100
            )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 开始爬取", type="primary"):
                urls_to_crawl = []

                if url_input.strip():
                    urls_to_crawl.append(url_input.strip())

                if batch_urls.strip():
                    batch_list = [url.strip() for url in batch_urls.strip().split('\n') if url.strip()]
                    urls_to_crawl.extend(batch_list)

                if not urls_to_crawl:
                    st.warning("请输入至少一个URL")
                else:
                    # 开始爬取
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = []

                    for i, url in enumerate(urls_to_crawl):
                        status_text.text(f"正在爬取: {url}")

                        def update_progress(message):
                            status_text.text(f"{url}: {message}")

                        result = crawler.crawl_url(url, update_progress)
                        results.append(result)

                        progress_bar.progress((i + 1) / len(urls_to_crawl))

                    # 保存结果到session state
                    st.session_state.crawl_results = results

                    # 显示结果
                    status_text.text("爬取完成!")

                    success_count = sum(1 for r in results if r["success"])
                    st.success(f"✅ 成功爬取 {success_count}/{len(results)} 个页面")

        # 显示爬取历史结果
        if st.session_state.crawl_results:
            st.subheader("爬取结果")
            for result in st.session_state.crawl_results:
                url_key = url_to_key(result['url'])
                with st.expander(f"{'✅' if result['success'] else '❌'} {result['url']}"):
                    if result["success"]:
                        st.write("**页面信息:**")
                        metadata = result["metadata"]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"- **标题**: {metadata.get('title', 'N/A')}")
                            st.write(f"- **语言**: {metadata.get('language', 'N/A')}")
                        with col2:
                            st.write(f"- **内容长度**: {len(result['content'])} 字符")
                            st.write(f"- **爬取时间**: {metadata.get('crawled_at', 'N/A')}")

                        st.write("**存储状态:**")
                        for service, status in result["storage_status"].items():
                            st.write(f"- {service}: {status}")

                        if st.button(f"查看内容", key=f"content_{url_key}"):
                            st.session_state.show_content[url_key] = True
                            st.rerun()

                        # 显示内容（如果用户点击了查看内容）
                        if st.session_state.show_content.get(url_key, False):
                            st.text_area("内容预览:", result["content"][:1000] + "..." if len(result["content"]) > 1000 else result["content"], height=200, key=f"content_display_{url_key}")
                            if st.button(f"隐藏内容", key=f"hide_{url_key}"):
                                st.session_state.show_content[url_key] = False
                                st.rerun()
                    else:
                        st.error(f"爬取失败: {result['error']}")

            # 清空历史按钮
            if st.button("🗑️ 清空爬取历史"):
                st.session_state.crawl_results = []
                st.session_state.show_content = {}
                st.rerun()

        with col2:
            if st.button("🧹 清理所有数据"):
                confirm = st.checkbox("确认清理所有数据")
                if confirm and st.button("确认清理", type="secondary"):
                    try:
                        # 清理所有存储
                        crawler.engine.execute_query(f"TRUNCATE TABLE {crawler.doc_store.table_name}")
                        crawler.engine.execute_query(f"TRUNCATE TABLE {crawler.cache_store.table_name}")
                        # FileStore 使用 Volume Store，需要用不同的清理方法
                        # crawler.engine.execute_query(f"TRUNCATE TABLE {crawler.file_store.table_name}")
                        if crawler.vector_store:
                            crawler.engine.execute_query(f"TRUNCATE TABLE {crawler.vector_store.table_name}")
                        st.success("✅ 所有数据已清理")
                    except Exception as e:
                        st.error(f"清理失败: {e}")

    with tab2:
        st.header("内容搜索")

        search_query = st.text_input("请输入搜索关键词:", placeholder="输入要搜索的内容...")

        col1, col2 = st.columns(2)
        with col1:
            search_type = st.selectbox(
                "搜索方式:",
                ["semantic", "keyword"],
                format_func=lambda x: "语义搜索" if x == "semantic" else "关键词搜索"
            )

        if st.button("🔍 搜索", type="primary"):
            if search_query.strip():
                with st.spinner("搜索中..."):
                    results = crawler.search_documents(search_query, search_type)

                if results:
                    st.success(f"找到 {len(results)} 个相关结果")

                    for i, result in enumerate(results):
                        with st.expander(f"结果 {i+1}: {result['metadata'].get('title', '无标题')}"):
                            st.write(f"**搜索方式**: {result['search_type']}")
                            st.write(f"**URL**: {result['metadata'].get('url', 'N/A')}")
                            st.write(f"**爬取时间**: {result['metadata'].get('crawled_at', 'N/A')}")
                            st.write("**内容预览**:")
                            st.text_area("搜索结果内容", result["content"], height=150, key=f"result_{i}", label_visibility="collapsed")
                else:
                    st.info("没有找到相关结果")
            else:
                st.warning("请输入搜索关键词")

    with tab3:
        st.header("存储统计")

        # 检查是否需要强制刷新统计
        force_refresh = st.session_state.get('force_stats_refresh', False)
        if force_refresh:
            st.session_state.force_stats_refresh = False  # 清除标记
            with st.spinner("自动刷新统计信息..."):
                stats = crawler.get_storage_stats()
                st.info("📊 统计数据已自动更新")

        if st.button("🔄 刷新统计") or force_refresh:
            if not force_refresh:  # 避免重复获取
                with st.spinner("获取统计信息..."):
                    stats = crawler.get_storage_stats()

            if "error" not in stats:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "文档总数",
                        stats.get("documents", {}).get("doc_count", 0)
                    )
                    avg_length = stats.get('documents', {}).get('avg_content_length', 0) or 0
                    st.metric(
                        "平均文档长度",
                        f"{avg_length:.0f} 字符"
                    )

                with col2:
                    st.metric(
                        "缓存条目数",
                        stats.get("cache", {}).get("cache_count", 0)
                    )

                with col3:
                    st.metric(
                        "文件总数",
                        stats.get("files", {}).get("file_count", 0)
                    )
                    file_size = stats.get("files", {}).get("total_size", 0)

                    if isinstance(file_size, (int, float)) and file_size > 0:
                        if file_size >= 1024 * 1024:  # >= 1MB
                            size_display = f"{file_size / 1024 / 1024:.2f} MB"
                        elif file_size >= 1024:  # >= 1KB
                            size_display = f"{file_size / 1024:.2f} KB"
                        else:  # < 1KB
                            size_display = f"{file_size} bytes"
                    elif isinstance(file_size, str):
                        size_display = file_size
                    else:
                        size_display = "0 bytes"

                    st.metric("总文件大小", size_display)

                # 存储服务状态
                st.subheader("存储服务状态")

                services = [
                    ("文档存储", "ClickZettaDocumentStore", "存储网页内容和元数据"),
                    ("键值存储", "ClickZettaStore", "存储爬取状态和缓存"),
                    ("文件存储", "ClickZettaFileStore", "存储原始HTML文件"),
                    ("向量存储", "ClickZettaVectorStore", "语义搜索支持")
                ]

                for service_name, class_name, description in services:
                    with st.expander(f"📦 {service_name} ({class_name})"):
                        st.write(description)

                        # 显示对应的数据量
                        if service_name == "文档存储":
                            st.metric("存储文档数", stats.get("documents", {}).get("doc_count", 0))
                        elif service_name == "键值存储":
                            st.metric("缓存条目数", stats.get("cache", {}).get("cache_count", 0))
                        elif service_name == "文件存储":
                            st.metric("存储文件数", stats.get("files", {}).get("file_count", 0))
                        elif service_name == "向量存储":
                            if crawler.vector_store:
                                st.write("✅ 可用 (DashScope集成)")
                            else:
                                st.write("⚠️ 不可用 (需要DashScope API密钥)")
            else:
                st.error(f"获取统计失败: {stats['error']}")

    with tab4:
        st.header("功能演示")

        st.markdown("""
        ### 🎯 演示场景

        这个示例完整展示了ClickZetta存储服务的四大核心能力:
        """)

        # 功能卡片
        features = [
            {
                "title": "📚 文档存储 (ClickZettaDocumentStore)",
                "description": "存储结构化文档内容和元数据，支持SQL查询",
                "example": "存储网页标题、内容、作者、发布时间等信息",
                "benefits": ["SQL可查询", "元数据丰富", "结构化存储"]
            },
            {
                "title": "🗃️ 键值存储 (ClickZettaStore)",
                "description": "高性能键值对存储，适合缓存和状态管理",
                "example": "存储爬取状态、内容哈希、最后更新时间",
                "benefits": ["高性能读写", "原子操作", "批量处理"]
            },
            {
                "title": "📁 文件存储 (ClickZettaFileStore)",
                "description": "基于ClickZetta Volume的二进制文件存储",
                "example": "存储原始HTML文件、图片、PDF等",
                "benefits": ["二进制支持", "大文件优化", "版本管理"]
            },
            {
                "title": "🔍 向量存储 (ClickZettaVectorStore)",
                "description": "支持语义搜索的向量数据库功能",
                "example": "基于内容相似性的智能文档检索",
                "benefits": ["语义理解", "相似性搜索", "AI驱动"]
            }
        ]

        for i, feature in enumerate(features):
            with st.expander(f"{feature['title']}", expanded=i==0):
                st.write(feature['description'])
                st.write(f"**示例用途**: {feature['example']}")

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**主要优势**:")
                    for benefit in feature['benefits']:
                        st.write(f"- {benefit}")

                with col2:
                    if i == 0:  # 文档存储
                        st.code('''
doc_store.store_document(
    key=url_hash,
    content=page_content,
    metadata={
        "title": title,
        "url": url,
        "crawled_at": timestamp
    }
)''')
                    elif i == 1:  # 键值存储
                        st.code('''
cache_store.mset([
    ("status:123", b"completed"),
    ("hash:123", content_hash),
    ("time:123", timestamp)
])''')
                    elif i == 2:  # 文件存储
                        st.code('''
file_store.store_file(
    key="page.html",
    content=raw_html,
    content_type="text/html"
)''')
                    else:  # 向量存储
                        st.code('''
vector_store.add_documents([
    Document(
        page_content=content,
        metadata=metadata
    )
])''')

        st.markdown("""
        ### 🚀 技术优势

        1. **统一平台**: 所有存储需求在一个ClickZetta平台解决
        2. **ACID事务**: 确保数据一致性和可靠性
        3. **高性能**: ClickZetta的云原生架构提供10倍性能提升
        4. **易扩展**: 从GB到PB级数据的无缝扩展
        5. **SQL兼容**: 熟悉的SQL查询语法
        6. **AI就绪**: 原生支持向量搜索和机器学习
        """)

        st.markdown("""
        ### 📈 实际应用场景

        - **企业知识库**: 爬取并存储内部文档、政策、流程
        - **竞品分析**: 监控竞争对手网站内容变化
        - **新闻聚合**: 收集多源新闻进行分析和推荐
        - **学术研究**: 收集论文、报告进行文献综述
        - **电商监控**: 跟踪产品价格、评价、库存变化
        """)

if __name__ == "__main__":
    main()