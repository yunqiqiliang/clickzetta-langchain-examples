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
                            # 调试信息：显示每个文件的详细信息
                            # st.write(f"文件: {file_info[0]}, 大小: {file_size} bytes")

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

def main():
    """主函数"""
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

        if st.button("🔄 刷新统计"):
            with st.spinner("获取统计信息..."):
                stats = crawler.get_storage_stats()

            if "error" not in stats:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "文档总数",
                        stats.get("documents", {}).get("doc_count", 0)
                    )
                    st.metric(
                        "平均文档长度",
                        f"{stats.get('documents', {}).get('avg_content_length', 0):.0f} 字符"
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