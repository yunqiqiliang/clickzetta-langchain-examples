# ClickZetta Web Crawler & Storage Demo

这个示例展示了如何使用LangChain的网站爬取功能结合ClickZetta的强大存储服务，创建一个完整的网络内容收集、处理和存储系统。

## 🌟 功能特色

### 🕷️ 智能网络爬取
- **多种爬取方式**: 支持单页面、多页面批量爬取
- **内容提取**: 自动提取标题、正文、元数据
- **错误处理**: 优雅处理网络错误和无效URL
- **爬取历史**: 避免重复爬取相同内容

### 💾 ClickZetta存储服务展示
- **文档存储**: 使用ClickZettaDocumentStore存储网页内容
- **键值存储**: 使用ClickZettaStore缓存爬取状态
- **文件存储**: 使用ClickZettaFileStore保存原始HTML
- **向量存储**: 使用ClickZettaVectorStore进行语义搜索

### 🔍 高级功能
- **内容分析**: 自动摘要和关键词提取
- **语义搜索**: 基于内容相似性的智能检索
- **全文搜索**: 传统关键词搜索
- **混合搜索**: 结合向量和全文搜索的最佳结果

## 🏗️ 架构设计

```
网页URL → 爬取器 → 内容提取 → 多层存储
                                    ↓
                              ┌─────────────┐
                              │ ClickZetta  │
                              │ 存储服务    │
                              └─────────────┘
                                    ↓
                    ┌─────────┬─────────┬─────────┬─────────┐
                    │文档存储  │键值存储  │文件存储  │向量存储  │
                    │原始内容  │爬取状态  │HTML文件 │语义搜索  │
                    └─────────┴─────────┴─────────┴─────────┘
```

## 🚀 使用方法

### 启动应用
```bash
# 从项目根目录启动
./start.sh web-crawler

# 或直接启动
cd clickzetta-web-crawler
streamlit run streamlit_app.py
```

### 环境配置
确保`.env`文件包含：
```bash
CLICKZETTA_SERVICE=your-service
CLICKZETTA_INSTANCE=your-instance
CLICKZETTA_WORKSPACE=your-workspace
CLICKZETTA_SCHEMA=your-schema
CLICKZETTA_USERNAME=your-username
CLICKZETTA_PASSWORD=your-password
CLICKZETTA_VCLUSTER=your-vcluster
DASHSCOPE_API_KEY=your-dashscope-api-key
```

## 📊 存储服务演示

### 1. 文档存储 (ClickZettaDocumentStore)
```python
# 存储网页内容和元数据
doc_store.store_document(
    key=url_hash,
    content=page_content,
    metadata={
        "url": url,
        "title": title,
        "crawled_at": timestamp,
        "word_count": len(content.split()),
        "language": detected_language
    }
)
```

### 2. 键值存储 (ClickZettaStore)
```python
# 缓存爬取状态
cache_store.mset([
    (f"crawl_status:{url_hash}", b"completed"),
    (f"last_modified:{url_hash}", last_modified.encode()),
    (f"content_hash:{url_hash}", content_hash.encode())
])
```

### 3. 文件存储 (ClickZettaFileStore)
```python
# 保存原始HTML文件
file_store.store_file(
    key=f"{url_hash}.html",
    content=raw_html.encode(),
    content_type="text/html"
)
```

### 4. 向量存储 (ClickZettaVectorStore)
```python
# 存储文档向量用于语义搜索
vector_store.add_documents([
    Document(
        page_content=processed_content,
        metadata=enriched_metadata
    )
])
```

## 🎯 演示场景

### 场景1: 新闻网站爬取
- 爬取新闻网站的文章
- 自动提取标题、作者、发布时间
- 进行内容摘要和情感分析
- 支持按主题、时间、情感搜索

### 场景2: 技术文档收集
- 爬取技术博客和文档
- 提取代码片段和技术要点
- 建立知识库支持技术问答
- 相似技术文章推荐

### 场景3: 电商产品信息
- 爬取产品页面信息
- 提取价格、规格、评价
- 价格趋势分析
- 产品对比和推荐

## 🔧 技术特性

### LangChain集成
- **WebBaseLoader**: 网页内容加载
- **RecursiveUrlLoader**: 递归爬取
- **BeautifulSoup**: HTML解析
- **Document transformers**: 内容处理

### ClickZetta优势
- **ACID事务**: 确保数据一致性
- **并发安全**: 支持多爬虫并发
- **查询性能**: SQL优化的数据检索
- **存储成本**: 高效的数据压缩

### 智能功能
- **内容去重**: 基于内容哈希避免重复
- **增量更新**: 仅爬取变更内容
- **失效检测**: 自动检测失效链接
- **内容分类**: AI驱动的自动分类

## 📈 性能监控

应用内置性能监控面板：
- 爬取速度统计
- 存储空间使用
- 查询响应时间
- 错误率分析

## 🔍 搜索能力

### 多模式搜索
1. **关键词搜索**: 传统全文检索
2. **语义搜索**: 基于内容理解的相似性
3. **混合搜索**: 结合关键词和语义的最佳匹配
4. **高级过滤**: 按时间、来源、类型筛选

### 搜索优化
- 搜索结果排序算法
- 查询扩展和同义词
- 搜索建议和自动完成
- 个性化推荐

## 💡 扩展建议

1. **定时爬取**: 集成调度系统定期更新
2. **分布式爬取**: 多节点并行处理
3. **内容分析**: 集成更多AI分析功能
4. **数据导出**: 支持多种格式导出
5. **API接口**: 提供RESTful API访问

## 🎓 学习价值

通过这个示例，您将学会：
- ClickZetta存储服务的实际应用
- 大规模网络数据的处理模式
- LangChain生态的深度集成
- 企业级数据存储最佳实践
- AI驱动的内容理解和检索

这个示例完美展示了ClickZetta在处理非结构化数据方面的强大能力，是学习现代数据栈的绝佳起点！