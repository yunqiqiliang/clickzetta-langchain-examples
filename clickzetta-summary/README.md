# ClickZetta 文档智能摘要

基于 ClickZetta 向量存储的企业级文档摘要系统，支持中英文文档的智能摘要生成。

## ✨ 功能特性

- 🚀 **ClickZetta 向量存储** - 高性能向量检索和存储
- 🧠 **多模型支持** - 支持 DashScope (通义千问) 和 OpenAI 模型
- 🌏 **中文优化** - 专门针对中文文档的处理优化
- 📊 **灵活配置** - 支持多种摘要风格和长度设置
- 🎯 **企业级** - 连接池管理、错误处理、状态监控

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
# ClickZetta 配置
CLICKZETTA_SERVICE=your-service
CLICKZETTA_INSTANCE=your-instance
CLICKZETTA_WORKSPACE=your-workspace
CLICKZETTA_SCHEMA=your-schema
CLICKZETTA_USERNAME=your-username
CLICKZETTA_PASSWORD=your-password
CLICKZETTA_VCLUSTER=your-vcluster

# AI 模型配置 (选择其一)
DASHSCOPE_API_KEY=your-dashscope-key  # 推荐，中文效果更好
OPENAI_API_KEY=your-openai-key
```

### 3. 运行应用

```bash
streamlit run streamlit_app.py
```

## 📖 使用说明

### 基本使用流程

1. **配置连接** - 在侧边栏配置 ClickZetta 连接参数
2. **选择模型** - 选择 DashScope 或 OpenAI 模型
3. **上传文档** - 支持 PDF 格式文档
4. **设置摘要** - 选择语言、长度和风格
5. **生成摘要** - 点击"开始摘要"按钮

### 摘要配置选项

#### 模型提供商
- **DashScope (推荐)**: 通义千问系列模型，中文处理效果更好
- **OpenAI**: GPT 系列模型，英文处理效果优秀

#### 摘要设置
- **语言**: 中文、English、自动检测
- **长度**: 100-500 字可调节
- **风格**: 简洁概述、详细分析、要点列表

## 🏗️ 技术架构

### 核心组件

```python
# ClickZetta 引擎
engine = ClickZettaEngine(
    service="your-service",
    instance="your-instance",
    # ... 其他配置
)

# 向量存储
vectorstore = ClickZettaVectorStore(
    engine=engine,
    embeddings=embeddings,
    table_name="document_summary_vectors"
)

# 文档处理流程
documents → 向量化 → 存储到ClickZetta → 检索相关内容 → 生成摘要
```

### 数据流程

1. **文档加载**: PyPDFLoader 解析 PDF 文档
2. **向量化**: DashScope/OpenAI 嵌入模型生成向量
3. **存储**: ClickZetta 向量存储保存文档向量
4. **检索**: 相似性搜索获取相关文档片段
5. **摘要**: 语言模型基于检索内容生成摘要

## 🎯 与原版对比

| 特性 | 原版 (Chroma) | ClickZetta 版本 |
|------|---------------|-----------------|
| 向量存储 | Chroma (内存) | ClickZetta (企业级) |
| 数据持久化 | 临时存储 | 永久存储 |
| 中文支持 | 基础 | 专门优化 |
| 模型选择 | OpenAI only | DashScope + OpenAI |
| 配置管理 | 简单 | 企业级配置 |
| 错误处理 | 基础 | 完善的错误处理 |
| 性能监控 | 无 | 状态监控面板 |

## 🔧 高级配置

### 连接池优化

```python
engine = ClickZettaEngine(
    # ... 基础配置
    connection_timeout=60,
    query_timeout=1800,
    hints={
        "sdk.job.timeout": 3600,
        "query_tag": "document_summary"
    }
)
```

### 自定义摘要提示词

可以通过修改 `summary_prompt` 来自定义摘要生成逻辑：

```python
custom_prompt = PromptTemplate(
    input_variables=["text"],
    template="请根据以下内容生成执行摘要：\n\n{text}\n\n摘要："
)
```

## 📊 性能优化建议

1. **文档大小**: 建议单个文档不超过 10MB
2. **向量维度**: 根据需求选择合适的嵌入模型
3. **检索数量**: 调整 similarity_search 的 k 值
4. **连接复用**: 使用单例模式复用 ClickZetta 连接

## ❓ 常见问题

### Q: 如何处理大型文档？
A: 系统会自动将大型文档分割为页面，并使用向量检索选择最相关的内容进行摘要。

### Q: 为什么推荐使用 DashScope？
A: DashScope 的通义千问模型对中文文档的理解和处理能力更强，生成的中文摘要质量更高。

### Q: 文档数据会被永久保存吗？
A: 是的，文档向量会存储在 ClickZetta 数据库中，便于后续检索和分析。

### Q: 如何自定义摘要风格？
A: 可以在代码中修改 `style_instructions` 字典，添加自定义的摘要风格。

## 📞 技术支持

如有问题，请联系：
- GitHub Issues: [项目问题反馈](https://github.com/yunqiqiliang/langchain-clickzetta/issues)
- 企业支持: 联系云器科技团队

---

🚀 **Powered by ClickZetta + LangChain**