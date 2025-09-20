#!/bin/bash

# ClickZetta LangChain Examples 快速启动脚本
#
# 使用方法:
#   ./start.sh summary      # 启动文档摘要应用
#   ./start.sh qa          # 启动问答系统
#   ./start.sh search      # 启动混合搜索
#   ./start.sh sql         # 启动SQL问答
#   ./start.sh crawler     # 启动网络爬虫应用
#   ./start.sh             # 显示帮助信息

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示 logo
show_logo() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              ClickZetta LangChain Examples                   ║"
    echo "║                                                              ║"
    echo "║    企业级 AI 应用示例 - 基于 ClickZetta + DashScope          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 显示帮助信息
show_help() {
    echo -e "${YELLOW}使用方法:${NC}"
    echo "  ./start.sh <app_name>"
    echo ""
    echo -e "${YELLOW}可用应用:${NC}"
    echo -e "  ${GREEN}summary${NC}  - 文档智能摘要系统"
    echo -e "  ${GREEN}qa${NC}       - 智能问答系统"
    echo -e "  ${GREEN}search${NC}   - 混合搜索系统"
    echo -e "  ${GREEN}sql${NC}      - SQL 智能问答系统"
    echo -e "  ${GREEN}crawler${NC}  - 网络爬虫存储演示"
    echo ""
    echo -e "${YELLOW}示例:${NC}"
    echo "  ./start.sh summary    # 启动文档摘要应用"
    echo "  ./start.sh qa         # 启动问答系统"
    echo "  ./start.sh crawler    # 启动网络爬虫应用"
    echo ""
    echo -e "${YELLOW}首次使用:${NC}"
    echo "  1. 复制环境配置: cp .env.example .env"
    echo "  2. 编辑配置文件: nano .env"
    echo "  3. 安装依赖: pip install -r requirements.txt"
    echo "  4. 启动应用: ./start.sh <app_name>"
}

# 检查环境
check_environment() {
    echo -e "${BLUE}🔍 检查运行环境...${NC}"

    # 检查 Python 版本
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 未安装${NC}"
        exit 1
    fi

    # 检查 Python 版本是否满足要求 (>=3.11.0)
    python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    required_version="3.11"
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
        echo -e "${RED}❌ Python 版本不满足要求${NC}"
        echo -e "${YELLOW}当前版本: $python_version, 要求版本: >= $required_version${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Python 版本: $python_version${NC}"

    # 检查 pip
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}❌ pip 未安装${NC}"
        exit 1
    fi

    # 检查 .env 文件
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}⚠️  .env 文件不存在${NC}"
        echo -e "${BLUE}💡 提示: 请先运行 'cp .env.example .env' 并配置环境变量${NC}"
        return 1
    fi

    # 检查是否有虚拟环境
    if [ ! -d ".venv" ]; then
        echo -e "${YELLOW}🔧 未发现虚拟环境，正在创建...${NC}"
        if ! python3 -m venv .venv; then
            echo -e "${RED}❌ 虚拟环境创建失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ 虚拟环境创建成功${NC}"
    fi

    # 激活虚拟环境
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo -e "${GREEN}✅ 虚拟环境已激活${NC}"
    else
        echo -e "${RED}❌ 虚拟环境激活失败${NC}"
        exit 1
    fi

    # 检查关键依赖
    python3 -c "import streamlit" 2>/dev/null || {
        echo -e "${YELLOW}⚠️  缺少依赖包，正在安装...${NC}"
        if ! pip install -r requirements.txt; then
            echo -e "${RED}❌ 依赖安装失败${NC}"
            echo -e "${YELLOW}💡 请尝试以下解决方案:${NC}"
            echo "  1. 删除虚拟环境重新创建: rm -rf .venv"
            echo "  2. 升级 pip: python3 -m pip install --upgrade pip"
            echo "  3. 检查 Python 版本: python3 --version"
            exit 1
        fi

        # 再次检查安装是否成功
        python3 -c "import streamlit" 2>/dev/null || {
            echo -e "${RED}❌ streamlit 安装失败，请检查 Python 环境${NC}"
            exit 1
        }
    }

    echo -e "${GREEN}✅ 环境检查通过${NC}"
    return 0
}

# 启动应用
start_app() {
    local app_name=$1
    local app_dir=""
    local app_title=""

    case $app_name in
        "summary")
            app_dir="clickzetta-summary"
            app_title="文档智能摘要系统"
            ;;
        "qa")
            app_dir="clickzetta-qa"
            app_title="智能问答系统"
            ;;
        "search")
            app_dir="clickzetta-hybrid-search"
            app_title="混合搜索系统"
            ;;
        "sql")
            app_dir="clickzetta-sql-chat"
            app_title="SQL 智能问答系统"
            ;;
        "crawler")
            app_dir="clickzetta-web-crawler"
            app_title="网络爬虫存储演示"
            ;;
        *)
            echo -e "${RED}❌ 未知应用: $app_name${NC}"
            show_help
            exit 1
            ;;
    esac

    if [ ! -d "$app_dir" ]; then
        echo -e "${RED}❌ 应用目录不存在: $app_dir${NC}"
        exit 1
    fi

    echo -e "${BLUE}🚀 启动 $app_title...${NC}"
    echo -e "${YELLOW}📱 应用将在浏览器中打开: http://localhost:8501${NC}"
    echo -e "${YELLOW}⏹️  按 Ctrl+C 停止应用${NC}"
    echo ""

    cd "$app_dir"

    # 确保虚拟环境已激活
    if [ -z "$VIRTUAL_ENV" ]; then
        echo -e "${YELLOW}🔧 激活虚拟环境...${NC}"
        source ../.venv/bin/activate
    fi

    # 确保找到 streamlit 命令
    if ! command -v streamlit &> /dev/null; then
        echo -e "${YELLOW}💡 使用 python3 -m streamlit 启动${NC}"
        python3 -m streamlit run streamlit_app.py
    else
        streamlit run streamlit_app.py
    fi
}

# 主函数
main() {
    show_logo

    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    local app_name=$1

    # 检查环境
    if ! check_environment; then
        exit 1
    fi

    # 启动应用
    start_app "$app_name"
}

# 运行主函数
main "$@"