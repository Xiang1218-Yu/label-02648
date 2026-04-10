#!/bin/bash
# =============================================================================
# 清理脚本
# 清理项目生成的临时文件和缓存
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 默认参数
ALL=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --all|-a)
            ALL=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --all, -a       清理所有（包括虚拟环境）"
            echo "  --help, -h      显示帮助"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

print_info "开始清理项目..."

# 清理 Python 缓存
print_info "清理 Python 缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.py[cod]" -delete 2>/dev/null || true
find . -type f -name "*$py.class" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".coverage" -delete 2>/dev/null || true
find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
print_success "Python 缓存清理完成"

# 清理日志文件
print_info "清理日志文件..."
if [ -d "logs" ]; then
    find logs -type f -name "*.log" -delete 2>/dev/null || true
    print_success "日志文件清理完成"
fi

# 清理结果文件（可选）
print_info "清理结果文件..."
if [ -d "results" ]; then
    find results -type f ! -name ".gitkeep" -delete 2>/dev/null || true
    print_success "结果文件清理完成"
fi

# 清理模型文件（可选）
print_info "清理模型文件..."
if [ -d "models" ]; then
    find models -type f ! -name ".gitkeep" -delete 2>/dev/null || true
    print_success "模型文件清理完成"
fi

# 清理数据文件（保留原始数据）
print_info "清理生成的数据文件..."
if [ -f "data/simulated_data.csv" ]; then
    rm -f "data/simulated_data.csv"
    print_success "模拟数据已删除"
fi

# 清理虚拟环境
if [ "$ALL" = true ]; then
    print_warning "清理虚拟环境..."
    if [ -d "venv" ]; then
        rm -rf venv
        print_success "虚拟环境已删除"
    fi
    if [ -d "env" ]; then
        rm -rf env
        print_success "环境目录已删除"
    fi
fi

print_success "项目清理完成!"

if [ "$ALL" = true ]; then
    print_warning "虚拟环境已删除，需要重新运行 scripts/setup_env.sh"
fi
