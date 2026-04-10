#!/bin/bash
# =============================================================================
# 代码质量检查脚本
# 运行 black, isort, flake8, mypy 等代码检查工具
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

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 默认参数
FIX=false
CHECK_ONLY=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix|-f)
            FIX=true
            shift
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --fix, -f       自动修复代码格式问题"
            echo "  --check-only    仅检查，不修复"
            echo "  --help, -h      显示帮助"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    print_warning "未检测到虚拟环境，尝试激活..."
    if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        print_success "已激活虚拟环境"
    else
        print_error "未找到虚拟环境，请先运行 scripts/setup_env.sh"
        exit 1
    fi
fi

# 检查工具是否安装
check_tool() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 未安装，请先安装开发依赖: pip install -e '.[dev]'"
        exit 1
    fi
}

check_tool black
check_tool isort
check_tool flake8
check_tool mypy

TARGETS="src data main.py config.py"

# Black 代码格式化
print_header "Running Black"
if [ "$FIX" = true ]; then
    print_info "正在格式化代码..."
    black $TARGETS
    print_success "Black 格式化完成"
else
    print_info "检查代码格式..."
    if black --check $TARGETS; then
        print_success "Black 检查通过"
    else
        print_error "Black 检查失败，运行 '$0 --fix' 修复"
        exit 1
    fi
fi

# isort 导入排序
print_header "Running isort"
if [ "$FIX" = true ]; then
    print_info "正在排序导入..."
    isort $TARGETS
    print_success "isort 排序完成"
else
    print_info "检查导入排序..."
    if isort --check-only $TARGETS; then
        print_success "isort 检查通过"
    else
        print_error "isort 检查失败，运行 '$0 --fix' 修复"
        exit 1
    fi
fi

# Flake8 代码风格检查
print_header "Running Flake8"
print_info "检查代码风格..."
if flake8 $TARGETS --max-line-length=100 --extend-ignore=E203,W503; then
    print_success "Flake8 检查通过"
else
    print_error "Flake8 检查失败"
    exit 1
fi

# MyPy 类型检查
print_header "Running MyPy"
print_info "进行类型检查..."
if mypy $TARGETS --ignore-missing-imports; then
    print_success "MyPy 检查通过"
else
    print_warning "MyPy 发现类型问题（非阻塞）"
fi

print_header "代码质量检查完成"
