#!/bin/bash
# =============================================================================
# 环境初始化脚本
# 用于设置 Python 虚拟环境并安装依赖
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
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

print_info "项目目录: $PROJECT_ROOT"

# 检查 Python 版本
print_info "检查 Python 版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "需要 Python $REQUIRED_VERSION 或更高版本，当前版本: $PYTHON_VERSION"
    exit 1
fi
print_success "Python 版本检查通过: $PYTHON_VERSION"

# 创建虚拟环境
VENV_DIR="$PROJECT_ROOT/venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "虚拟环境已存在: $VENV_DIR"
    read -p "是否重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "删除旧虚拟环境..."
        rm -rf "$VENV_DIR"
        print_info "创建新虚拟环境..."
        python3 -m venv "$VENV_DIR"
    fi
else
    print_info "创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
fi

print_success "虚拟环境创建完成"

# 激活虚拟环境
print_info "激活虚拟环境..."
source "$VENV_DIR/bin/activate"

# 升级 pip
print_info "升级 pip..."
pip install --upgrade pip

# 安装依赖
print_info "安装项目依赖..."
pip install -r requirements.txt

# 可选：安装开发依赖
read -p "是否安装开发依赖? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "安装开发依赖..."
    pip install -e ".[dev]"
fi

# 安装项目为可编辑模式
print_info "安装项目为可编辑模式..."
pip install -e .

print_success "环境设置完成!"
print_info "激活虚拟环境命令: source venv/bin/activate"
print_info "运行项目命令: python main.py"
