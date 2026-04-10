#!/bin/bash
# =============================================================================
# 完整流程运行脚本
# 运行完整的数据处理、训练、评估流程
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
SKIP_EDA=false
USE_OPTUNA=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-eda)
            SKIP_EDA=true
            shift
            ;;
        --use-optuna)
            USE_OPTUNA=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --skip-eda      跳过探索性数据分析"
            echo "  --use-optuna    使用 Optuna 进行超参数优化"
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

# 记录开始时间
START_TIME=$(date +%s)

print_header "开始执行完整流程"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 修改配置文件（临时）
CONFIG_FILE="$PROJECT_ROOT/config.py"
if [ "$SKIP_EDA" = true ]; then
    print_info "跳过 EDA 分析"
    sed -i.bak 's/ENABLE_EDA = True/ENABLE_EDA = False/' "$CONFIG_FILE"
fi

if [ "$USE_OPTUNA" = true ]; then
    print_info "启用 Optuna 超参数优化"
    sed -i.bak 's/USE_OPTUNA = False/USE_OPTUNA = True/' "$CONFIG_FILE"
fi

# 运行主程序
print_info "运行主程序..."
python main.py

EXIT_CODE=$?

# 恢复配置文件
if [ -f "$CONFIG_FILE.bak" ]; then
    mv "$CONFIG_FILE.bak" "$CONFIG_FILE"
fi

# 计算运行时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

if [ $EXIT_CODE -eq 0 ]; then
    print_header "流程执行成功"
    print_success "总运行时间: ${MINUTES}分${SECONDS}秒"
    print_info "结果目录: $PROJECT_ROOT/results/"
    print_info "模型目录: $PROJECT_ROOT/models/"
    print_info "日志文件: $PROJECT_ROOT/logs/project.log"
else
    print_error "流程执行失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi
