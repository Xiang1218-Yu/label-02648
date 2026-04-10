#!/bin/bash
# =============================================================================
# 测试运行脚本
# 运行项目测试并生成覆盖率报告
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
TEST_PATH="tests"
COVERAGE=false
VERBOSE=false
MARKERS=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --markers|-m)
            MARKERS="$2"
            shift 2
            ;;
        --path|-p)
            TEST_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --coverage, -c     生成覆盖率报告"
            echo "  --verbose, -v      详细输出"
            echo "  --markers, -m      指定测试标记 (如: -m 'not slow')"
            echo "  --path, -p         指定测试路径"
            echo "  --help, -h         显示帮助"
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

# 构建 pytest 命令
PYTEST_CMD="pytest $TEST_PATH"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD -m '$MARKERS'"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov=data --cov-report=term-missing --cov-report=html:htmlcov"
    print_info "将生成 HTML 覆盖率报告: htmlcov/index.html"
fi

# 运行测试
print_info "运行测试: $PYTEST_CMD"
eval $PYTEST_CMD

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "所有测试通过!"
else
    print_error "测试失败，退出码: $TEST_EXIT_CODE"
    exit $TEST_EXIT_CODE
fi
