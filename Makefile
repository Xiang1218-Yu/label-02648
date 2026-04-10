# =============================================================================
# 外卖订单量预测与调度优化系统 - Makefile
# 提供常用命令的快捷方式
# =============================================================================

.PHONY: help install install-dev test test-cov lint lint-fix clean run docker-build docker-run

# 默认目标
.DEFAULT_GOAL := help

# 颜色定义
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# Python 解释器
PYTHON := python3
PIP := pip

# =============================================================================
# 帮助
# =============================================================================
help:
	@echo "$(BLUE)外卖订单量预测与调度优化系统 - 可用命令$(NC)"
	@echo ""
	@echo "$(GREEN)环境设置$(NC)"
	@echo "  make install          安装项目依赖"
	@echo "  make install-dev      安装开发依赖"
	@echo "  make setup            运行完整的环境设置脚本"
	@echo ""
	@echo "$(GREEN)运行$(NC)"
	@echo "  make run              运行主程序"
	@echo "  make run-pipeline     运行完整流程"
	@echo "  make run-optuna       运行带超参数优化的流程"
	@echo ""
	@echo "$(GREEN)测试$(NC)"
	@echo "  make test             运行测试"
	@echo "  make test-cov         运行测试并生成覆盖率报告"
	@echo "  make test-verbose     运行测试（详细输出）"
	@echo ""
	@echo "$(GREEN)代码质量$(NC)"
	@echo "  make lint             运行代码检查"
	@echo "  make lint-fix         自动修复代码格式"
	@echo "  make format           格式化代码"
	@echo ""
	@echo "$(GREEN)清理$(NC)"
	@echo "  make clean            清理临时文件"
	@echo "  make clean-all        清理所有（包括虚拟环境）"
	@echo ""
	@echo "$(GREEN)Docker$(NC)"
	@echo "  make docker-build     构建 Docker 镜像"
	@echo "  make docker-run       运行 Docker 容器"
	@echo "  make docker-compose   使用 docker-compose 启动"
	@echo ""
	@echo "$(GREEN)其他$(NC)"
	@echo "  make notebook         启动 Jupyter Notebook"
	@echo "  make update-reqs      更新 requirements.txt"

# =============================================================================
# 环境设置
# =============================================================================
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

setup:
	@chmod +x scripts/setup_env.sh
	@./scripts/setup_env.sh

# =============================================================================
# 运行
# =============================================================================
run:
	$(PYTHON) main.py

run-pipeline:
	@chmod +x scripts/run_pipeline.sh
	@./scripts/run_pipeline.sh

run-optuna:
	@chmod +x scripts/run_pipeline.sh
	@./scripts/run_pipeline.sh --use-optuna

# =============================================================================
# 测试
# =============================================================================
test:
	@chmod +x scripts/run_tests.sh
	@./scripts/run_tests.sh

test-cov:
	@chmod +x scripts/run_tests.sh
	@./scripts/run_tests.sh --coverage

test-verbose:
	@chmod +x scripts/run_tests.sh
	@./scripts/run_tests.sh --verbose

# =============================================================================
# 代码质量
# =============================================================================
lint:
	@chmod +x scripts/lint.sh
	@./scripts/lint.sh

lint-fix:
	@chmod +x scripts/lint.sh
	@./scripts/lint.sh --fix

format: lint-fix

# =============================================================================
# 清理
# =============================================================================
clean:
	@chmod +x scripts/clean.sh
	@./scripts/clean.sh

clean-all:
	@chmod +x scripts/clean.sh
	@./scripts/clean.sh --all

# =============================================================================
# Docker
# =============================================================================
docker-build:
	docker build -t food-delivery-forecast:latest .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data -v $(PWD)/results:/app/results food-delivery-forecast:latest

docker-compose:
	docker-compose up --build

docker-compose-down:
	docker-compose down

# =============================================================================
# 其他
# =============================================================================
notebook:
	jupyter notebook notebooks/

update-reqs:
	$(PIP) freeze > requirements.txt

# 快捷命令
i: install
id: install-dev
t: test
tc: test-cov
l: lint
lf: lint-fix
r: run
rp: run-pipeline
c: clean
ca: clean-all
