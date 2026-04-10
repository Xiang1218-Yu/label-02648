.PHONY: help install install-dev test clean lint format train

.DEFAULT_GOAL := help

VENV_NAME := venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip

help: ## 显示帮助信息
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## 创建虚拟环境
	python3 -m venv $(VENV_NAME)
	@echo "虚拟环境已创建，运行 'source $(VENV_NAME)/bin/activate' 激活"

install: venv ## 安装核心依赖
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: install ## 安装开发依赖
	$(PIP) install -e ".[dev,notebooks]"

test: ## 运行测试
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

test-html: ## 运行测试并生成HTML覆盖率报告
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=html
	@echo "HTML覆盖率报告: htmlcov/index.html"

lint: ## 代码检查
	$(PYTHON) -m flake8 src/ tests/
	$(PYTHON) -m mypy src/

format: ## 代码格式化
	$(PYTHON) -m black src/ tests/ scripts/

train: ## 开始训练
	$(PYTHON) main.py

clean: ## 清理项目
	rm -rf $(VENV_NAME)
	rm -rf build/ dist/ *.egg-info
	rm -rf .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
