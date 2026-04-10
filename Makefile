# 外卖订单预测与调度优化系统 Makefile

.PHONY: all install install-dev clean clean-pyc clean-build lint format test train evaluate run help

# 默认目标
all: help

# 安装依赖
install:
	pip install --upgrade pip
	pip install -e .

# 安装开发依赖
install-dev: install
	pip install -e ".[dev]"
	pre-commit install

# 清理
clean: clean-pyc clean-build clean-logs

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info
	rm -fr *.egg

clean-logs:
	rm -f logs/*.log
	rm -rf results/*

clean-models:
	rm -rf models/*

# 代码检查
lint:
	flake8 src/ data/ scripts/ main.py config.py
	mypy src/ --ignore-missing-imports

# 代码格式化
format:
	isort src/ data/ scripts/ main.py config.py
	black src/ data/ scripts/ main.py config.py

# 运行测试
test:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing

# 快速测试
test-fast:
	python -m pytest tests/ -v -x --ignore=tests/test_full_pipeline.py

# 生成数据
data:
	python scripts/generate_data.py

# 训练模型
train:
	python scripts/train_model.py --all

# 评估模型
evaluate:
	python scripts/evaluate.py

# 运行主程序
run:
	python main.py

# 运行完整流水线
pipeline:
	python scripts/run_pipeline.py --full

# Docker 相关
docker-build:
	docker-compose build

docker-run:
	docker-compose up

docker-stop:
	docker-compose down

# 帮助信息
help:
	@echo '外卖订单预测与调度优化系统 - 可用命令:'
	@echo ''
	@echo '安装命令:'
	@echo '  install      安装项目依赖'
	@echo '  install-dev  安装项目依赖及开发依赖 (代码检查/测试)'
	@echo ''
	@echo '代码质量:'
	@echo '  lint         运行代码检查 (flake8, mypy)'
	@echo '  format       运行代码格式化 (black, isort)'
	@echo ''
	@echo '项目执行:'
	@echo '  data         生成模拟数据'
	@echo '  train        训练所有模型'
	@echo '  evaluate     评估模型性能'
	@echo '  run          运行完整主程序'
	@echo '  pipeline     运行完整流水线脚本'
	@echo '  test         运行所有测试'
	@echo '  test-fast    快速测试 (跳过长耗时)'
	@echo ''
	@echo '清理命令:'
	@echo '  clean        清理所有临时文件 (pyc, build, logs)'
	@echo '  clean-models 清理已保存模型'
	@echo ''
	@echo 'Docker命令:'
	@echo '  docker-build 构建Docker镜像'
	@echo '  docker-run   运行Docker容器'
	@echo '  docker-stop  停止Docker容器'
