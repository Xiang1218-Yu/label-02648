#!/usr/bin/env python
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        prog="food-delivery",
        description="外卖订单预测与调度优化系统 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run the main application")
    run_parser.add_argument(
        "-e",
        "--env",
        choices=["dev", "prod"],
        default="dev",
        help="Environment to run in",
    )

    data_parser = subparsers.add_parser("data", help="Data processing tools")
    data_parser.add_argument("--generate", type=int, help="Generate N days of data")
    data_parser.add_argument(
        "--preprocess", action="store_true", help="Preprocess data"
    )

    train_parser = subparsers.add_parser("train", help="Model training tools")
    train_parser.add_argument(
        "-m", "--model", nargs="+", default=["random_forest"], help="Models to train"
    )
    train_parser.add_argument(
        "--use-optuna", action="store_true", help="Use Optuna for hyperparameter tuning"
    )

    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "-c", "--coverage", action="store_true", help="Show coverage report"
    )
    test_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )

    lint_parser = subparsers.add_parser("lint", help="Run code quality checks")
    lint_parser.add_argument(
        "--fix", action="store_true", help="Auto-fix issues if possible"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        os.environ["PROJECT_ENV"] = args.env
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "config_module", os.path.join(os.path.dirname(__file__), "config.py")
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        from main import main as run_main

        run_main()

    elif args.command == "data":
        from scripts.data_processing import generate_data, preprocess_data

        if args.generate:
            generate_data(args.generate)
        if args.preprocess:
            preprocess_data()

    elif args.command == "train":
        print(f"Training models: {args.model}")
        print(f"Use Optuna: {args.use_optuna}")
        from main import main as run_main

        run_main()

    elif args.command == "test":
        cmd = ["python", "-m", "pytest"]
        if args.coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        if args.verbose:
            cmd.append("-v")
        os.system(" ".join(cmd))

    elif args.command == "lint":
        if args.fix:
            os.system("black .")
            os.system("isort .")
        os.system("flake8 .")
        os.system("mypy src/")


if __name__ == "__main__":
    main()
