#!/usr/bin/env python
"""
测试运行脚本

用法:
    python run_tests.py              # 运行所有测试
    python run_tests.py --unit       # 只运行单元测试
    python run_tests.py --integration # 只运行集成测试
    python run_tests.py --security   # 只运行安全测试
    python run_tests.py --coverage   # 运行测试并生成覆盖率报告
    python run_tests.py --report     # 生成HTML报告
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_tests(args):
    """运行测试"""
    cmd = ["python", "-m", "pytest"]

    if args.unit:
        cmd.extend([
            "tests/test_indicators.py",
            "tests/test_predictor.py",
            "tests/test_risk_control.py",
        ])
    elif args.integration:
        cmd.extend([
            "tests/test_integration.py",
            "tests/test_auto_trader_logic.py",
        ])
    elif args.security:
        cmd.extend(["tests/test_security.py"])
    elif args.memory:
        cmd.extend(["tests/test_memory_and_resources.py"])
    elif args.edge:
        cmd.extend(["tests/test_edge_cases.py"])
    else:
        cmd.append("tests/")

    if args.verbose:
        cmd.append("-v")

    if args.coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_report",
            "--cov-exclude=tests/*",
        ])

    if args.report:
        cmd.extend([
            "--html=test_report.html",
            "--self-contained-html",
        ])

    if args.fail_fast:
        cmd.append("-x")

    if args.parallel:
        cmd.extend(["-n", "auto"])

    print(f"运行命令: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def check_dependencies():
    """检查依赖"""
    required = ["pytest", "numpy"]
    optional = ["pytest-asyncio", "pytest-cov", "pytest-html", "pytest-xdist"]

    missing_required = []
    missing_optional = []

    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing_required.append(pkg)

    for pkg in optional:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing_optional.append(pkg)

    if missing_required:
        print(f"缺少必需依赖: {', '.join(missing_required)}")
        print(f"请运行: pip install {' '.join(missing_required)}")
        return False

    if missing_optional:
        print(f"可选依赖未安装: {', '.join(missing_optional)}")
        print("某些功能可能不可用")

    return True


def main():
    parser = argparse.ArgumentParser(description="运行测试套件")

    parser.add_argument("--unit", action="store_true", help="只运行单元测试")
    parser.add_argument("--integration", action="store_true", help="只运行集成测试")
    parser.add_argument("--security", action="store_true", help="只运行安全测试")
    parser.add_argument("--memory", action="store_true", help="只运行内存测试")
    parser.add_argument("--edge", action="store_true", help="只运行边界条件测试")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--coverage", action="store_true", help="生成覆盖率报告")
    parser.add_argument("--report", action="store_true", help="生成HTML报告")
    parser.add_argument("-x", "--fail-fast", action="store_true", help="遇到失败立即停止")
    parser.add_argument("-p", "--parallel", action="store_true", help="并行运行测试")

    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
