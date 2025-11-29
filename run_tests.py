#!/usr/bin/env python3
"""Test runner script for the trading system."""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Test runner for the trading system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests"
        self.reports_dir = project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_tests(
        self,
        test_type: str = "all",
        markers: Optional[List[str]] = None,
        coverage: bool = True,
        verbose: bool = True,
        parallel: bool = False,
        output_format: str = "terminal"
    ) -> int:
        """Run tests with specified configuration.
        
        Args:
            test_type: Type of tests to run (unit, integration, performance, all)
            markers: Additional pytest markers to include/exclude
            coverage: Whether to generate coverage report
            verbose: Verbose output
            parallel: Run tests in parallel
            output_format: Output format (terminal, html, json)
        
        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        cmd = ["python", "-m", "pytest"]
        
        # Add test directory
        cmd.append(str(self.test_dir))
        
        # Configure test selection based on type
        if test_type == "unit":
            cmd.extend(["-m", "unit and not integration and not performance"])
        elif test_type == "integration":
            cmd.extend(["-m", "integration and not performance"])
        elif test_type == "performance":
            cmd.extend(["-m", "performance"])
        elif test_type == "fast":
            cmd.extend(["-m", "not slow and not performance"])
        elif test_type == "smoke":
            cmd.extend(["-m", "smoke"])
        # "all" runs everything (no marker filter)
        
        # Add custom markers
        if markers:
            for marker in markers:
                cmd.extend(["-m", marker])
        
        # Configure output verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Configure parallel execution
        if parallel:
            try:
                import pytest_xdist
                cmd.extend(["-n", "auto"])
            except ImportError:
                print("Warning: pytest-xdist not installed, running sequentially")
        
        # Configure coverage
        if coverage:
            cmd.extend([
                "--cov=trading_system_v2",
                "--cov-report=term-missing",
                f"--cov-report=html:{self.reports_dir}/coverage_html",
                f"--cov-report=xml:{self.reports_dir}/coverage.xml",
                "--cov-fail-under=80"
            ])
        
        # Configure output format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format in ["html", "all"]:
            cmd.extend([
                f"--html={self.reports_dir}/report_{timestamp}.html",
                "--self-contained-html"
            ])
        
        if output_format in ["json", "all"]:
            cmd.extend([
                f"--json-report",
                f"--json-report-file={self.reports_dir}/report_{timestamp}.json"
            ])
        
        if output_format in ["junit", "all"]:
            cmd.extend([
                f"--junit-xml={self.reports_dir}/junit_{timestamp}.xml"
            ])
        
        # Add additional pytest options
        cmd.extend([
            "--tb=short",  # Shorter traceback format
            "--strict-markers",  # Strict marker validation
            "--durations=10",  # Show 10 slowest tests
            "--maxfail=5",  # Stop after 5 failures
        ])
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Test reports will be saved to: {self.reports_dir}")
        
        # Run tests
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
        except KeyboardInterrupt:
            print("\nTests interrupted by user")
            return 130
        except Exception as e:
            print(f"Error running tests: {e}")
            return 1
    
    def run_linting(self) -> int:
        """Run code linting and formatting checks."""
        print("Running code quality checks...")
        
        commands = [
            # Black formatting check
            ["python", "-m", "black", "--check", "--diff", "."],
            # isort import sorting check
            ["python", "-m", "isort", "--check-only", "--diff", "."],
            # flake8 linting
            ["python", "-m", "flake8", "."],
            # mypy type checking
            ["python", "-m", "mypy", "trading_system_v2"],
            # pylint code analysis
            ["python", "-m", "pylint", "trading_system_v2"],
        ]
        
        overall_result = 0
        
        for cmd in commands:
            tool_name = cmd[2]  # Extract tool name
            print(f"\nRunning {tool_name}...")
            
            try:
                result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {tool_name}: PASSED")
                else:
                    print(f"❌ {tool_name}: FAILED")
                    print(result.stdout)
                    print(result.stderr)
                    overall_result = 1
                    
            except FileNotFoundError:
                print(f"⚠️  {tool_name}: Not installed, skipping")
            except Exception as e:
                print(f"❌ {tool_name}: Error - {e}")
                overall_result = 1
        
        return overall_result
    
    def run_security_checks(self) -> int:
        """Run security vulnerability checks."""
        print("Running security checks...")
        
        commands = [
            # bandit security linting
            ["python", "-m", "bandit", "-r", "trading_system_v2", "-f", "json", "-o", str(self.reports_dir / "bandit_report.json")],
            # safety dependency vulnerability check
            ["python", "-m", "safety", "check", "--json", "--output", str(self.reports_dir / "safety_report.json")],
        ]
        
        overall_result = 0
        
        for cmd in commands:
            tool_name = cmd[2]  # Extract tool name
            print(f"\nRunning {tool_name}...")
            
            try:
                result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {tool_name}: No issues found")
                else:
                    print(f"⚠️  {tool_name}: Issues found, check report")
                    # Don't fail on security warnings, just report them
                    
            except FileNotFoundError:
                print(f"⚠️  {tool_name}: Not installed, skipping")
            except Exception as e:
                print(f"❌ {tool_name}: Error - {e}")
        
        return overall_result
    
    def generate_test_summary(self) -> None:
        """Generate a test summary report."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        # Find latest test reports
        json_reports = list(self.reports_dir.glob("report_*.json"))
        if json_reports:
            latest_report = max(json_reports, key=lambda p: p.stat().st_mtime)
            
            try:
                with open(latest_report) as f:
                    report_data = json.load(f)
                
                summary = report_data.get('summary', {})
                print(f"Total Tests: {summary.get('total', 'N/A')}")
                print(f"Passed: {summary.get('passed', 'N/A')}")
                print(f"Failed: {summary.get('failed', 'N/A')}")
                print(f"Skipped: {summary.get('skipped', 'N/A')}")
                print(f"Duration: {summary.get('duration', 'N/A')}s")
                
            except Exception as e:
                print(f"Could not parse test report: {e}")
        
        # Coverage summary
        coverage_file = self.reports_dir / "coverage.xml"
        if coverage_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                
                coverage_elem = root.find('.//coverage')
                if coverage_elem is not None:
                    line_rate = float(coverage_elem.get('line-rate', 0)) * 100
                    branch_rate = float(coverage_elem.get('branch-rate', 0)) * 100
                    print(f"\nCoverage:")
                    print(f"Line Coverage: {line_rate:.1f}%")
                    print(f"Branch Coverage: {branch_rate:.1f}%")
                    
            except Exception as e:
                print(f"Could not parse coverage report: {e}")
        
        print(f"\nReports saved to: {self.reports_dir}")
        print("="*60)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Trading System Test Runner")
    
    parser.add_argument(
        "--type", "-t",
        choices=["unit", "integration", "performance", "fast", "smoke", "all"],
        default="all",
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--markers", "-m",
        nargs="*",
        help="Additional pytest markers to include"
    )
    
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage reporting"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet output"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["terminal", "html", "json", "junit", "all"],
        default="terminal",
        help="Output format"
    )
    
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run code linting and formatting checks"
    )
    
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run security vulnerability checks"
    )
    
    parser.add_argument(
        "--all-checks",
        action="store_true",
        help="Run tests, linting, and security checks"
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(project_root)
    
    overall_result = 0
    
    # Run requested checks
    if args.all_checks or not any([args.lint, args.security]):
        # Run tests (default behavior)
        print("Running tests...")
        result = runner.run_tests(
            test_type=args.type,
            markers=args.markers,
            coverage=not args.no_coverage,
            verbose=not args.quiet,
            parallel=args.parallel,
            output_format=args.format
        )
        overall_result = max(overall_result, result)
    
    if args.lint or args.all_checks:
        result = runner.run_linting()
        overall_result = max(overall_result, result)
    
    if args.security or args.all_checks:
        result = runner.run_security_checks()
        overall_result = max(overall_result, result)
    
    # Generate summary
    runner.generate_test_summary()
    
    # Exit with appropriate code
    if overall_result == 0:
        print("\n🎉 All checks passed!")
    else:
        print("\n❌ Some checks failed. See reports for details.")
    
    sys.exit(overall_result)


if __name__ == "__main__":
    main()