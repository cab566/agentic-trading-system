#!/usr/bin/env python3
"""
System Verification Script

This script verifies:
1. System reality - what's actually running and operational
2. Testing capabilities - what tests exist and their status
3. Data state - whether data is real or test data
4. Safe deployment - environment isolation and safety measures
5. Automated deployment - CI/CD and automation capabilities
"""

import os
import sys
import sqlite3
import subprocess
import json
from datetime import datetime
from pathlib import Path
import psutil

class SystemVerifier:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_reality': {},
            'testing_status': {},
            'data_state': {},
            'deployment_safety': {},
            'automation_capabilities': {}
        }
    
    def verify_system_reality(self):
        """Check what's actually running and operational"""
        print("\n=== SYSTEM REALITY VERIFICATION ===")
        
        # Check running processes
        running_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                if any('system_orchestrator' in str(cmd) or 'trading_system' in str(cmd) 
                      for cmd in proc.info['cmdline'] or []):
                    running_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': ' '.join(proc.info['cmdline'] or []),
                        'running_since': datetime.fromtimestamp(proc.info['create_time']).isoformat()
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        self.results['system_reality']['running_processes'] = running_processes
        print(f"Found {len(running_processes)} trading system processes running")
        
        # Check system files
        critical_files = [
            'system_orchestrator.py',
            'main.py',
            'deploy.py',
            'requirements.txt',
            '.env'
        ]
        
        file_status = {}
        for file in critical_files:
            file_path = self.project_root / file
            file_status[file] = {
                'exists': file_path.exists(),
                'size': file_path.stat().st_size if file_path.exists() else 0,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
            }
        
        self.results['system_reality']['critical_files'] = file_status
        
        # Check environment configuration
        env_files = ['.env', '.env.live', '.env.example']
        env_status = {}
        for env_file in env_files:
            env_path = self.project_root / env_file
            if env_path.exists():
                with open(env_path, 'r') as f:
                    content = f.read()
                    env_status[env_file] = {
                        'exists': True,
                        'trading_mode': 'live' if 'TRADING_MODE=live' in content else 'paper' if 'TRADING_MODE=paper' in content else 'unknown',
                        'has_real_keys': 'your-actual' not in content and 'demo-key' not in content,
                        'size': len(content)
                    }
            else:
                env_status[env_file] = {'exists': False}
        
        self.results['system_reality']['environment_config'] = env_status
        
        return len(running_processes) > 0
    
    def verify_testing_capabilities(self):
        """Check testing framework and test coverage"""
        print("\n=== TESTING CAPABILITIES VERIFICATION ===")
        
        # Check test files
        test_files = list(self.project_root.glob('test*.py')) + list((self.project_root / 'tests').glob('*.py') if (self.project_root / 'tests').exists() else [])
        
        test_status = {}
        for test_file in test_files:
            with open(test_file, 'r') as f:
                content = f.read()
                test_status[test_file.name] = {
                    'size': len(content),
                    'test_functions': content.count('def test_'),
                    'async_tests': content.count('async def test_'),
                    'mock_usage': content.count('mock') + content.count('Mock'),
                    'integration_tests': 'integration' in content.lower()
                }
        
        self.results['testing_status']['test_files'] = test_status
        
        # Check pytest configuration
        pytest_config = self.project_root / 'pytest.ini'
        if pytest_config.exists():
            with open(pytest_config, 'r') as f:
                config_content = f.read()
                self.results['testing_status']['pytest_config'] = {
                    'exists': True,
                    'has_markers': 'markers' in config_content,
                    'test_paths': 'testpaths' in config_content,
                    'coverage': 'coverage' in config_content
                }
        
        # Try running a simple test
        try:
            result = subprocess.run(['python', '-m', 'pytest', '--collect-only', '-q'], 
                                  capture_output=True, text=True, cwd=self.project_root, timeout=30)
            self.results['testing_status']['pytest_collection'] = {
                'success': result.returncode == 0,
                'output': result.stdout[:500],
                'errors': result.stderr[:500]
            }
        except Exception as e:
            self.results['testing_status']['pytest_collection'] = {
                'success': False,
                'error': str(e)
            }
        
        print(f"Found {len(test_files)} test files")
        return len(test_files) > 0
    
    def verify_data_state(self):
        """Check if data is real or test data"""
        print("\n=== DATA STATE VERIFICATION ===")
        
        # Check database
        db_path = self.project_root / 'data' / 'trading_data.db'
        if db_path.exists():
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get table info
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                db_info = {'tables': tables, 'table_data': {}}
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                        sample_data = cursor.fetchall()
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = [col[1] for col in cursor.fetchall()]
                        
                        db_info['table_data'][table] = {
                            'count': count,
                            'columns': columns,
                            'sample_data': sample_data[:3]  # First 3 rows
                        }
                    else:
                        db_info['table_data'][table] = {'count': 0}
                
                conn.close()
                self.results['data_state']['database'] = db_info
                
            except Exception as e:
                self.results['data_state']['database'] = {'error': str(e)}
        else:
            self.results['data_state']['database'] = {'exists': False}
        
        # Check log files
        log_dir = self.project_root / 'logs'
        if log_dir.exists():
            log_files = list(log_dir.glob('*.log'))
            log_info = {}
            for log_file in log_files[-5:]:  # Last 5 log files
                stat = log_file.stat()
                log_info[log_file.name] = {
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'has_content': stat.st_size > 0
                }
            self.results['data_state']['logs'] = log_info
        
        # Check backup files
        backup_dir = self.project_root / 'backups'
        if backup_dir.exists():
            backups = list(backup_dir.glob('*'))
            self.results['data_state']['backups'] = {
                'count': len(backups),
                'latest': max(backups, key=lambda x: x.stat().st_mtime).name if backups else None
            }
        
        return True
    
    def verify_deployment_safety(self):
        """Check environment isolation and safety measures"""
        print("\n=== DEPLOYMENT SAFETY VERIFICATION ===")
        
        # Check current environment
        current_env = os.environ.get('TRADING_MODE', 'unknown')
        
        # Check for safety measures
        safety_checks = {
            'paper_trading_available': (self.project_root / '.env').exists() and 'paper' in open(self.project_root / '.env').read(),
            'live_env_separate': (self.project_root / '.env.live').exists(),
            'example_env_exists': (self.project_root / '.env.example').exists(),
            'deployment_script_exists': (self.project_root / 'deploy.py').exists(),
            'current_mode': current_env
        }
        
        # Check Docker support
        try:
            docker_result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            safety_checks['docker_available'] = docker_result.returncode == 0
        except FileNotFoundError:
            safety_checks['docker_available'] = False
        
        # Check if system has risk management
        risk_files = list(self.project_root.glob('*risk*.py'))
        safety_checks['risk_management_files'] = len(risk_files)
        
        self.results['deployment_safety'] = safety_checks
        
        return safety_checks['paper_trading_available']
    
    def verify_automation_capabilities(self):
        """Check CI/CD and automation capabilities"""
        print("\n=== AUTOMATION CAPABILITIES VERIFICATION ===")
        
        automation_features = {
            'deployment_script': (self.project_root / 'deploy.py').exists(),
            'docker_compose': (self.project_root / 'docker-compose.yml').exists(),
            'dockerfile': (self.project_root / 'Dockerfile').exists(),
            'requirements_file': (self.project_root / 'requirements.txt').exists(),
            'integration_tests': (self.project_root / 'system_integration_tests.py').exists(),
            'github_actions': (self.project_root / '.github' / 'workflows').exists(),
            'makefile': (self.project_root / 'Makefile').exists()
        }
        
        # Check if deployment script is functional
        if automation_features['deployment_script']:
            try:
                result = subprocess.run(['python', 'deploy.py', '--help'], 
                                      capture_output=True, text=True, cwd=self.project_root, timeout=10)
                automation_features['deployment_script_functional'] = result.returncode == 0
            except Exception:
                automation_features['deployment_script_functional'] = False
        
        self.results['automation_capabilities'] = automation_features
        
        return automation_features['deployment_script']
    
    def generate_report(self):
        """Generate comprehensive verification report"""
        print("\n" + "="*60)
        print("SYSTEM VERIFICATION REPORT")
        print("="*60)
        
        # System Reality
        reality = self.results['system_reality']
        print(f"\n🔍 SYSTEM REALITY:")
        print(f"   Running Processes: {len(reality.get('running_processes', []))}")
        print(f"   Critical Files: {sum(1 for f in reality.get('critical_files', {}).values() if f.get('exists'))} / {len(reality.get('critical_files', {}))}")
        
        env_config = reality.get('environment_config', {})
        current_mode = None
        for env_file, config in env_config.items():
            if config.get('exists') and env_file == '.env':
                current_mode = config.get('trading_mode', 'unknown')
        print(f"   Current Trading Mode: {current_mode or 'unknown'}")
        
        # Testing Status
        testing = self.results['testing_status']
        print(f"\n🧪 TESTING STATUS:")
        print(f"   Test Files: {len(testing.get('test_files', {}))}")
        total_tests = sum(f.get('test_functions', 0) for f in testing.get('test_files', {}).values())
        print(f"   Total Test Functions: {total_tests}")
        pytest_works = testing.get('pytest_collection', {}).get('success', False)
        print(f"   Pytest Functional: {'✅' if pytest_works else '❌'}")
        
        # Data State
        data = self.results['data_state']
        print(f"\n💾 DATA STATE:")
        db_info = data.get('database', {})
        if 'tables' in db_info:
            total_records = sum(table.get('count', 0) for table in db_info.get('table_data', {}).values())
            print(f"   Database Records: {total_records}")
            print(f"   Database Tables: {len(db_info.get('tables', []))}")
        else:
            print(f"   Database: {'❌ Not accessible' if 'error' in db_info else '❌ Not found'}")
        
        log_count = len(data.get('logs', {}))
        print(f"   Log Files: {log_count}")
        backup_count = data.get('backups', {}).get('count', 0)
        print(f"   Backup Files: {backup_count}")
        
        # Safety
        safety = self.results['deployment_safety']
        print(f"\n🛡️  DEPLOYMENT SAFETY:")
        print(f"   Paper Trading Available: {'✅' if safety.get('paper_trading_available') else '❌'}")
        print(f"   Environment Isolation: {'✅' if safety.get('live_env_separate') else '❌'}")
        print(f"   Docker Support: {'✅' if safety.get('docker_available') else '❌'}")
        print(f"   Risk Management Files: {safety.get('risk_management_files', 0)}")
        
        # Automation
        automation = self.results['automation_capabilities']
        print(f"\n🤖 AUTOMATION CAPABILITIES:")
        print(f"   Deployment Script: {'✅' if automation.get('deployment_script') else '❌'}")
        print(f"   Docker Support: {'✅' if automation.get('docker_compose') and automation.get('dockerfile') else '❌'}")
        print(f"   Integration Tests: {'✅' if automation.get('integration_tests') else '❌'}")
        print(f"   CI/CD Pipeline: {'✅' if automation.get('github_actions') else '❌ (No GitHub Actions found)'}")
        
        # Overall Assessment
        print(f"\n📊 OVERALL ASSESSMENT:")
        
        is_real = len(reality.get('running_processes', [])) > 0
        is_tested = total_tests > 0 and pytest_works
        is_safe = safety.get('paper_trading_available', False)
        can_deploy = automation.get('deployment_script', False)
        
        print(f"   ✅ System is REAL and OPERATIONAL: {is_real}")
        print(f"   ✅ System is TESTED: {is_tested}")
        print(f"   ✅ Safe testing environment available: {is_safe}")
        print(f"   ✅ Automated deployment capable: {can_deploy}")
        
        if current_mode == 'paper':
            print(f"   🟡 Currently in PAPER TRADING mode (safe for testing)")
        elif current_mode == 'live':
            print(f"   🔴 Currently in LIVE TRADING mode (real money at risk)")
        else:
            print(f"   🟡 Trading mode unclear")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if not is_tested:
            print(f"   - Fix test suite issues before deployment")
        if not is_safe:
            print(f"   - Ensure paper trading environment is properly configured")
        if current_mode != 'paper':
            print(f"   - Switch to paper trading mode for safe testing")
        if not automation.get('github_actions'):
            print(f"   - Consider adding CI/CD pipeline for automated testing")
        
        return {
            'is_real': is_real,
            'is_tested': is_tested,
            'is_safe': is_safe,
            'can_deploy': can_deploy,
            'current_mode': current_mode
        }
    
    def run_verification(self):
        """Run complete system verification"""
        print("Starting comprehensive system verification...")
        
        self.verify_system_reality()
        self.verify_testing_capabilities()
        self.verify_data_state()
        self.verify_deployment_safety()
        self.verify_automation_capabilities()
        
        assessment = self.generate_report()
        
        # Save detailed results
        results_file = self.project_root / 'system_verification_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return assessment

if __name__ == "__main__":
    verifier = SystemVerifier()
    assessment = verifier.run_verification()
    
    # Exit with appropriate code
    all_good = all([
        assessment['is_real'],
        assessment['is_tested'],
        assessment['is_safe'],
        assessment['can_deploy']
    ])
    
    sys.exit(0 if all_good else 1)