#!/usr/bin/env python3
"""
EchoVault Secret Scanner
Scans for potential secrets in code before committing to GitHub
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure proper encoding for Windows
if sys.platform == 'win32':
    # Set console encoding to UTF-8 on Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Color codes for terminal output
RED = '\033[91m'
YELLOW = '\033[93m'
GREEN = '\033[92m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Patterns to detect various types of secrets
SECRET_PATTERNS = [
    # Database connections
    (r'postgresql://[^:]+:[^@]+@[^/]+/\w+', 'PostgreSQL DSN'),
    (r'postgres://[^:]+:[^@]+@[^/]+/\w+', 'PostgreSQL DSN'),
    (r'mysql://[^:]+:[^@]+@[^/]+/\w+', 'MySQL DSN'),
    
    # JWT tokens
    (r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', 'JWT Token'),
    
    # API Keys and Secrets
    (r'[a-f0-9]{32,}', 'Possible API Key/Secret (32+ hex chars)'),
    (r'[A-Za-z0-9]{40,}', 'Possible API Key (40+ alphanumeric)'),
    (r'sk_[a-zA-Z0-9]{32,}', 'Secret Key'),
    (r'pk_[a-zA-Z0-9]{32,}', 'Public Key (might be paired with secret)'),
    (r'api[_-]?key[_-]?[=:]\s*["\']?[A-Za-z0-9_-]{20,}', 'API Key Assignment'),
    
    # Cloudflare specific
    (r'https://[a-f0-9-]+\.r2\.cloudflarestorage\.com', 'R2 Storage Endpoint'),
    (r'[a-f0-9]{32}\.r2\.cloudflarestorage\.com', 'R2 Account ID in URL'),
    
    # Qdrant specific
    (r'https://[a-f0-9-]+\.[a-z0-9-]+\.aws\.cloud\.qdrant\.io', 'Qdrant Cloud URL'),
    
    # Neon specific
    (r'ep-[a-z]+-[a-z]+-[a-z0-9]+', 'Neon Endpoint ID'),
    (r'npg_[A-Za-z0-9]+', 'Neon Password Token'),
    
    # AWS patterns
    (r'AKIA[0-9A-Z]{16}', 'AWS Access Key ID'),
    (r'[0-9a-zA-Z/+=]{40}', 'Possible AWS Secret Key'),
    
    # Generic patterns
    (r'["\']?password["\']?\s*[:=]\s*["\'][^"\']{8,}["\']', 'Password Assignment'),
    (r'["\']?token["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', 'Token Assignment'),
    (r'["\']?secret["\']?\s*[:=]\s*["\'][^"\']{10,}["\']', 'Secret Assignment'),
]

# Files and directories to skip
SKIP_PATTERNS = {
    '.git', '.env.example', '.env.template', 'node_modules', 
    'venv', 'env', '__pycache__', '.pytest_cache', 
    'scan_secrets.py', 'secret_scanner.py'
}

# File extensions to scan
SCAN_EXTENSIONS = {
    '.py', '.js', '.ts', '.json', '.yaml', '.yml', 
    '.toml', '.ini', '.cfg', '.conf', '.config',
    '.sh', '.bash', '.zsh', '.fish', '.ps1',
    '.md', '.txt', '.rst', '.xml', '.html'
}

def should_scan_file(filepath: Path) -> bool:
    """Determine if a file should be scanned."""
    # Skip if in skip patterns
    for skip in SKIP_PATTERNS:
        if skip in str(filepath):
            return False
    
    # Only scan certain extensions
    if filepath.suffix.lower() not in SCAN_EXTENSIONS:
        return False
    
    return True

def scan_file(filepath: Path) -> List[Tuple[int, str, str]]:
    """Scan a single file for secrets."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines, 1):
            # Skip comments and empty lines
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('//'):
                continue
                
            for pattern, desc in SECRET_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    # Additional checks to reduce false positives
                    if 'example' in line.lower() or 'template' in line.lower():
                        continue
                    if '<your' in line.lower() or 'your-' in line.lower():
                        continue
                    if 'xxx' in line.lower() or '...' in line:
                        continue
                        
                    issues.append((line_num, desc, line.strip()))
                    break  # Only report first match per line
                    
    except Exception as e:
        print(f"{YELLOW}Warning: Could not scan {filepath}: {e}{RESET}")
    
    return issues

def scan_directory(path: Path) -> dict:
    """Scan directory recursively for secrets."""
    results = {}
    
    for root, dirs, files in os.walk(path):
        # Remove directories we should skip
        dirs[:] = [d for d in dirs if d not in SKIP_PATTERNS]
        
        for file in files:
            filepath = Path(root) / file
            if should_scan_file(filepath):
                issues = scan_file(filepath)
                if issues:
                    results[str(filepath)] = issues
    
    return results

def print_results(results: dict) -> bool:
    """Print scan results and return True if issues found."""
    if not results:
        print(f"{GREEN}✓ No secrets detected!{RESET}")
        print(f"{BLUE}Scanned files are safe to commit.{RESET}")
        return False
    
    print(f"{RED}⚠️  POTENTIAL SECRETS DETECTED!{RESET}\n")
    
    total_issues = 0
    for filepath, issues in results.items():
        print(f"{YELLOW}File: {filepath}{RESET}")
        for line_num, desc, content in issues:
            total_issues += 1
            print(f"  Line {line_num}: {desc}")
            # Ensure content is properly encoded
            display_content = content[:100] + ('...' if len(content) > 100 else '')
            print(f"  {RED}>>> {display_content}{RESET}")
        print()
    
    print(f"{RED}Total issues found: {total_issues}{RESET}")
    print(f"{YELLOW}Please review and remove all secrets before committing!{RESET}")
    
    return True

def main():
    """Main function."""
    print(f"{BLUE}EchoVault Secret Scanner{RESET}")
    print(f"{BLUE}{'=' * 50}{RESET}\n")
    
    # Get the directory to scan (current directory by default)
    scan_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    
    if not scan_path.exists():
        print(f"{RED}Error: Path {scan_path} does not exist!{RESET}")
        sys.exit(1)
    
    print(f"Scanning: {scan_path.absolute()}")
    print(f"This may take a moment...\n")
    
    # Perform the scan
    results = scan_directory(scan_path)
    
    # Print results
    has_issues = print_results(results)
    
    # Exit with appropriate code
    sys.exit(1 if has_issues else 0)

if __name__ == '__main__':
    main()
