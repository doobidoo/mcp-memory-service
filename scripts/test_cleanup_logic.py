#!/usr/bin/env python3
"""
Test script for Docker Hub cleanup logic
Tests the retention policy rules without actual API calls
"""

import re
from datetime import datetime, timedelta, timezone

def should_keep_tag(tag_name, tag_date, keep_versions=5, cutoff_date=None):
    """Test version of the retention policy logic"""
    if cutoff_date is None:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
    
    # Always keep these tags
    protected_tags = ["latest", "slim", "main", "stable"]
    if tag_name in protected_tags:
        return True, "Protected tag"
    
    # Keep semantic version tags (v1.2.3)
    if re.match(r'^v?\d+\.\d+\.\d+$', tag_name):
        return True, "Semantic version"
    
    # Keep major.minor tags (1.0, 2.1)
    if re.match(r'^v?\d+\.\d+$', tag_name):
        return True, "Major.minor version"
    
    # Delete buildcache tags older than cutoff
    if tag_name.startswith("buildcache-"):
        if tag_date < cutoff_date:
            return False, "Old buildcache tag"
        return True, "Recent buildcache tag"
    
    # Delete sha/digest tags older than cutoff
    if tag_name.startswith("sha256-") or (len(tag_name) == 7 and tag_name.isalnum()):
        if tag_date < cutoff_date:
            return False, "Old sha/digest tag"
        return True, "Recent sha/digest tag"
    
    # Delete test/dev tags older than cutoff
    if any(x in tag_name.lower() for x in ["test", "dev", "tmp", "temp"]):
        if tag_date < cutoff_date:
            return False, "Old test/dev tag"
        return True, "Recent test/dev tag"
    
    # Keep if recent
    if tag_date >= cutoff_date:
        return True, "Recent tag"
    
    return False, "Old tag"

def test_retention_policy():
    """Test various tag scenarios"""
    now = datetime.now(timezone.utc)
    old_date = now - timedelta(days=40)
    recent_date = now - timedelta(days=10)
    cutoff = now - timedelta(days=30)
    
    test_cases = [
        # (tag_name, tag_date, expected_keep, expected_reason)
        ("latest", old_date, True, "Protected tag"),
        ("slim", old_date, True, "Protected tag"),
        ("main", old_date, True, "Protected tag"),
        ("stable", old_date, True, "Protected tag"),
        
        ("v6.6.0", old_date, True, "Semantic version"),
        ("6.6.0", old_date, True, "Semantic version"),
        ("v6.6", old_date, True, "Major.minor version"),
        ("6.6", old_date, True, "Major.minor version"),
        
        ("buildcache-linux-amd64", old_date, False, "Old buildcache tag"),
        ("buildcache-linux-amd64", recent_date, True, "Recent buildcache tag"),
        
        ("sha256-abc123", old_date, False, "Old sha/digest tag"),
        ("abc1234", old_date, False, "Old sha/digest tag"),
        ("sha256-abc123", recent_date, True, "Recent sha/digest tag"),
        
        ("test-feature", old_date, False, "Old test/dev tag"),
        ("dev-branch", old_date, False, "Old test/dev tag"),
        ("tmp-build", recent_date, True, "Recent test/dev tag"),
        
        ("feature-xyz", old_date, False, "Old tag"),
        ("feature-xyz", recent_date, True, "Recent tag"),
    ]
    
    print("Testing Docker Hub Cleanup Retention Policy")
    print("=" * 60)
    print(f"Cutoff date: {cutoff.strftime('%Y-%m-%d')}")
    print()
    
    passed = 0
    failed = 0
    
    for tag_name, tag_date, expected_keep, expected_reason in test_cases:
        should_keep, reason = should_keep_tag(tag_name, tag_date, cutoff_date=cutoff)
        
        # Format date for display
        date_str = tag_date.strftime('%Y-%m-%d')
        days_old = (now - tag_date).days
        
        # Check if test passed
        if should_keep == expected_keep and reason == expected_reason:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
            
        # Print result
        action = "KEEP" if should_keep else "DELETE"
        print(f"{status}: {tag_name:30} ({days_old:3}d old) -> {action:6} ({reason})")
        
        if status == "✗ FAIL":
            print(f"       Expected: {action:6} ({expected_reason})")
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0

if __name__ == "__main__":
    success = test_retention_policy()
    exit(0 if success else 1)