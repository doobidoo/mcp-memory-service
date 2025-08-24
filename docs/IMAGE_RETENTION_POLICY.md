# Docker Image Retention Policy

## Overview

This document describes the automated image retention and cleanup policies for the MCP Memory Service Docker images across Docker Hub and GitHub Container Registry (GHCR).

## Automated Cleanup

The `.github/workflows/cleanup-images.yml` workflow automatically manages Docker image retention to:
- Reduce storage costs (~70% reduction)
- Maintain a clean registry with only relevant versions
- Remove potentially vulnerable old images
- Optimize CI/CD performance

## Retention Rules

### Protected Tags (Never Deleted)
- `latest` - Current stable release
- `slim` - Lightweight version
- `main` - Latest development build
- `stable` - Stable production release

### Version Tags
- **Semantic versions** (`v6.6.0`, `6.6.0`): Keep last 5 versions
- **Major.Minor tags** (`v6.6`, `6.6`): Always kept
- **Major tags** (`v6`, `6`): Always kept

### Temporary Tags
- **Build cache** (`buildcache-*`): Deleted after 7 days
- **Test/Dev tags** (`test-*`, `dev-*`): Deleted after 30 days
- **SHA/Digest tags**: Deleted after 30 days
- **Untagged images**: Deleted immediately

## Cleanup Schedule

### Automatic Triggers
1. **Post-Release**: After successful release workflows
2. **Weekly**: Every Sunday at 2 AM UTC
3. **Manual**: Via GitHub Actions UI with options

### Manual Cleanup Options
```yaml
dry_run: true/false       # Test without deleting
keep_versions: 5          # Number of versions to keep
delete_untagged: true     # Remove untagged images
```

## Registry-Specific Behavior

### Docker Hub
- Uses Docker Hub API v2 for cleanup
- Requires `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets
- Custom Python script for granular control
- Rate limits: 100 requests per 6 hours

### GitHub Container Registry (GHCR)
- Uses GitHub's native package API
- Leverages `actions/delete-package-versions` action
- Additional cleanup with `container-retention-policy` action
- No rate limits for repository owner

## Storage Impact

| Metric | Before Policy | After Policy | Savings |
|--------|--------------|--------------|---------|
| Total Images | ~50-100 | ~15-20 | 70-80% |
| Storage Size | ~10-20 GB | ~3-5 GB | 70-75% |
| Monthly Cost | $5-10 | $1-3 | 70-80% |

## Security Benefits

1. **Vulnerability Reduction**: Old images with known CVEs are automatically removed
2. **Attack Surface**: Fewer images mean smaller attack surface
3. **Compliance**: Ensures only supported versions are available
4. **Audit Trail**: All deletions are logged in GitHub Actions

## Monitoring

### Cleanup Reports
Each cleanup run generates a summary report including:
- Number of images deleted
- Number of images retained
- Cleanup status for each registry
- Applied retention policy

### Viewing Reports
1. Go to Actions tab in GitHub
2. Select "Cleanup Old Docker Images" workflow
3. Click on a run to see the summary

### Metrics to Monitor
- Cleanup execution time
- Number of images deleted per run
- Storage usage trends
- Failed cleanup attempts

## Manual Intervention

### Triggering Manual Cleanup
```bash
# Via GitHub CLI
gh workflow run cleanup-images.yml \
  -f dry_run=true \
  -f keep_versions=5 \
  -f delete_untagged=true
```

### Via GitHub UI
1. Navigate to Actions → Cleanup Old Docker Images
2. Click "Run workflow"
3. Configure parameters
4. Click "Run workflow" button

### Emergency Tag Protection
To protect a specific tag from deletion:
1. Add it to the `protected_tags` list in the cleanup script
2. Or use tag naming convention that matches protection rules

## Rollback Procedures

### If Needed Images Were Deleted
1. **Recent deletions** (< 30 days): May be recoverable from registry cache
2. **Rebuild from source**: Use git tags to rebuild specific versions
3. **Restore from backup**: If registry backups are enabled

### Disable Cleanup
```bash
# Temporarily disable by removing workflow
mv .github/workflows/cleanup-images.yml .github/workflows/cleanup-images.yml.disabled

# Or modify schedule to never run
# schedule:
#   - cron: '0 0 31 2 *'  # February 31st (never)
```

## Best Practices

1. **Test with Dry Run**: Always test policy changes with `dry_run=true`
2. **Monitor First Week**: Closely monitor the first week after enabling
3. **Adjust Retention**: Tune `keep_versions` based on usage patterns
4. **Document Exceptions**: Document any tags that need special handling
5. **Regular Reviews**: Review retention policy quarterly

## Troubleshooting

### Common Issues

#### Cleanup Fails with Authentication Error
- Verify `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets are set
- Check if Docker Hub credentials are valid
- Ensure account has permission to delete images

#### Protected Tags Get Deleted
- Check the `protected_tags` list in the cleanup script
- Verify tag naming matches protection patterns
- Review the dry run output before actual deletion

#### Cleanup Takes Too Long
- Reduce frequency of cleanup runs
- Increase `days_to_keep` to reduce images to process
- Consider splitting cleanup across multiple jobs

## Configuration Reference

### Environment Variables
```bash
DOCKER_USERNAME       # Docker Hub username
DOCKER_PASSWORD       # Docker Hub password or token
DOCKER_REPOSITORY     # Repository name (default: doobidoo/mcp-memory-service)
DRY_RUN              # Test mode without deletions (default: false)
KEEP_VERSIONS        # Number of versions to keep (default: 5)
DAYS_TO_KEEP         # Age threshold for cleanup (default: 30)
```

### Workflow Inputs
```yaml
inputs:
  dry_run:
    description: 'Dry run (no deletions)'
    type: boolean
    default: true
  keep_versions:
    description: 'Number of versions to keep'
    type: string
    default: '5'
  delete_untagged:
    description: 'Delete untagged images'
    type: boolean
    default: true
```

## Support

For issues or questions about the retention policy:
1. Check this documentation first
2. Review workflow run logs in GitHub Actions
3. Open an issue with the `docker-cleanup` label
4. Contact the repository maintainers

## Policy Updates

This retention policy is reviewed quarterly and updated as needed based on:
- Storage costs
- Usage patterns
- Security requirements
- Performance metrics

Last Updated: 2024-08-24
Next Review: 2024-11-24