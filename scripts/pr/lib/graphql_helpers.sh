#!/bin/bash
# GraphQL helper functions for PR review thread management
#
# This library provides GraphQL operations for managing GitHub PR review threads.
# GitHub's REST API cannot resolve review threads - only GraphQL supports this.
#
# Usage:
#   source scripts/pr/lib/graphql_helpers.sh
#   get_review_threads 212
#   resolve_review_thread "MDEyOlB1bGxSZXF..." "Fixed in commit abc123"

set -e

# Get repository owner and name from git remote
# Returns: "owner/repo"
get_repo_info() {
    gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || {
        # Fallback: parse from git remote
        git remote get-url origin | sed -E 's|.*[:/]([^/]+/[^/]+)(\.git)?$|\1|'
    }
}

# Get all review threads for a PR with their IDs
# Usage: get_review_threads <PR_NUMBER>
# Returns: JSON with thread IDs, status, paths, comments
get_review_threads() {
    local pr_number=$1
    local repo_info=$(get_repo_info)
    local owner=$(echo "$repo_info" | cut -d'/' -f1)
    local repo=$(echo "$repo_info" | cut -d'/' -f2)

    gh api graphql -f query='
    query($pr: Int!, $owner: String!, $repo: String!) {
        repository(owner: $owner, name: $repo) {
            pullRequest(number: $pr) {
                reviewThreads(first: 100) {
                    nodes {
                        id
                        isResolved
                        isOutdated
                        path
                        line
                        originalLine
                        diffSide
                        comments(first: 10) {
                            nodes {
                                id
                                author { login }
                                body
                                createdAt
                            }
                        }
                    }
                }
            }
        }
    }' -f owner="$owner" -f repo="$repo" -F pr="$pr_number"
}

# Resolve a specific review thread
# Usage: resolve_review_thread <THREAD_ID> [COMMENT]
# Returns: 0 on success, 1 on failure
resolve_review_thread() {
    local thread_id=$1
    local comment=${2:-""}

    # Add explanatory comment if provided
    if [ -n "$comment" ]; then
        add_thread_reply "$thread_id" "$comment" || {
            echo "Warning: Failed to add comment, proceeding with resolution" >&2
        }
    fi

    # Resolve the thread
    gh api graphql -f query='
    mutation($threadId: ID!) {
        resolveReviewThread(input: {threadId: $threadId}) {
            thread {
                id
                isResolved
            }
        }
    }' -f threadId="$thread_id" > /dev/null
}

# Add a reply to a review thread
# Usage: add_thread_reply <THREAD_ID> <COMMENT>
# Returns: 0 on success, 1 on failure
add_thread_reply() {
    local thread_id=$1
    local comment=$2

    if [ -z "$comment" ]; then
        echo "Error: Comment body is required" >&2
        return 1
    fi

    gh api graphql -f query='
    mutation($threadId: ID!, $body: String!) {
        addPullRequestReviewThreadReply(input: {
            pullRequestReviewThreadId: $threadId
            body: $body
        }) {
            comment {
                id
            }
        }
    }' -f threadId="$thread_id" -f body="$comment" > /dev/null
}

# Get unresolved threads matching specific criteria
# Usage: get_unresolved_threads_for_file <PR_NUMBER> <FILE_PATH>
# Returns: JSON array of matching threads
get_unresolved_threads_for_file() {
    local pr_number=$1
    local file_path=$2

    get_review_threads "$pr_number" | \
        jq -r --arg file "$file_path" \
        '.data.repository.pullRequest.reviewThreads.nodes[] |
        select(.isResolved == false and .path == $file) |
        {id: .id, line: .line, comment: .comments.nodes[0].body}'
}

# Check if a line was modified in a specific commit
# Usage: was_line_modified <FILE_PATH> <LINE_NUMBER> <COMMIT_SHA>
# Returns: 0 if modified, 1 if not
was_line_modified() {
    local file_path=$1
    local line_number=$2
    local commit_sha=$3

    # Get the diff for the specific file
    # Check if line number appears in any hunk header (@@...@@)
    git diff "${commit_sha}^" "$commit_sha" -- "$file_path" | \
        awk '/^@@/ {
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            match($0, /\+([0-9]+)(,([0-9]+))?/, new_pos)
            new_start = new_pos[1]
            new_count = new_pos[3] ? new_pos[3] : 1
            new_end = new_start + new_count - 1

            # Check if target line is in this hunk
            if (line >= new_start && line <= new_end) {
                found = 1
                exit
            }
        }
        END { exit !found }' line="$line_number"
}

# Get all files modified in a commit
# Usage: get_modified_files <COMMIT_SHA>
# Returns: List of file paths (one per line)
get_modified_files() {
    local commit_sha=${1:-HEAD}
    git diff-tree --no-commit-id --name-only -r "$commit_sha"
}

# Count unresolved threads for a PR
# Usage: count_unresolved_threads <PR_NUMBER>
# Returns: Integer count
count_unresolved_threads() {
    local pr_number=$1

    get_review_threads "$pr_number" | \
        jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)] | length'
}

# Get thread summary statistics
# Usage: get_thread_stats <PR_NUMBER>
# Returns: JSON with total, resolved, unresolved, outdated counts
get_thread_stats() {
    local pr_number=$1

    get_review_threads "$pr_number" | \
        jq '{
            total: (.data.repository.pullRequest.reviewThreads.nodes | length),
            resolved: ([.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == true)] | length),
            unresolved: ([.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)] | length),
            outdated: ([.data.repository.pullRequest.reviewThreads.nodes[] | select(.isOutdated == true)] | length)
        }'
}

# Check if gh CLI supports GraphQL (requires v2.20.0+)
# Returns: 0 if supported, 1 if not
check_graphql_support() {
    if ! command -v gh &> /dev/null; then
        echo "Error: GitHub CLI (gh) is not installed" >&2
        echo "Install from: https://cli.github.com/" >&2
        return 1
    fi

    local gh_version=$(gh --version | head -1 | grep -oP '\d+\.\d+\.\d+' || echo "0.0.0")
    local major=$(echo "$gh_version" | cut -d'.' -f1)
    local minor=$(echo "$gh_version" | cut -d'.' -f2)

    if [ "$major" -lt 2 ] || ([ "$major" -eq 2 ] && [ "$minor" -lt 20 ]); then
        echo "Error: GitHub CLI version $gh_version is too old" >&2
        echo "GraphQL support requires v2.20.0 or later" >&2
        echo "Update with: gh upgrade" >&2
        return 1
    fi

    return 0
}
