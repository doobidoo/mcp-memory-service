# Advanced Claude Code Commands - Real-World Examples

This guide showcases advanced usage patterns and real-world workflows using MCP Memory Service Claude Code commands.

## Table of Contents
- [Development Workflows](#development-workflows)
- [Project Management](#project-management)
- [Learning & Knowledge Management](#learning--knowledge-management)
- [Team Collaboration](#team-collaboration)
- [Debugging & Troubleshooting](#debugging--troubleshooting)
- [Advanced Search Techniques](#advanced-search-techniques)
- [Automation & Scripting](#automation--scripting)

---

## Development Workflows

### Full-Stack Development Session

**Scenario**: Working on a web application with authentication

```bash
# Start development session with context capture
claude /memory-context --summary "Starting OAuth 2.0 integration for user authentication"

# Store architecture decisions as you make them
claude /memory-store --tags "architecture,oauth,security" \
  "Using Authorization Code flow with PKCE for mobile app security"

claude /memory-store --tags "database,schema" --type "reference" \
  "User table schema: id, email, oauth_provider, oauth_id, created_at, last_login"

# Store implementation details
claude /memory-store --tags "implementation,react" \
  "React auth context uses useReducer for state management with actions: LOGIN, LOGOUT, REFRESH_TOKEN"

# Store configuration details (marked as private)
claude /memory-store --tags "config,oauth" --type "reference" --private \
  "Auth0 configuration: domain=dev-xyz.auth0.com, audience=https://api.myapp.com"

# Later, recall decisions when working on related features
claude /memory-recall "what did we decide about OAuth implementation yesterday?"

# Search for specific implementation patterns
claude /memory-search --tags "react,auth" "state management patterns"

# End session with comprehensive context capture
claude /memory-context --summary "Completed OAuth integration - ready for testing" \
  --include-files --include-commits
```

### Bug Fixing Workflow

**Scenario**: Tracking and resolving a complex bug

```bash
# Store bug discovery
claude /memory-store --tags "bug,critical,payment" --type "task" \
  "Payment processing fails for amounts over $1000 - investigation needed"

# Store investigation findings
claude /memory-store --tags "bug,payment,stripe" \
  "Issue traced to Stripe API rate limiting on high-value transactions"

# Store attempted solutions
claude /memory-store --tags "bug,payment,attempted-fix" \
  "Tried increasing timeout from 5s to 30s - did not resolve issue"

# Store working solution
claude /memory-store --tags "bug,payment,solution" --type "decision" \
  "Fixed by implementing exponential backoff retry mechanism with max 3 attempts"

# Create searchable reference for future
claude /memory-store --tags "reference,stripe,best-practice" \
  "Stripe high-value transactions require retry logic - see payment-service.js line 245"

# Later, search for similar issues
claude /memory-search --tags "bug,stripe" "rate limiting payment"
```

### Code Review & Refactoring

**Scenario**: Systematic code improvement process

```bash
# Store code review insights
claude /memory-store --tags "code-review,performance,database" \
  "N+1 query problem in user dashboard - fetching posts individually instead of batch"

# Store refactoring decisions
claude /memory-store --tags "refactoring,database,optimization" --type "decision" \
  "Replaced individual queries with single JOIN query - reduced DB calls from 50+ to 1"

# Store before/after metrics
claude /memory-store --tags "performance,metrics,improvement" \
  "Dashboard load time: Before=2.3s, After=0.4s (83% improvement)"

# Track technical debt
claude /memory-store --tags "technical-debt,todo" --type "task" \
  "TODO: Extract user dashboard logic into dedicated service class"

# Review improvements over time
claude /memory-recall "what performance improvements did we make this month?"
```

---

## Project Management

### Sprint Planning & Tracking

**Scenario**: Agile development with memory-enhanced tracking

```bash
# Start of sprint
claude /memory-context --summary "Sprint 15 planning - Focus on user onboarding improvements"

# Store sprint goals
claude /memory-store --tags "sprint-15,goals,onboarding" --type "planning" \
  "Sprint 15 goals: Simplify signup flow, add email verification, implement welcome tour"

# Track daily progress
claude /memory-store --tags "sprint-15,progress,day-1" \
  "Completed signup form validation and error handling - 2 story points"

# Store blockers and risks
claude /memory-store --tags "sprint-15,blocker,email" --type "task" \
  "Email service integration blocked - waiting for IT to configure SendGrid account"

# Mid-sprint review
claude /memory-recall "what blockers did we identify this sprint?"
claude /memory-search --tags "sprint-15,progress"

# Sprint retrospective
claude /memory-store --tags "sprint-15,retrospective" --type "meeting" \
  "Sprint 15 retro: Delivered 18/20 points, email blocker resolved, team velocity improving"

# Cross-sprint analysis
claude /memory-search --tags "retrospective" --limit 5
claude /memory-recall "what patterns do we see in our sprint blockers?"
```

### Feature Development Lifecycle

**Scenario**: End-to-end feature development tracking

```bash
# Feature inception
claude /memory-store --tags "feature,user-profiles,inception" --type "planning" \
  "User profiles feature: Allow users to customize avatar, bio, social links, privacy settings"

# Requirements gathering
claude /memory-store --tags "feature,user-profiles,requirements" \
  "Requirements: Image upload (max 2MB), bio text (max 500 chars), 5 social links, public/private toggle"

# Technical design
claude /memory-store --tags "feature,user-profiles,design" --type "architecture" \
  "Design: New profiles table, S3 for image storage, React profile editor component"

# Implementation milestones
claude /memory-store --tags "feature,user-profiles,milestone" \
  "Milestone 1 complete: Database schema created and migrated to production"

# Testing notes
claude /memory-store --tags "feature,user-profiles,testing" \
  "Testing discovered: Large images cause timeout - need client-side compression"

# Launch preparation
claude /memory-store --tags "feature,user-profiles,launch" \
  "Launch checklist: DB migration ✓, S3 bucket ✓, feature flag ready ✓, docs updated ✓"

# Post-launch analysis
claude /memory-store --tags "feature,user-profiles,metrics" \
  "Week 1 metrics: 45% adoption rate, avg 3.2 social links per profile, 12% privacy toggle usage"

# Feature evolution tracking
claude /memory-search --tags "feature,user-profiles" --limit 20
```

---

## Learning & Knowledge Management

### Technology Research & Evaluation

**Scenario**: Evaluating new technologies for adoption

```bash
# Research session start
claude /memory-context --summary "Researching GraphQL vs REST API for mobile app backend"

# Store research findings
claude /memory-store --tags "research,graphql,pros" \
  "GraphQL benefits: Single endpoint, client-defined queries, strong typing, introspection"

claude /memory-store --tags "research,graphql,cons" \
  "GraphQL challenges: Learning curve, caching complexity, N+1 query risk, server complexity"

# Store comparison data
claude /memory-store --tags "research,performance,comparison" \
  "Performance test: GraphQL 340ms avg, REST 280ms avg for mobile app typical queries"

# Store team feedback
claude /memory-store --tags "research,team-feedback,graphql" \
  "Team survey: 60% excited about GraphQL, 30% prefer REST familiarity, 10% neutral"

# Store decision and rationale
claude /memory-store --tags "decision,architecture,graphql" --type "decision" \
  "Decision: Adopt GraphQL for new features, maintain REST for existing APIs during 6-month transition"

# Create reference documentation
claude /memory-store --tags "reference,graphql,implementation" \
  "GraphQL implementation guide: Use Apollo Server, implement DataLoader for N+1 prevention"

# Later research sessions
claude /memory-recall "what did we learn about GraphQL performance last month?"
claude /memory-search --tags "research,comparison" "technology evaluation"
```

### Personal Learning Journal

**Scenario**: Building a personal knowledge base

```bash
# Daily learning capture
claude /memory-store --tags "learning,javascript,async" \
  "Learned: Promise.allSettled() waits for all promises unlike Promise.all() which fails fast"

claude /memory-store --tags "learning,css,flexbox" \
  "CSS trick: flex-grow: 1 on middle item makes it expand to fill available space"

# Code snippets and examples
claude /memory-store --tags "snippet,react,custom-hook" --type "reference" \
  "Custom hook pattern: useLocalStorage - encapsulates localStorage with React state sync"

# Book and article insights
claude /memory-store --tags "book,clean-code,insight" \
  "Clean Code principle: Functions should do one thing well - if function has 'and' in description, split it"

# Conference and talk notes
claude /memory-store --tags "conference,react-conf,2024" \
  "React Conf 2024: New concurrent features in React 18.3, Server Components adoption patterns"

# Weekly knowledge review
claude /memory-recall "what did I learn about React this week?"
claude /memory-search --tags "learning,javascript" --limit 10

# Monthly learning patterns
claude /memory-search --tags "learning" --since "last month"
```

---

## Team Collaboration

### Cross-Team Communication

**Scenario**: Working with multiple teams on shared systems

```bash
# Store cross-team decisions
claude /memory-store --tags "team,frontend,backend,api-contract" --type "decision" \
  "API contract agreed: User service will return ISO 8601 timestamps, frontend will handle timezone conversion"

# Store meeting outcomes
claude /memory-store --tags "meeting,security-team,compliance" \
  "Security review outcome: Authentication service approved for production with rate limiting requirement"

# Store shared resource information
claude /memory-store --tags "shared,database,access" --type "reference" \
  "Shared analytics DB access: Use service account sa-analytics@company.com, read-only access"

# Track dependencies and blockers
claude /memory-store --tags "dependency,infrastructure,blocker" --type "task" \
  "Blocked on infrastructure team: Need production K8s namespace for user-service deployment"

# Store team contact information
claude /memory-store --tags "team,contacts,infrastructure" --type "reference" --private \
  "Infrastructure team: Primary contact Alex Chen (alex@company.com), escalation Sarah Kim (sarah@company.com)"

# Regular cross-team syncs
claude /memory-recall "what dependencies do we have on the infrastructure team?"
claude /memory-search --tags "team,backend" "shared decisions"
```

### Code Handoff & Documentation

**Scenario**: Preparing code handoff to another developer

```bash
# Store system overview for handoff
claude /memory-store --tags "handoff,system-overview,payment-service" \
  "Payment service architecture: Express.js API, PostgreSQL DB, Redis cache, Stripe integration"

# Document key implementation details
claude /memory-store --tags "handoff,implementation,payment-service" \
  "Key files: server.js (main app), routes/payments.js (API endpoints), services/stripe.js (integration logic)"

# Store operational knowledge
claude /memory-store --tags "handoff,operations,payment-service" \
  "Monitoring: Grafana dashboard 'Payment Service', alerts on Slack #payments-alerts channel"

# Document gotchas and edge cases
claude /memory-store --tags "handoff,gotchas,payment-service" \
  "Known issues: Webhook retries can cause duplicate processing - check payment_id before processing"

# Store testing information
claude /memory-store --tags "handoff,testing,payment-service" \
  "Testing: npm test (unit), npm run test:integration, Stripe test cards in test-data.md"

# Create comprehensive handoff package
claude /memory-search --tags "handoff,payment-service" --export
```

---

## Debugging & Troubleshooting

### Production Issue Investigation

**Scenario**: Investigating and resolving production incidents

```bash
# Store incident details
claude /memory-store --tags "incident,p1,database,performance" --type "task" \
  "P1 Incident: Database connection timeouts causing 504 errors, affecting 15% of users"

# Store investigation timeline
claude /memory-store --tags "incident,investigation,timeline" \
  "10:15 AM - First reports of timeouts, 10:22 AM - Confirmed DB connection pool exhaustion"

# Store root cause analysis
claude /memory-store --tags "incident,root-cause,connection-pool" \
  "Root cause: Connection pool size (10) insufficient for increased traffic, no connection recycling"

# Store immediate fixes applied
claude /memory-store --tags "incident,fix,immediate" \
  "Immediate fix: Increased connection pool to 50, enabled connection recycling, deployed at 11:30 AM"

# Store monitoring improvements
claude /memory-store --tags "incident,monitoring,improvement" \
  "Added monitoring: DB connection pool utilization alerts at 80% threshold"

# Store prevention measures
claude /memory-store --tags "incident,prevention,long-term" --type "task" \
  "Long-term prevention: Implement connection pool auto-scaling, add load testing to CI/CD"

# Post-incident review
claude /memory-store --tags "incident,postmortem,lessons" \
  "Lessons learned: Need proactive monitoring of resource utilization, not just error rates"

# Search for similar incidents
claude /memory-search --tags "incident,database" "connection timeout"
```

### Performance Optimization Tracking

**Scenario**: Systematic performance improvement initiative

```bash
# Baseline measurements
claude /memory-store --tags "performance,baseline,api-response" \
  "Baseline metrics: API avg response time 850ms, p99 2.1s, DB query avg 340ms"

# Store optimization experiments
claude /memory-store --tags "performance,experiment,caching" \
  "Experiment 1: Added Redis caching for user profiles - 30% response time improvement"

claude /memory-store --tags "performance,experiment,database" \
  "Experiment 2: Optimized N+1 queries in posts endpoint - 45% DB query time reduction"

# Track measurement methodology
claude /memory-store --tags "performance,methodology,testing" \
  "Load testing setup: k6 script, 100 VU, 5min ramp-up, 10min steady state, production-like data"

# Store optimization results
claude /memory-store --tags "performance,results,final" \
  "Final metrics: API avg response time 420ms (51% improvement), p99 980ms (53% improvement)"

# Document optimization techniques
claude /memory-store --tags "performance,techniques,reference" --type "reference" \
  "Optimization techniques applied: Redis caching, query optimization, connection pooling, response compression"

# Performance trend analysis
claude /memory-recall "what performance improvements did we achieve this quarter?"
```

---

## Advanced Search Techniques

### Complex Query Patterns

**Scenario**: Advanced search strategies for large knowledge bases

```bash
# Multi-tag searches with boolean logic
claude /memory-search --tags "architecture,database" --content "performance"
# Finds memories tagged with both architecture AND database, containing performance-related content

# Time-constrained searches
claude /memory-search --tags "bug,critical" --since "last week" --limit 20
# Recent critical bugs only

# Project-specific technical searches
claude /memory-search --project "user-service" --type "decision" --content "authentication"
# Architecture decisions about authentication in specific project

# Minimum relevance threshold searches
claude /memory-search --min-score 0.8 "microservices communication patterns"
# Only highly relevant results about microservices communication

# Comprehensive metadata searches
claude /memory-search --include-metadata --tags "api,design" --export
# Full metadata export for API design memories
```

### Research and Analysis Queries

**Scenario**: Analyzing patterns and trends in stored knowledge

```bash
# Trend analysis across time periods
claude /memory-recall "what architectural decisions did we make in Q1?"
claude /memory-recall "what architectural decisions did we make in Q2?"
# Compare decision patterns across quarters

# Technology adoption tracking
claude /memory-search --tags "adoption" --content "react"
claude /memory-search --tags "adoption" --content "vue"
# Compare technology adoption discussions

# Problem pattern identification
claude /memory-search --tags "bug,database" --limit 50
# Identify common database-related issues

# Team learning velocity analysis
claude /memory-search --tags "learning" --since "last month"
# Recent learning activities

# Decision outcome tracking
claude /memory-search --tags "decision" --content "outcome"
# Find decisions with documented outcomes
```

---

## Automation & Scripting

### Automated Memory Capture

**Scenario**: Scripting common memory operations

```bash
# Daily standup automation
#!/bin/bash
# daily-standup.sh
DATE=$(date +"%Y-%m-%d")
echo "What did you accomplish yesterday?" | read YESTERDAY
echo "What are you working on today?" | read TODAY
echo "Any blockers?" | read BLOCKERS

claude /memory-store --tags "standup,daily,$DATE" \
  "Yesterday: $YESTERDAY. Today: $TODAY. Blockers: $BLOCKERS"

# Git commit message enhancement
#!/bin/bash
# enhanced-commit.sh
COMMIT_MSG="$1"
BRANCH=$(git branch --show-current)
FILES=$(git diff --name-only --cached)

claude /memory-store --tags "commit,$BRANCH,development" \
  "Commit: $COMMIT_MSG. Files: $FILES. Branch: $BRANCH"

git commit -m "$COMMIT_MSG"

# End-of-day summary
#!/bin/bash
# eod-summary.sh
claude /memory-context --summary "End of day summary - $(date +%Y-%m-%d)" \
  --include-files --include-commits
```

### Batch Operations and Analysis

**Scenario**: Processing multiple memories for analysis

```bash
# Export all architectural decisions for documentation
claude /memory-search --tags "architecture,decision" --limit 100 --export
# Creates exportable report of all architectural decisions

# Weekly learning summary
claude /memory-search --tags "learning" --since "last week" --export
# Export week's learning for review

# Project retrospective data gathering
claude /memory-search --project "user-service" --tags "bug,issue" --export
claude /memory-search --project "user-service" --tags "success,milestone" --export
# Gather both problems and successes for retrospective

# Technical debt analysis
claude /memory-search --tags "technical-debt,todo" --include-metadata --export
# Comprehensive technical debt report

# Performance trend analysis
claude /memory-search --tags "performance,metrics" --limit 50 --export
# Historical performance data for trend analysis
```

---

## Best Practices Summary

### Effective Tagging Strategies
- **Hierarchical tags**: Use `team`, `team-frontend`, `team-frontend-react`
- **Temporal tags**: Include sprint numbers, quarters, years
- **Context tags**: Project names, feature names, team names
- **Type tags**: `decision`, `bug`, `learning`, `reference`, `todo`
- **Priority tags**: `critical`, `important`, `nice-to-have`

### Content Organization
- **Specific titles**: Instead of "Fixed bug", use "Fixed payment timeout bug in Stripe integration"
- **Context inclusion**: Always include relevant project and time context
- **Outcome documentation**: Store not just decisions but their outcomes
- **Link related memories**: Reference related decisions and implementations

### Search Optimization
- **Use multiple search strategies**: Combine tags, content, and time-based searches
- **Iterate searches**: Start broad, then narrow down with additional filters
- **Export important results**: Save comprehensive analyses for documentation
- **Regular reviews**: Weekly/monthly searches to identify patterns

### Workflow Integration
- **Start sessions with context**: Use `/memory-context` to set session scope
- **Real-time capture**: Store decisions and insights as they happen
- **End sessions with summary**: Capture session outcomes and next steps
- **Regular retrospectives**: Use search to analyze patterns and improvements

---

**These advanced patterns will transform your development workflow with persistent, searchable memory that grows with your expertise!** 🚀
