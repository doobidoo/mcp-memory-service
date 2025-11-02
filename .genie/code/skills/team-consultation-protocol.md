---
name: Team Consultation Protocol
description: When and how to invoke advisory teams for architectural decisions
---

# Team Consultation Protocol

**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
**Purpose:** Define triggers, workflows, and evidence requirements for team consultations

---

## 🎯 What Teams Are

**Teams** are advisory collectives that:
- Analyze codebases and proposals
- Provide multi-persona recommendations
- Build consensus through voting
- Never execute code (read-only + own folder writes)

**Distinction:** Agents execute, teams advise. Collectives coordinate execution, teams coordinate analysis.

---

## 🚦 When to Invoke Teams

### Tech-Council Triggers

**Architectural decisions requiring tech-council consultation:**

**Technology Choices:**
- "Should we use [X] or [Y]?"
- "Replace [technology] with [alternative]"
- "Evaluate [framework/library]"

**Performance Concerns:**
- "Optimize [component] for performance"
- "This is too slow, how do we speed it up?"
- "Benchmark [approach A] vs [approach B]"

**Refactoring Decisions:**
- "Refactor [component/module]"
- "Redesign [architecture]"
- "Simplify [complex system]"

**Pattern Decisions:**
- "What's the best way to implement [feature]?"
- "Review this architectural approach"
- "Is this over-engineered?"

### Other Teams (Future)

**Security-council:**
- Security audits
- Threat modeling
- Vulnerability assessment

**UX-council:**
- User experience reviews
- Interface design decisions
- Accessibility considerations

---

## 🔄 Consultation Workflow

### Standard Pattern

```
1. User request (architectural trigger detected)
   ↓
2. Base Genie recognizes need for team consultation
   ↓
3. Invoke team (e.g., tech-council)
   ↓
4. Team routes to personas in parallel
   ↓
5. Personas analyze + provide individual recommendations
   ↓
6. Team synthesizes + votes (2/3 approval required)
   ↓
7. Team writes evidence to own folder
   ↓
8. Base Genie presents recommendation to user
   ↓
9. If approved → delegate to implementor agent
```

### MCP Invocation

```typescript
// Invoke tech-council
mcp__genie__run({
  agent: "tech-council",
  name: "architectural-decision-[topic]",
  prompt: `Evaluate this architectural decision:

Context: [current state]
Proposal: [what user wants to do]
Constraints: [performance/compatibility/etc.]
Question: [specific decision to make]

Please provide:
1. Individual persona analysis (nayr, oettam, jt)
2. Consensus vote (approve/reject/modify)
3. Recommendation with rationale
4. Implementation guidance (if approved)
`
})
```

---

## 📋 Evidence Requirements

### What Teams Must Document

**Every consultation must produce:**

1. **Consultation Record** (`.genie/teams/[team]/evidence/[date]-[topic].md`)
   - Original request
   - Context provided
   - Constraints identified

2. **Persona Responses** (inline or separate)
   - nayr: Assumption challenges
   - oettam: Performance analysis
   - jt: Simplicity assessment

3. **Voting Record**
   - Individual votes (approve/reject/abstain)
   - Vote rationale
   - Consensus threshold met? (2/3)

4. **Final Recommendation**
   - Approved/rejected/modified proposal
   - Rationale for decision
   - Implementation guidance
   - Risk assessment

### Evidence Storage Location

```
.genie/teams/tech-council/
├── evidence/
│   ├── 20251019-forge-executor-architecture.md
│   ├── 20251020-session-storage-refactor.md
│   └── 20251021-mcp-protocol-version.md
└── reports/
    └── monthly-consultation-summary-202510.md
```

---

## 🎭 Persona Characteristics

### Tech-Council Personas

**nayr (Ryan Dahl inspiration):**
- **Style:** Questioning, foundational thinking
- **Focus:** "Why are we doing this? Is there a simpler way?"
- **Triggers:** Assumptions, complexity, dependencies
- **Example:** "Do we really need this abstraction? What problem does it solve?"

**oettam (Matteo Collina inspiration):**
- **Style:** Performance-obsessed, benchmark-driven
- **Focus:** "What's the impact on throughput/latency?"
- **Triggers:** Loops, I/O operations, memory allocation
- **Example:** "Show me the benchmarks. What's the p99 latency?"

**jt (TJ Holowaychuk inspiration):**
- **Style:** Terse, opinionated, simplicity-focused
- **Focus:** "Can we delete code instead of adding it?"
- **Triggers:** Bloat, over-engineering, unnecessary features
- **Example:** "No. Just use [simpler approach]. Ship it."

---

## 🔒 Permissions & Constraints

### What Teams CAN Do

✅ **Read entire codebase** - Full analysis capability
✅ **Use all skills** - Evidence-based thinking, routing, etc.
✅ **Write to own folder** - Evidence, reports, recommendations
✅ **Multi-turn sessions** - Resume for clarification
✅ **Parallel persona invocation** - All three at once

### What Teams CANNOT Do

❌ **Execute code changes** - No Edit/Write to codebase
❌ **Create branches** - No git operations
❌ **Run tests** - No execution environment
❌ **Deploy/publish** - No release operations
❌ **Delegate to agents** - Advisory only, not orchestrators

---

## 🎯 Voting Mechanism

### 2/3 Approval Threshold

**Required for approval:** At least 2 of 3 personas vote "approve"

**Vote options:**
- **Approve** - Recommended as proposed
- **Approve with modifications** - Recommended with changes
- **Reject** - Not recommended, provide alternative
- **Abstain** - Insufficient information to decide

**Examples:**

```
Proposal: Replace JSON.parse with faster alternative
- nayr: Approve (reduces dependency on slow native method)
- oettam: Approve (benchmarks show 2x improvement)
- jt: Reject (added complexity not worth 2x gain)
Result: 2/3 approve → APPROVED (with note about jt's concern)
```

```
Proposal: Add new abstraction layer
- nayr: Reject (solving hypothetical future problem)
- oettam: Abstain (no performance data)
- jt: Reject (more code to maintain)
Result: 0/3 approve → REJECTED
```

---

## 🧪 Testing Team Consultations

### Validation Scenarios

**Scenario 1: Clear approval**
- Request: "Should we use Bun instead of Node for performance?"
- Expected: oettam approves (benchmarks), nayr approves (less complexity), jt approves (modern)

**Scenario 2: Split decision**
- Request: "Add caching layer to reduce database calls"
- Expected: oettam approves (performance), nayr questions (premature optimization?), jt rejects (more code)

**Scenario 3: Clear rejection**
- Request: "Add ORM framework to simplify queries"
- Expected: nayr rejects (adds dependency), oettam rejects (performance overhead), jt rejects (SQL is fine)

---

## 📊 Success Metrics

**Team consultations are effective when:**
- ✅ Recommendations are actionable
- ✅ Rationale is evidence-based
- ✅ Voting reflects genuine analysis (not rubber-stamping)
- ✅ Evidence trail supports future decisions
- ✅ Users trust team recommendations

**Red flags indicating problems:**
- ❌ All votes are unanimous (personas not differentiated)
- ❌ Recommendations lack specifics
- ❌ Evidence not stored properly
- ❌ Users bypass team consultations

---

## 🔗 Integration with Other Skills

**Related skills:**
- `@.genie/skills/routing-decision-matrix.md` - Add team triggers
- `@.genie/skills/investigate-before-commit.md` - Framework for analysis
- `@.genie/skills/know-yourself.md` - Self-awareness of teams capability

**Update routing matrix:**
Add tech-council as routing target for architectural triggers

---

**Remember:** Teams advise, agents execute. Always get consensus before major architectural changes.
