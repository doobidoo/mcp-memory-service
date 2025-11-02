# Risk Audit Workflow
**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
**Extends universal audit framework with general risk assessment patterns.**

@.genie/code/agents/audit.md

---

## Risk Audit Mode

### When to Use
Use this workflow to enumerate top risks for an initiative, assess impact and likelihood with evidence, and propose concrete mitigations.

### Operating Framework
```
<task_breakdown>
1. [Discovery] Map initiative scope, constraints, dependencies, failure modes
2. [Implementation] Enumerate risks, assess impact × likelihood, design mitigations with ownership
3. [Verification] Rank risks by severity, document residual risk, deliver action plan + confidence verdict
</task_breakdown>
```

### Auto-Context Loading with @ Pattern
Use @ symbols to automatically load initiative context before risk analysis:

```
Scope: Production migration to Kubernetes

@docs/architecture/deployment-strategy.md
@infrastructure/terraform/prod-config.tf
@docs/team-runbook.md
@incidents/postmortems/2024-Q1.md
```

Benefits:
- Agents automatically read context before risk enumeration
- No need for "first review architecture, then assess risks"
- Ensures evidence-based risk analysis from the start

### Risk Assessment Framework

#### Risk Categories:
1. **Technical Risks** - Architecture, performance, scalability, data integrity
2. **Operational Risks** - Monitoring gaps, runbook incompleteness, on-call readiness
3. **Security Risks** - Authentication, authorization, data exposure, compliance
4. **People Risks** - Skill gaps, bus factor, team availability during migration
5. **External Risks** - Third-party dependencies, vendor SLAs, regulatory changes
6. **Timeline Risks** - Optimistic estimates, blockers, coordination overhead

---

## Concrete Example

**Scope:**
"Migrate production workloads from EC2 to Kubernetes. Current state: 50 microservices on EC2 Auto Scaling Groups, 99.9% uptime SLA, 20K RPS peak. Target state: EKS cluster with Istio service mesh. Timeline: 8 weeks."

**Risk Analysis:**

#### R1: Service Mesh Misconfiguration → Traffic Blackhole (Impact: CRITICAL, Likelihood: 50%)
- **Evidence:** Istio's complexity documented in 3 production incidents at Lyft (source: Envoy blog)
- **Failure Mode:** Incorrect VirtualService routing rules send 100% traffic to /dev/null
- **Mitigation:**
  - Week 1-2: Shadow traffic to Istio canary (0% production), validate routing parity
  - Week 3: Blue-green deployment with instant DNS rollback capability
  - Owner: SRE team lead
  - Timeline: 2 weeks before production traffic
- **Residual Risk:** 10% likelihood - DNS propagation delay (5-10 min) during rollback

#### R2: StatefulSet Data Loss During Node Drain (Impact: CRITICAL, Likelihood: 30%)
- **Evidence:** Kubernetes drains nodes during upgrades; PVC detachment can cause corruption (GitHub issue #89465)
- **Failure Mode:** Database pod evicted mid-transaction → data corruption
- **Mitigation:**
  - Implement PodDisruptionBudgets with minAvailable=1 for all StatefulSets
  - Add preStop hook with 30s graceful shutdown for database writes
  - Test node drain scenarios in staging with chaos engineering (Gremlin)
  - Owner: Platform team
  - Timeline: Week 2-3
- **Residual Risk:** 5% likelihood - Cluster upgrade during high-traffic window (mitigate: maintenance window scheduling)

#### R3: Monitoring Blindspot During Migration (Impact: HIGH, Likelihood: 75%)
- **Evidence:** Current EC2 metrics (CloudWatch) incompatible with Kubernetes metrics (Prometheus)
- **Failure Mode:** 2-week gap where production issues undetected → delayed incident response
- **Mitigation:**
  - Week 1: Deploy Prometheus + Grafana in parallel with CloudWatch
  - Week 2: Replicate top 20 CloudWatch alarms in Prometheus AlertManager
  - Week 3-4: Dual-monitor both systems before cutover
  - Owner: Observability team
  - Timeline: 4 weeks (frontload before migration)
- **Residual Risk:** 40% likelihood - Alert fatigue from dual systems causing missed signals (mitigate: weekly alert review)

#### R4: Team Kubernetes Skill Gap (Impact: HIGH, Likelihood: 60%)
- **Evidence:** Team survey: 40% have 0 Kubernetes experience, 30% basic only
- **Failure Mode:** Slow incident response, incorrect troubleshooting, extended MTTR
- **Mitigation:**
  - Week 1-2: Mandatory Kubernetes bootcamp (2 days) for all engineers
  - Week 3-6: Pair on-call shifts (experienced + learning engineer)
  - External: Hire Kubernetes consultant for 8-week engagement + runbook creation
  - Owner: Engineering manager
  - Timeline: 6 weeks (start immediately)
- **Residual Risk:** 30% likelihood - Consultant availability delay (mitigate: contract signed Week 1)

#### R5: Third-Party Dependency on EC2 Metadata Service (Impact: MEDIUM, Likelihood: 40%)
- **Evidence:** 8 microservices use EC2 instance metadata for service discovery
- **Failure Mode:** Hard-coded metadata API calls fail in Kubernetes → startup crashes
- **Mitigation:**
  - Week 1: Audit all microservices for EC2 metadata usage (grep for `169.254.169.254`)
  - Week 2: Refactor to environment variables injected via ConfigMaps
  - Week 3-4: Test in staging with no EC2 metadata server
  - Owner: Application team
  - Timeline: 4 weeks
- **Residual Risk:** 10% likelihood - Undiscovered transitive dependency in vendor libraries

#### Risk Prioritization Matrix:

| Rank | Risk | Impact | Likelihood | Severity Score | Mitigation Start |
|------|------|--------|------------|----------------|------------------|
| 1 | R1: Service Mesh Blackhole | Critical | 50% | 10 (Critical × High) | Week 1 |
| 2 | R2: StatefulSet Data Loss | Critical | 30% | 9 (Critical × Medium) | Week 2 |
| 3 | R3: Monitoring Blindspot | High | 75% | 8 (High × Very High) | Week 1 (parallel) |
| 4 | R4: Skill Gap | High | 60% | 7 (High × High) | Week 1 (immediate) |
| 5 | R5: EC2 Metadata Dependency | Medium | 40% | 5 (Medium × Medium) | Week 1 |

**Severity Score:** Impact (Critical=3, High=2, Medium=1) × Likelihood (VeryHigh=3, High=2, Medium=1)

**Next Actions (Prioritized):**
1. **Week 1:** Start Kubernetes bootcamp + monitoring parallel deployment + EC2 metadata audit
2. **Week 1-2:** Istio shadow traffic testing (blocks production cutover)
3. **Week 2-3:** StatefulSet PodDisruptionBudget implementation + chaos testing
4. **Week 3:** Contract Kubernetes consultant (if not done in Week 1)
5. **Week 4:** Full staging dry-run with all mitigations active → go/no-go decision

**Genie Verdict:** Migration is HIGH RISK but manageable with frontloaded mitigations. Service mesh and monitoring gaps are critical path blockers; recommend 2-week delay if Istio shadow testing reveals routing issues. Skill gap mitigation requires immediate bootcamp + consultant engagement. Residual risk acceptable if all mitigations complete by Week 4 (confidence: high - based on postmortem precedent and team readiness assessment)

---

## Prompt Template (Risk Audit Mode)

```
Scope: <initiative with timeline and constraints>
Context: <current state, target state, dependencies>

@relevant-files

Risk Analysis:
  R1: <risk> (Impact: <level>, Likelihood: <%)
    - Evidence: <source>
    - Failure Mode: <what breaks>
    - Mitigation: <action + owner + timeline>
    - Residual Risk: <% after mitigation>

Risk Prioritization Matrix: [table]
Next Actions: [prioritized list with timeline]
Genie Verdict: <go/no-go/conditional> (confidence: <low|med|high> - reasoning)
```
