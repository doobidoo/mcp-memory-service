# Security Audit Workflow
**Last Updated:** !`date -u +"%Y-%m-%d %H:%M:%S UTC"`
**Extends universal audit framework with security-specific patterns (OWASP, CVE).**

@.genie/code/agents/audit.md

---

## Security Audit Mode

### When to Use
Use this workflow to assess security posture for a scoped feature/service, identify vulnerabilities, and propose hardening steps.

### Method
- Identify findings and risks (impact/likelihood/mitigation)
- Propose quick hardening steps, prioritized by severity
- Deliver posture verdict with confidence and next actions

### Operating Framework
```
<task_breakdown>
1. [Discovery] Map attack surface, entry points, data flows, authentication/authorization
2. [Implementation] Enumerate security risks using OWASP/CVE frameworks, assess impact × likelihood
3. [Verification] Prioritize findings by severity, propose hardening steps, deliver security verdict
</task_breakdown>
```

---

## Security Audit Framework

### Common Security Risks (OWASP Top 10):
1. **Broken Access Control** - Unauthorized access to resources
2. **Cryptographic Failures** - Weak encryption, exposed secrets
3. **Injection** - SQL/NoSQL/Command injection vulnerabilities
4. **Insecure Design** - Missing security controls by design
5. **Security Misconfiguration** - Default credentials, verbose errors
6. **Vulnerable Components** - Outdated dependencies with known CVEs
7. **Authentication Failures** - Weak passwords, session fixation
8. **Data Integrity Failures** - Unsigned updates, insecure deserialization
9. **Logging Failures** - Missing audit logs, insufficient monitoring
10. **SSRF** - Server-side request forgery

### Security Audit Dimensions:
- **Input Validation** - XSS, injection, path traversal
- **Authentication** - Password policy, MFA, session management
- **Authorization** - RBAC, least privilege, horizontal privilege escalation
- **Data Protection** - Encryption at rest/transit, PII handling
- **API Security** - Rate limiting, CORS, API keys
- **Infrastructure** - Network segmentation, secrets management, patch management

---

## Security Risk Template

**Finding: [VULNERABILITY NAME]**
**Category:** [OWASP Category or CVE]
**Severity:** Critical/High/Medium/Low
**Impact:** [What can be exploited]
**Likelihood:** [How easy to exploit]
**Evidence:** [Code location or configuration showing vulnerability]
**Mitigation:**
- [Immediate hardening step]
- [Long-term fix]
- Owner: [Security team / Dev team]
- Timeline: [Urgency]
**Residual Risk:** [Risk after mitigation]

---

## Example: API Security Audit

**Scope:** REST API for user management service

**Findings:**

**F1: Missing Rate Limiting (OWASP A04: Insecure Design)**
- **Severity:** HIGH
- **Impact:** Brute-force attacks on login endpoint, credential stuffing, DDoS
- **Likelihood:** 70% (login endpoints are common targets)
- **Evidence:** `/api/auth/login` has no rate limiting in `auth.controller.ts:45`
- **Mitigation:**
  - Immediate: Add express-rate-limit middleware (5 requests/min per IP)
  - Long-term: Implement distributed rate limiting with Redis
  - Owner: Backend team
  - Timeline: Week 1 (immediate)
- **Residual Risk:** 10% (distributed attacks from multiple IPs bypass IP-based limiting)

**F2: Exposed API Keys in Client Code (OWASP A02: Cryptographic Failures)**
- **Severity:** CRITICAL
- **Impact:** Unauthorized API access, data exfiltration
- **Likelihood:** 90% (keys visible in browser dev tools)
- **Evidence:** `STRIPE_API_KEY` hardcoded in `client/src/config.ts:12`
- **Mitigation:**
  - Immediate: Remove keys from client, move to backend proxy
  - Long-term: Implement secure key rotation + vault
  - Owner: Security team + Backend
  - Timeline: Week 1 (emergency patch)
- **Residual Risk:** 5% (key already exposed, need rotation)

**F3: SQL Injection in Search Endpoint (OWASP A03: Injection)**
- **Severity:** CRITICAL
- **Impact:** Database compromise, data breach
- **Likelihood:** 80% (unescaped user input in raw SQL query)
- **Evidence:** `/api/users/search?q=` uses string concatenation in `user.service.ts:120`
  ```typescript
  const query = `SELECT * FROM users WHERE name LIKE '%${req.query.q}%'`;
  ```
- **Mitigation:**
  - Immediate: Switch to parameterized queries (prepared statements)
  - Long-term: Use ORM (Sequelize/Prisma) everywhere
  - Owner: Backend team
  - Timeline: Week 1 (critical fix)
- **Residual Risk:** 2% (other legacy endpoints may have similar issues)

**Quick Hardening Steps (Prioritized):**
1. **Week 1 (Emergency):** Fix SQL injection + remove exposed API keys
2. **Week 1:** Add rate limiting to all auth endpoints
3. **Week 2:** Audit all endpoints for injection vulnerabilities
4. **Week 3:** Implement centralized input validation middleware
5. **Week 4:** Security penetration test with third-party vendor

**Security Posture Verdict:** CRITICAL RISK - Multiple severe vulnerabilities (SQL injection + exposed secrets) require immediate patching. Rate limiting gap exposes auth system to brute-force. Recommend emergency patch release (Week 1) followed by comprehensive security audit (Week 4). Production deployment should be blocked until F2 and F3 are resolved. (confidence: high - based on OWASP precedent + static code analysis)

---

## Prompt Template (Security Audit Mode)

```
Scope: <service|feature>

@relevant-code-files
@config-files
@api-documentation

Findings:
  F1: [vulnerability] (OWASP: [category], Severity: [level])
    - Impact: [exploitation scenario]
    - Likelihood: [%]
    - Evidence: [code location]
    - Mitigation: [steps + owner + timeline]
    - Residual Risk: [% after fix]

Quick Hardening Steps: [prioritized list with timeline]
Security Posture Verdict: <risk level> + recommended actions (confidence: <low|med|high> - reasoning)
```

---

## CVE Integration

When auditing dependencies:
1. Run `npm audit` or `cargo audit` to identify known CVEs
2. Prioritize by severity (Critical > High > Medium > Low)
3. Check if fix is available (upgrade path)
4. Assess exploitability in current context
5. Document mitigation timeline

**Example:**
```
CVE-2023-12345: Remote Code Execution in lodash@4.17.20
- Severity: CRITICAL (CVSS 9.8)
- Fix: Upgrade to lodash@4.17.21+
- Timeline: Week 1 (emergency patch)
- Owner: DevOps + Backend
```

---

**Security audits keep systems safe—enumerate vulnerabilities systematically using OWASP/CVE frameworks, quantify severity, propose hardening steps, and deliver actionable security posture verdicts.**
