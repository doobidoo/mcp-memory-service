# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions of MCP Memory Service:

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| 11.x.x  | :white_check_mark: | Current stable release - full support |
| < 11.0  | :x:                | No longer supported - upgrade to 11.x |

There is no maintenance branch for older major versions. Security fixes land on
`main` and ship in the next 11.x release; they are not backported.

## Reporting a Vulnerability

We take the security of MCP Memory Service seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

Development of this project happens on
[GitHub](https://github.com/doobidoo/mcp-memory-service). Issues there are public from
the moment they are opened, which is fine for everything except an unfixed
vulnerability. GitHub Security Advisories are the confidential channel for those: the
report stays private until a fix ships, and that is where sensitive reports go.

**For sensitive security issues** - anything exploitable, or anything you would not
want public before a fix ships:

- Open a private report through [GitHub Security Advisories](https://github.com/doobidoo/mcp-memory-service/security/advisories/new)
- Provide detailed information about the vulnerability (see below)

**For non-sensitive security concerns** - hardening suggestions, dependency
questions, defaults you find too permissive - open a regular
[GitHub issue](https://github.com/doobidoo/mcp-memory-service/issues).

### What to Include

When reporting a vulnerability, please include:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential security impact and affected versions
3. **Reproduction**: Step-by-step instructions to reproduce the issue
4. **Environment**:
   - Python version
   - Operating system
   - Storage backend (SQLite-vec, Cloudflare, Hybrid)
   - Installation method (pip, Docker, source)
5. **Proof of Concept**: Code or commands demonstrating the vulnerability (if applicable)
6. **Suggested Fix**: Any ideas for fixing the issue (optional)

### Response Timeline

We aim to respond to security reports according to the following timeline:

- **Acknowledgment**: Within 48 hours of report
- **Initial Assessment**: Within 5 business days
- **Status Updates**: Weekly until resolved
- **Fix Development**: 7-14 days for high-severity issues
- **Patch Release**: As soon as fix is validated and tested
- **Public Disclosure**: After patch is released (coordinated with reporter)

### Severity Classification

We use the following severity levels to prioritize security issues:

**Critical** 🔴
- Remote code execution
- Authentication bypass
- Data exfiltration from other users' memories
- Complete system compromise

**High** 🟠
- Privilege escalation
- SQL injection
- Cross-site scripting (XSS) in dashboard
- Denial of service affecting all users

**Medium** 🟡
- Information disclosure (limited scope)
- Cross-site request forgery (CSRF)
- Local file access vulnerabilities
- Resource exhaustion (single user)

**Low** 🟢
- Timing attacks
- Security configuration issues
- Low-impact information leaks

## Security Best Practices

### For Users

1. **Keep Updated**: Always use the latest stable version
2. **Secure Configuration**:
   - Use strong API keys (`openssl rand -base64 32`)
   - Enable HTTPS for HTTP server mode
   - Restrict network access to localhost unless needed
3. **Credential Management**:
   - Never commit `.env` files with credentials
   - Use environment variables for sensitive data
   - Rotate Cloudflare API tokens regularly
4. **Authentication**: Enable OAuth 2.1 for multi-user deployments
5. **Monitoring**: Review logs for suspicious activity
6. **Backups**: Regularly backup your memory database

### For Contributors

1. **Dependency Security**:
   - Review dependency updates for known vulnerabilities
   - Use `pip-audit` to scan for security issues
   - Keep dependencies up to date
2. **Input Validation**:
   - Sanitize all user input
   - Use parameterized queries (no string concatenation)
   - Validate file uploads and document ingestion
3. **Authentication & Authorization**:
   - Use secure session management
   - Implement proper access controls
   - Follow OAuth 2.1 security best practices
4. **Sensitive Data**:
   - Never log API keys, tokens, or passwords
   - Encrypt sensitive data at rest (user responsibility)
   - Use secure random number generation
5. **Code Review**: All PRs must pass security review before merge

## Known Security Considerations

### SQLite-vec Backend
- **Local File Access**: Database file should have appropriate permissions (600)
- **Concurrent Access**: Use proper locking to prevent corruption
- **Backup Encryption**: User responsibility to encrypt backups

### Cloudflare Backend
- **API Token Security**: Tokens have full account access - guard carefully
- **Rate Limiting**: Cloudflare enforces rate limits (10k requests/min)
- **Data Residency**: Data stored in Cloudflare's network per your account settings

### Hybrid Backend
- **Synchronization**: Ensure secure sync between local and cloud storage
- **Credential Exposure**: Both SQLite and Cloudflare credentials needed

### Web Dashboard
- **HTTPS Recommended**: Use HTTPS in production environments
- **XSS Protection**: All user input is escaped before rendering
- **CSRF Protection**: Implement for state-changing operations
- **Session Security**: Enable secure cookies in production

### MCP Protocol
- **Local Access Only**: MCP server typically runs locally via stdin/stdout
- **Process Isolation**: Each client gets isolated server process
- **No Network Exposure**: By default, MCP mode has no network attack surface

## Security Updates

Security patches are released as:
- **Patch versions** (11.x.Y) for low/medium severity
- **Minor versions** (11.X.0) for high severity requiring API changes
- **Out-of-band releases** for critical vulnerabilities

Security advisories are published at:
- [GitHub Security Advisories](https://github.com/doobidoo/mcp-memory-service/security/advisories)
- [CHANGELOG.md](CHANGELOG.md) with `[SECURITY]` tag
- Release notes for affected versions

## Disclosure Policy

We follow **coordinated disclosure**:

1. Vulnerability reported privately
2. We confirm and develop a fix
3. Security advisory drafted (private)
4. Patch released with security note
5. Public disclosure 7 days after patch release
6. Reporter credited (if desired)

We appreciate security researchers following responsible disclosure practices and will acknowledge contributors in our security advisories.

## Security Hall of Fame

We recognize security researchers who help make MCP Memory Service more secure.
Reporters are credited by name in the advisory for the vulnerability they found -
see the [published advisories](https://github.com/doobidoo/mcp-memory-service/security/advisories)
for the full list.

## Contact

For security concerns that don't fit the above categories:
- **General Security Questions**: [GitHub issue](https://github.com/doobidoo/mcp-memory-service/issues)
- **Project Security**: See reporting instructions above

---

**Last Updated**: August 2026
**Policy Version**: 1.1
