# Security Policy

## Reporting Security Issues

If you discover a security vulnerability in Roampal, please report it responsibly:

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues via:
- Email: roampal@protonmail.com
- GitHub Security Advisories (if repository is public)

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

## Security Response

We take security seriously and will respond to valid reports within:
- **24-48 hours**: Initial acknowledgment
- **7 days**: Assessment and timeline for fix
- **30 days**: Patch release (for critical issues)

## Security Considerations

**⚠️ IMPORTANT: Roampal is a single-user local application.**

### What We Protect

✅ **Local Data Privacy**
- All data stored locally in AppData (`C:\Users\<user>\AppData\Local\Roampal\data` on Windows)
- No cloud storage or external data transmission
- 100% offline operation after initial model download

✅ **CORS Protection**
- Backend restricted to `localhost` origins only
- Prevents unauthorized remote access

✅ **Input Validation**
- Message length limits (10,000 characters)
- Control character sanitization
- File upload restrictions (10MB max, .txt/.md only)

✅ **Path Traversal Protection**
- UUID validation for file operations
- Restricted file system access

### What We Don't Protect

❌ **Multi-User Access**
- Not designed for shared hosting
- No authentication/authorization system
- Single-user design by intention

❌ **Network Security**
- Assumes localhost environment
- No encryption for local API calls
- Not designed for internet-facing deployment

## Best Practices for Users

1. **Local Use Only**
   - Do not expose ports 8000 (backend), 11434 (Ollama), or 1234 (LM Studio) to the internet
   - Do not deploy on shared hosting
   - Use firewall to restrict access to localhost only

2. **Keep Dependencies Updated**
   - Update Roampal when new versions are released
   - Update LLM models (Ollama/LM Studio) periodically
   - Monitor security advisories

3. **Data Backup**
   - Backup AppData directory regularly (`C:\Users\<user>\AppData\Local\Roampal\` on Windows)
   - Store backups securely offline
   - Test restore procedures

## Scope

This security policy applies to:
- Roampal core application
- Official releases and builds
- Open source repository

Out of scope:
- Third-party forks or modifications
- User-modified configurations
- Ollama security (handled by Ollama project)
- LM Studio security (handled by LM Studio)
- Downloaded LLM model security (handled by model providers)
