"""TOON Format Documentation resource module.

This module provides comprehensive documentation for the TOON (Terser Object Notation)
format specification, optimized for LLM token efficiency and semantic memory storage.
"""

TOON_FORMAT_DOCUMENTATION = """# TOON Format Specification

TOON (Terser Object Notation) is a compact data format optimized for LLM token efficiency.

## Structure

TOON uses newline-separated records with pipe-delimited fields:
```
field1|field2|field3|...|fieldN
```

## Field Types

- **String**: Raw text (pipes escaped as \\p, newlines as \\n)
- **Number**: Raw numeric value
- **Boolean**: true/false
- **null**: Empty field or literal "null"
- **Array**: Comma-separated values
- **Object**: Nested pipe-delimited structure

## Memory Result Format

Each memory record contains these fields in order:
```
content|tags|metadata|created_at|updated_at|content_hash|similarity_score
```

Example:
```
Meeting notes from Q4 planning|planning,q4,2024|{"priority":"high","team":"eng"}|2024-11-18T10:30:00Z|2024-11-18T10:30:00Z|abc123|0.95
```

## Parsing Strategy

1. Split by newlines to get records
2. Split each record by unescaped pipes
3. Unescape special characters (\\p → |, \\n → newline)
4. Parse field values according to type
5. Reconstruct objects from nested structures

## Examples

### Single Memory

```
Python best practices for async|python,async,best-practices|{"source":"docs"}|2024-01-15T09:00:00Z|2024-01-15T09:00:00Z|hash1|0.92
```

### Multiple Memories

```
Docker deployment guide|docker,deployment|{}|2024-02-01T14:00:00Z|2024-02-01T14:00:00Z|hash2|0.88
API authentication flow|api,auth,security|{"reviewed":true}|2024-03-10T11:30:00Z|2024-03-10T11:30:00Z|hash3|0.85
```

### With Complex Metadata

```
Database migration checklist|database,migration|{"steps":["backup","test","deploy"],"criticality":"high"}|2024-04-05T16:45:00Z|2024-04-05T16:45:00Z|hash4|0.90
```

### Empty Results

```
No memories found matching your query.
```

## Common Pitfalls

1. **Unescaped Pipes**: Always escape | as \\p in content
2. **Newline Handling**: Escape \\n in multi-line content
3. **Metadata Format**: Must be valid JSON string
4. **Date Format**: Must use ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)
5. **Empty Fields**: Use empty string, not "null" literal (except for null metadata)

## References

- Spec: https://github.com/toon-format/spec
- Library: https://github.com/toon-format/toon-python
"""
