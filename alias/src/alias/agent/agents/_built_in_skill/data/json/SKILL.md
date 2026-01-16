---
name: json-file
description: Guildlines for handling json files
type: json
---

# JSON Handling Specifications

# Inspection
- Inspect from outer to inner layers progressively
- Always verify JSON structure before processing
- Identify nested structures and data types

# Best Practices
- Validate JSON format before processing
- Handle malformed JSON gracefully
- Use appropriate libraries for JSON parsing and manipulation
- Consider memory usage when working with deeply nested or large JSON structures
- Extract only necessary data rather than loading entire JSON objects when possible
- For large JSON files, consider processing in chunks or using streaming parsers
- Strictly limit data volume during reads to prevent system crashes
