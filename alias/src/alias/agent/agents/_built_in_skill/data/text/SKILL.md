---
name: text-file
description: Guildlines for handling text files
type: text
---

# Text Files Handling Specifications

# Inspection
- Read first 5-10 lines to determine structure and encoding
- Always inspect text file structure before processing
- Identify delimiters, headers, or other structural elements

# Best Practices
- Handle encoding issues appropriately (UTF-8, ASCII, etc.)
- Validate file structure matches expectations
- Use appropriate parsing techniques based on file format
- Consider memory-efficient processing for very large files
- For large text files, read only the first few lines or process iteratively in chunks
- Strictly limit data volume during reads to prevent system crashes
