---
name: csv-excel-file
description: Guildlines for handling CSV/Excel files
type:
  - csv
  - excel
---


# CSV/Excel Handling Specifications

# Inspection
- Use `pandas.head(n)` to view column names and sample data before any operation
- Inspect data structure to identify potential issues (missing headers, irregular formats)

# Preprocessing
- If messy spreadsheets are detected (containing "Unnamed: X", NaN, NaT entries, missing headers, or multiple data blocks):
  - Prioritize using `clean_messy_spreadsheet` tool to extract key information and output as JSON
  - Only fall back to manual pandas row/block parsing if the cleaning tool fails

# Querying
- Use `head()`, `nrows`, or sampling to fetch minimal data for well-structured files
- Strictly limit data volume during reads to prevent system crashes

# Best Practices
- Always inspect data structure before processing
- Handle encoding issues appropriately
- Validate data types after loading
