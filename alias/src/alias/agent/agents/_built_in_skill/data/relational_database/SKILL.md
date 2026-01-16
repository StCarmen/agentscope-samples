---
name: database
description: Guildlines for handling databases
type: relational_db
---

# Database Handling Specifications

# Inspection
- Use `inspect_in_python` tool to inspect the database structure
- Preview first 5-10 rows before any operation
- Never use `run_ipython_cell` to operate database

# Querying
- Use `execute_sql_in_python` tool to execute SQL queries
- If a column name contains uppercase letters, MUST quote it with double quotes
- Before retrieving all data, use LIMIT 1 to infer schema and COUNT(*) to estimate row count
- Strictly limit data volume during queries to prevent system crashes
- Before retrieving all data, use LIMIT 1 to infer the schema and COUNT(*) to estimate the number of rows—this helps evaluate query efficiency and correctness upfront

# Best Practices
- Always verify database structure before querying
- Use appropriate sampling techniques for large datasets
- Optimize queries for efficiency based on schema inspection
