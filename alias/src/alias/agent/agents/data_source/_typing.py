# -*- coding: utf-8 -*-

from enum import Enum


class SourceAccessType(str, Enum):
    """Simple source access type classification"""

    LOCAL_FILE = "local_file"
    MCP_TOOL = "mcp_tool"
    HTTP_URL = "http_url"  # TBD

    def __str__(self):
        return self.value


class SourceType(str, Enum):
    """Simple source type classification"""

    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    PARQUET = "parquet"

    # Database sources
    RELATIONAL_DB = "relational_db"
    DOCUMENT_DB = "document_db"
    GRAPH_DB = "graph_db"

    # Other sources
    API = "api"

    UNKNOWN = "unknown"

    def __str__(self):
        return self.value

    @staticmethod
    def is_valid_source_type(value: str) -> bool:
        try:
            SourceType(value)
            return True
        except ValueError:
            return False


# Define mapping between SourceType and SourceAccessType
SOURCE_TYPE_TO_ACCESS_TYPE = {
    # File types -> LOCAL_FILE
    SourceType.CSV: SourceAccessType.LOCAL_FILE,
    SourceType.JSON: SourceAccessType.LOCAL_FILE,
    SourceType.EXCEL: SourceAccessType.LOCAL_FILE,
    SourceType.TEXT: SourceAccessType.LOCAL_FILE,
    SourceType.IMAGE: SourceAccessType.LOCAL_FILE,
    SourceType.PDF: SourceAccessType.LOCAL_FILE,
    SourceType.PARQUET: SourceAccessType.LOCAL_FILE,
    # Database types -> MCP_TOOL
    SourceType.RELATIONAL_DB: SourceAccessType.MCP_TOOL,
    SourceType.DOCUMENT_DB: SourceAccessType.MCP_TOOL,
    SourceType.GRAPH_DB: SourceAccessType.MCP_TOOL,
    # API type -> HTTP_URL
    SourceType.API: SourceAccessType.HTTP_URL,
    # Unknown type -> depends on endpoint
    SourceType.UNKNOWN: None,
}
