# -*- coding: utf-8 -*-
import os
from agentscope.tool import ToolResponse
from alias.agent.agents.ds_agent_utils import get_prompt_from_file
from alias.agent.agents.data_source._typing import SourceType
from alias.agent.agents.ds_agent_utils.ds_config import (
    PROMPT_DS_BASE_PATH,
    MODEL_CONFIG_NAME,
    VL_MODEL_NAME,
)
from alias.agent.tools.improved_tools.multimodal_to_text import (
    DashScopeMultiModalTools,
)
from alias.runtime.alias_sandbox.alias_sandbox import AliasSandbox


def data_profile(
    sandbox: AliasSandbox,
    path: str,
    source_type: SourceType,
) -> ToolResponse:
    """
    Generates a detailed profile and summary for a specified data source using Large Language Models.

    This function acts as a dispatcher that:
    1. Initializes the DashScope toolset.
    2. Selects the appropriate prompt template based on the source type.
    3. Delegates the profiling task to the specific multimodal tool method (text or image).

    Args:
        sandbox (AliasSandbox): The sandbox environment instance where file operations occur.
        path (str): The location of the data source.
                    - For files: A file path (e.g., '/workspace/data.csv') or URL.
                    - For databases: A connection string (DSN).
        source_type (SourceType): The type of the data source (CSV, EXCEL, IMAGE, or RELATIONAL_DB).

    Returns:
        ToolResponse: An object containing the generated text profile of the data.

    Raises:
        ValueError: If the provided `source_type` is not supported.
    """

    # 1. Initialize the DashScope Multi-Modal Tool wrapper
    # Requires the API key from environment variables
    dash_scope_multimodal_tool_set = DashScopeMultiModalTools(
        sandbox=sandbox,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    )

    # Map source types to their corresponding prompt template filenames
    profile_prompts = {
        SourceType.CSV: "_profile_csv_prompt.md",
        SourceType.EXCEL: "_profile_xlsx_prompt.md",
        SourceType.IMAGE: "_profile_image_prompt.md",
        SourceType.RELATIONAL_DB: "_profile_relationdb_prompt.md",
    }

    # 2. Handle Structured/Text-based Data (CSV, Excel, Database)
    if source_type in [
        SourceType.CSV,
        SourceType.RELATIONAL_DB,
        SourceType.EXCEL,
    ]:
        # Load the specific prompt for structured data profiling
        summary_prompt = get_prompt_from_file(
            os.path.join(
                PROMPT_DS_BASE_PATH,
                profile_prompts[source_type],
            ),
            False,
        )
        # Delegate to the data-to-text generation method
        return dash_scope_multimodal_tool_set.dashscope_data_to_text(
            path=path,
            prompt=summary_prompt,
            model=MODEL_CONFIG_NAME,
            source_type=source_type,
        )
    # 3. Handle Visual Data (Images)
    elif source_type == SourceType.IMAGE:
        # Load the specific prompt for image analysis
        summary_prompt = get_prompt_from_file(
            os.path.join(
                PROMPT_DS_BASE_PATH,
                "_profile_image_prompt.md",
            ),
            False,
        )
        # Delegate to the image-to-text generation method
        return dash_scope_multimodal_tool_set.dashscope_image_to_text(
            image_url=path,
            prompt=summary_prompt,
            model=VL_MODEL_NAME,
        )
    # 4. Handle Unsupported Types
    else:
        raise ValueError(
            f"Unsupported source type for Data Profile: {source_type}",
        )
