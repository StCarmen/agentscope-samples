# -*- coding: utf-8 -*-
from io import BytesIO
import os
import base64
import json
import tempfile
from alias.agent.agents.data_source._typing import SourceType
import requests
import dashscope
import pandas as pd
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock
from sqlalchemy import inspect, text, create_engine

from alias.agent.tools.sandbox_util import (
    get_workspace_file,
)
from alias.runtime.alias_sandbox import AliasSandbox


def _get_binary_buffer(
    sandbox: AliasSandbox,
    audio_file_url: str,
):
    if audio_file_url.startswith(("http://", "https://")):
        response = requests.get(audio_file_url)
        response.raise_for_status()
        audio_buffer = BytesIO(response.content)
    else:
        audio_buffer = BytesIO(
            base64.b64decode(get_workspace_file(sandbox, audio_file_url)),
        )
    return audio_buffer


def _tool_clean_json(raw_response: str):
    cleaned_response = raw_response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[len("```json") :].lstrip()
    if cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[len("```") :].lstrip()
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3].rstrip()
    return json.loads(cleaned_response)


class DashScopeProfile:
    """
    A set of tools based on DashScope models
    to profile multimodal data like csv, excel, database, image, etc.
    """

    def __init__(
        self,
        sandbox: AliasSandbox,
        dashscope_api_key: str,
    ):
        self.sandbox = sandbox
        self.api_key = dashscope_api_key

    def dashscope_audio_to_text_profile(
        self,
        audio_file_url: str,
        language: str = "en",
    ) -> ToolResponse:
        """
        Convert an audio file to text using DashScope's transcription service.

        Args:
            audio_file_url (`str`):
                The file path or URL to the audio file that needs to be
                transcribed.
            language (`str`, defaults to `"en"`):
                The language of the input audio in
                `ISO-639-1 format \
                <https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes>`_
                (e.g., "en", "zh", "fr"). Improves accuracy and latency.

        Returns:
            `ToolResponse`:
                A ToolResponse containing the generated content
                (ImageBlock/TextBlock/AudioBlock) or error information if the
                operation failed.
        """

        try:
            # Handle different types of audio file URLs
            if audio_file_url.startswith(("http://", "https://")):
                # For web URLs, use the URL directly
                audio_source = audio_file_url
            else:
                # For local files, save to a temporary file
                audio_buffer = _get_binary_buffer(
                    sandbox=self.sandbox,
                    audio_file_url=audio_file_url,
                )

                # Create a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(audio_file_url)[1],
                ) as temp_file:
                    temp_file.write(audio_buffer.getvalue())
                    audio_source = temp_file.name

            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "text": "Transcript the content in the audio "
                            "to text.",
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "audio": audio_source,
                        },
                    ],
                },
            ]

            response = dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model="qwen3-asr-flash",
                messages=messages,
                asr_options={
                    "enable_lid": True,
                    "language": language,
                },
            )

            # Clean up temporary file if created
            if not audio_file_url.startswith(("http://", "https://")):
                try:
                    os.unlink(audio_source)
                except Exception as _:  # noqa: F841
                    pass

            content = response.output["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = content[0]["text"]
            # TODO: add support for audio
            if content is not None:
                return ToolResponse(
                    [
                        TextBlock(
                            type="text",
                            text=content,
                        ),
                    ],
                )
            else:
                return ToolResponse(
                    [
                        TextBlock(
                            type="text",
                            text="Error: Failed to generate text from audio",
                        ),
                    ],
                )
        except Exception as _:  # noqa: F841
            import traceback

            return ToolResponse(
                [
                    TextBlock(
                        type="text",
                        text="Error: Failed to transcribe audio: "
                        f"{traceback.format_exc()}",
                    ),
                ],
            )

    def dashscope_image_to_text_profile(
        self,
        image_url: str,
        prompt: str = "Describe the image",
        model: str = "qwen-vl-plus",
    ) -> ToolResponse:
        """Generate text based on the given images.

        Args:
            image_url (`str`):
                The url of single or multiple images.
            prompt (`str`, defaults to 'Describe the image' ):
                The text prompt.
            model (`str`, defaults to 'qwen-vl-plus'):
                The model to use in DashScope MultiModal API.

        Returns:
            `ToolResponse`:
                A ToolResponse containing the generated content
                (ImageBlock/TextBlock/AudioBlock) or error information if the
                operation failed.
        """

        # Handle different types of image file URLs
        if image_url.startswith(("http://", "https://")):
            # For web URLs, use the URL directly
            image_source = image_url
        else:
            # For local files, save to a temporary file
            image_buffer = _get_binary_buffer(
                self.sandbox,
                image_url,
            )

            suffix = os.path.splitext(image_url)[1].lower() or ".png"
            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:
                temp_file.write(image_buffer.getvalue())
                image_source = temp_file.name

        contents = []
        # Convert image paths according to the model requirements
        contents.append(
            {
                "image": image_source,
            },
        )
        # append text
        contents.append({"text": prompt})

        # currently only support one round of conversation
        # if multiple rounds of conversation are needed,
        # it would be better to implement an Agent class
        sys_message = {
            "role": "system",
            "content": [{"text": "You are a helpful assistant."}],
        }
        user_message = {
            "role": "user",
            "content": contents,
        }
        messages = [sys_message, user_message]
        try:
            response = dashscope.MultiModalConversation.call(
                model=model,
                messages=messages,
                api_key=self.api_key,
            )
            content = response.output["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = content[0]["text"]
            content = _tool_clean_json(content)
            profile = {
                "name": os.path.basename(image_url),
                "description": content["description"],
                "details": content["details"],
            }
            if content is not None:
                return ToolResponse(
                    [
                        TextBlock(
                            type="text",
                            text=profile,
                        ),
                    ],
                )
            else:
                return ToolResponse(
                    [
                        TextBlock(
                            type="text",
                            text="Error: Failed to generate text",
                        ),
                    ],
                )
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            return ToolResponse(
                [
                    TextBlock(
                        type="text",
                        text=f"Failed to generate text: {str(e)}",
                    ),
                ],
            )

    def _extract_schema_from_original_file(self, file_path: str) -> dict:
        """
        Extracts schema information (columns, data types, samples) from a file.
        Supports CSV and Excel formats.

        Args:
            file_path (str): The path or URL to the CSV or Excel file.

        Returns:
            dict: A dictionary containing the schema of the file
            (tables, columns, etc.),
        """

        def copy_file_from_sandbox(file_path: str, suffix: str) -> str:
            """
            Copies a file from the sandbox environment
            or a URL to a local temporary file.

            Args:
                path (str): Source path or URL.
                ext (str): File extension (e.g., '.csv').

            Returns:
                str: The path to the local temporary file.
            """
            # Handle different types of file URLs
            if file_path.startswith(("http://", "https://")):
                # For web URLs, use the URL directly
                file_source = file_path
            else:
                # For local files, save to a temporary file
                file_buffer = _get_binary_buffer(
                    self.sandbox,
                    file_path,
                )
                # Create a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp_file:
                    temp_file.write(file_buffer.getvalue())
                    file_source = temp_file.name
            return file_source

        def extract_schema_from_table(df: pd.DataFrame, df_name: str) -> dict:
            """
            Analyzes a single DataFrame to extract column metadata and samples.

            Args:
                df (pd.DataFrame): The dataframe to analyze.
                table_name (str): Name of the table (or sheet/filename).

            Returns:
                dict: Schema metadata for the table.
            """
            col_list = []
            for col in df.columns:
                dtype_name = str(df[col].dtype).upper()
                # Get random samples to help LLM understand the data content
                # sample(frac=1): shuffle the data
                # head(n_samples): get the first n_samples,
                # if less than n_samples, retrieved here without any errors.
                candidates = (
                    df[col]
                    .drop_duplicates()
                    .sample(frac=1, random_state=42)
                    .head(5)
                    .astype(str)
                    .tolist()
                )
                # Limit the size not to exceed 1000 characters.
                # TODO: dynamic size control? 1000 is too small?
                samples = []
                length = 0
                for s in candidates:
                    if (length := length + len(s)) <= 1000:
                        samples.append(s)
                col_list.append(
                    {
                        "column name": col,
                        "data type": dtype_name,
                        "data samples": samples,
                    },
                )
            # Create a CSV snippet of the first few rows
            raw_data_snippet = df.head(5).to_csv(index=True)

            table_schema = {
                "name": df_name,
                "raw_data_snippet": raw_data_snippet,
                # Note: Row count logic might need optimization for large files
                # TODO: how to get the row count more efficiently, openpyxl.
                "row_count": len(df) if len(df) < 100 else None,
                "col_count": len(df.columns),
                "columns": col_list,
            }
            return table_schema

        def extract_csv(file_source: str, file_name):
            """
            Handlers schema extraction for CSV files.
            Treats the CSV as a single table.
            """
            import polars as pl

            # Use Polars for efficient row counting on large files
            df = pl.scan_csv(file_source, ignore_errors=True)
            row_count = df.select(pl.len()).collect().item()
            # Read a subset with Pandas for detailed schema analysis
            df = pd.read_csv(file_source, nrows=100).convert_dtypes()
            schema = extract_schema_from_table(df, file_name)
            schema["row_count"] = row_count
            return schema

        def extract_excel(file_source: str, file_name):
            """
            Handles schema extraction for Excel files.
            Treats each sheet as a separate table.
            """
            # TODO: handle irregular Excel structures (e.g., merged headers)
            excel_file = pd.ExcelFile(file_source)
            table_schemas = []
            schema["name"] = file_name
            for sheet_name in excel_file.sheet_names:
                # TODO: use openpyxl to read excel to avoid irregular excel.
                # Read a subset of each sheet
                df = pd.read_excel(
                    file_source,
                    sheet_name=sheet_name,
                    nrows=100,
                ).convert_dtypes()
                table_schema = extract_schema_from_table(df, sheet_name)
                table_schemas.append(table_schema)
            schema["tables"] = table_schemas
            return schema

        suffix = os.path.splitext(file_path)[1].lower()
        assert suffix in [".csv", ".xlsx", ".xls", ".xlsm"]

        # Initialize schema
        schema = {}
        file_name = os.path.basename(file_path)
        file_source = copy_file_from_sandbox(file_path, suffix)
        try:
            if suffix.lower() == ".csv":
                schema = extract_csv(file_source, file_name)
            elif suffix.lower() in [".xlsx", ".xls", ".xlsm"]:
                schema = extract_excel(file_source, file_name)
            return schema
        except Exception:
            import traceback

            print(f"Error processing file: {traceback.format_exc()}")
            return {}

    def _extract_schema_from_relational_database(self, dsn: str) -> dict:
        """
        Extracts metadata (schema) for all tables in a relational db.

        Args:
            dsn (str): The Database Source Name (connection string).
            eg. postgresql://user:XB6FqqgHk26h@47.238.87.81:49166/dacomp_001
        Returns:
            dict: A JSON-compatible dictionary containing database metadata
                  (table names, columns, row counts, samples).
        """

        options = {
            "isolation_level": "AUTOCOMMIT",
            # Test conns before use (handles MySQL 8hr timeout, network drops)
            "pool_pre_ping": True,
            # Keep minimal conns (MCP typically handles 1 request at a time)
            "pool_size": 1,
            # Allow temporary burst capacity for edge cases
            "max_overflow": 2,
            # Force refresh conns older than 1hr (under MySQL's 8hr default)
            "pool_recycle": 3600,
        }
        engine = create_engine(dsn, **options)
        try:
            connection = engine.connect()
        except Exception as e:
            print(f"Connection to {dsn} failed: {e}")
            raise Exception(f"Failed to connect to database: {e}")

        # Use DSN as the db identifier (can parsed cleaner)
        database_name = dsn
        inspector = inspect(connection)
        table_names = inspector.get_table_names()

        tables_data = []
        for table_name in table_names:
            try:
                # 1. Get column information
                columns = inspector.get_columns(table_name)
                col_count = len(columns)

                # 2. Get row count
                row_count_result = connection.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}"),
                ).fetchone()
                row_count = row_count_result[0] if row_count_result else 0

                # 3. Get raw data snippet (first 5 rows)
                raw_data_snippet = ""
                try:
                    result = connection.execute(
                        text(f"SELECT * FROM {table_name} LIMIT 5"),
                    )
                    rows = result.fetchall()
                    if rows:
                        column_names = [col["name"] for col in columns]
                        lines = []
                        # Add header
                        lines.append(", ".join(column_names))
                        # Add data rows
                        for row in rows:
                            row_values = []
                            for value in row:
                                if value is None:
                                    row_values.append("NULL")
                                else:
                                    # Escape commas and newlines
                                    val_str = str(value)
                                    if "," in val_str or "\n" in val_str:
                                        val_str = f'"{val_str}"'
                                    row_values.append(val_str)
                            lines.append(", ".join(row_values))
                        raw_data_snippet = "\n".join(lines)
                except Exception as e:
                    print(f"Error fetching {table_name} data: {str(e)}")
                    raw_data_snippet = None
                # 4. detailed column info (types and samples)
                column_details = []
                if rows:
                    for i, col in enumerate(columns):
                        col_name = col["name"]
                        col_type = str(col["type"])
                        # Extract samples for this column from the fetched rows
                        sample_values = []
                        for row in rows:
                            if i < len(row):
                                val = row[i]
                                sample_values.append(
                                    str(val) if val is not None else "NULL",
                                )

                        column_details.append(
                            {
                                "column name": col_name,
                                "data type": col_type,
                                "data sample": sample_values[:3],
                            },
                        )

                table_info = {
                    "name": table_name,
                    "row_count": row_count,
                    "col_count": col_count,
                    "raw_data_snippet": raw_data_snippet,
                    "columns": column_details,
                }

                tables_data.append(table_info)

            except Exception as e:
                # If one table fails, log it and continue to the next
                print(f"Error processing {table_name}: {str(e)}")
                return {}
        # Contruct the final schema
        schema = {
            "name": database_name,
            "tables": tables_data,
        }
        return schema

    def _llm_profile_by_content(self, content: str, model: str) -> dict:
        """
        Uses an LLM to generate a profile based on the provided content.

        Args:
            content (str): The text content (e.g., schema description).
            model (str): The model name to use for generation.

        Returns:
            dict: The profiled metadata in dict format, parsed from LLM resp.
        """
        try:
            sys_message = {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
            user_message = {
                "role": "user",
                "content": content,
            }
            messages = [sys_message, user_message]
            # dashscope.MultiModalConversation.call
            response = dashscope.Generation.call(
                model=model,
                messages=messages,
                api_key=self.api_key,
            )
            response = response.output["choices"][0]["message"]["content"]
            # Clean and parse the JSON response from the LLM
            cleaned_response = _tool_clean_json(response)
            if isinstance(cleaned_response, list):
                # Handle cases where the resp might be a list wrapping a dict
                cleaned_response = cleaned_response[0]["text"]
            return cleaned_response

        except Exception:
            import traceback

            print(traceback.format_exc())
            # Consider returning None or an empty dict on failure
            return {}

    def dashscope_data_to_text_profile(
        self,
        path: str,
        prompt: str = "Describe the csv",
        model: str = "qwen3-max",
        source_type: SourceType = SourceType.CSV,
    ) -> ToolResponse:
        """
        Generates a textual description/profile for
        different data source (File or Database) using DashScope.

        This function orchestrates the process of:
        1. Extracting schema from the source (CSV, Excel, or DB).
        2. Constructing a prompt with the schema.
        3. Invoking the LLM to generate a profile.
        4. Wrapping the result into a standardized format.

        Args:
            path (str): The file path or Database DSN.
            prompt (str): The prompt, should contain `{schema}` placeholder.
            model (str): The DashScope model to use.
            source_type (SourceType): The type, (CSV, EXCEL, RELATIONAL_DB).

        Returns:
            ToolResponse: Contains the generated data profile description.
        """

        def wrap_repsonse_into_schema(schema: dict, response: dict) -> dict:
            """
            Merges the original schema with the LLM-generated response.
            """
            new_schema = {}
            new_schema["name"] = schema["name"]
            new_schema["description"] = response["description"]
            #  # For flat files like CSV, they contain columns
            if "columns" in schema:
                new_schema["columns"] = schema["columns"]
            # # For multi-table sources like Excel/Database,
            # they contain tables. Each table contains columns and description
            if "tables" in schema and "tables" in response:
                new_schema["tables"] = []
                for i, table in enumerate(schema["tables"]):
                    # Ensure alignment between schema tables and resp tables
                    # TODO: It matches by order, by name would be more robust.
                    assert response["tables"][i]["name"] == table["name"]
                    new_table = {}
                    new_table["name"] = table["name"]
                    new_table["description"] = response["tables"][i][
                        "description"
                    ]
                    new_table["columns"] = table["columns"]
                    new_schema["tables"].append(new_table)
            return new_schema

        # 1. Extract Schema
        if source_type in [SourceType.CSV, SourceType.EXCEL]:
            schema = self._extract_schema_from_original_file(path)
        elif source_type == SourceType.RELATIONAL_DB:
            schema = self._extract_schema_from_relational_database(path)

        # 2. Prepare Prompt
        content = prompt.format(schema=schema)
        # 3. LLM Generation
        # TODO: verify whether the schema is valid,
        # like headerless csv or irregular excel, using LLM.
        # response = self._llm_verify_by_schema(content, model)
        response = self._llm_profile_by_content(content, model)
        # 4. Merge and Return
        new_schema = wrap_repsonse_into_schema(schema, response)

        return ToolResponse([TextBlock(type="text", text=new_schema)])
