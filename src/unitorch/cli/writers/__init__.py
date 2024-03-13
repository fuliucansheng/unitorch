# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.utils.io import GenericWriter
from unitorch.cli import add_default_section_for_init, register_writer


class WriterMixin:
    def from_pandas(self, df: pd.DataFrame):
        self.__pandas_dataframe__ = df

    def to_pandas(self):
        if hasattr(self, "__pandas_dataframe__"):
            return pd.DataFrame(self.__pandas_dataframe__)
        return pd.DataFrame()


class WriterOutputs:
    def __init__(
        self,
        process_outputs: pd.DataFrame,
    ):
        self.outputs = process_outputs
        self.schema = dict()

    def to_pandas(self):
        return self.outputs


@register_writer("core/writer/jsonl")
class GeneralJsonlWriter(GenericWriter):
    """Class for writing data in JSONL format."""

    def __init__(
        self,
        output_file: str,
        nrows_per_sample: Optional[int] = None,
        header: Optional[bool] = None,
        columns: Optional[List[str]] = None,
    ):
        """
        Initialize GeneralJsonlWriter.

        Args:
            output_file (str): The path to the output file.
            nrows_per_sample (int, optional): The number of rows per sample. Defaults to None.
            header (bool, optional): Whether to include a header in the output file. Defaults to None.
            columns (List[str], optional): The list of columns to include in the output file. Defaults to None.
        """
        self.header = header
        self.columns = columns
        self.skip_n_samples = (
            0
            if nrows_per_sample is None or not os.path.exists(output_file)
            else sum(1 for _ in open(output_file)) // nrows_per_sample
        )
        if self.skip_n_samples == 0:
            self.output_file = open(output_file, "w", encoding="utf-8")
        else:
            self.output_file = open(output_file, "a", encoding="utf-8")

    @classmethod
    @add_default_section_for_init("core/writer/jsonl")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of GeneralJsonlWriter from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            GeneralJsonlWriter: An instance of GeneralJsonlWriter.
        """
        pass

    def process_start(self, outputs: WriterOutputs):
        """
        Process the start of the writing process.

        Args:
            outputs (WriterOutputs): The writer outputs.
        """
        dataframe = outputs.to_pandas()
        if self.columns is not None:
            columns = set(dataframe.columns)
            dataframe = dataframe[[h for h in self.columns if h in columns]]
        string = dataframe.to_json(orient="records", lines=True)
        self.output_file.write(string)
        self.output_file.flush()

    def process_end(self):
        """Process the end of the writing process."""
        self.output_file.close()

    def process_chunk(self, outputs: WriterOutputs):
        """
        Process a chunk of data during the writing process.

        Args:
            outputs (WriterOutputs): The writer outputs.
        """
        dataframe = outputs.to_pandas()
        if self.columns is not None:
            columns = set(dataframe.columns)
            dataframe = dataframe[[h for h in self.columns if h in columns]]
        string = dataframe.to_json(orient="records", lines=True)
        self.output_file.write(string)
        self.output_file.flush()


@register_writer("core/writer/csv")
class GeneralCsvWriter(GenericWriter):
    """Class for writing data in CSV format."""

    def __init__(
        self,
        output_file: str,
        nrows_per_sample: Optional[int] = None,
        header: Optional[bool] = None,
        columns: Optional[List[str]] = None,
        sep: Optional[str] = "\t",
        quoting: Optional[int] = 3,
        escapechar: Optional[str] = None,
    ):
        """
        Initialize GeneralCsvWriter.

        Args:
            output_file (str): The path to the output file.
            nrows_per_sample (int, optional): The number of rows per sample. Defaults to None.
            header (bool, optional): Whether to include a header in the output file. Defaults to None.
            columns (List[str], optional): The list of columns to include in the output file. Defaults to None.
            sep (str, optional): The separator for the CSV file. Defaults to "\t".
            quoting (int, optional): The quoting style for the CSV file. Defaults to 3.
            escapechar (str, optional): The escape character for the CSV file. Defaults to None.
        """
        self.header = header
        self.columns = columns
        self.sep = sep
        self.quoting = quoting
        self.escapechar = escapechar
        has_header = int(header is True)
        self.skip_n_samples = (
            0
            if nrows_per_sample is None or not os.path.exists(output_file)
            else (sum(1 for _ in open(output_file)) - has_header) // nrows_per_sample
        )
        if self.skip_n_samples == 0:
            self.output_file = open(output_file, "w", encoding="utf-8")
        else:
            self.output_file = open(output_file, "a", encoding="utf-8")

    @classmethod
    @add_default_section_for_init("core/writer/csv")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of GeneralCsvWriter from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            GeneralCsvWriter: An instance of GeneralCsvWriter.
        """
        pass

    def process_start(self, outputs: WriterOutputs):
        """
        Process the start of the writing process.

        Args:
            outputs (WriterOutputs): The writer outputs.
        """
        dataframe = outputs.to_pandas()
        if self.columns is not None:
            columns = set(dataframe.columns)
            dataframe = dataframe[[h for h in self.columns if h in columns]]
        string = dataframe.to_csv(
            index=False,
            sep=self.sep,
            quoting=self.quoting,
            header=self.header and self.skip_n_samples == 0,
            escapechar=self.escapechar,
        )
        self.output_file.write(string)
        self.output_file.flush()

    def process_end(self):
        """Process the end of the writing process."""
        self.output_file.close()

    def process_chunk(self, outputs: WriterOutputs):
        """
        Process a chunk of data during the writing process.

        Args:
            outputs (WriterOutputs): The writer outputs.
        """
        dataframe = outputs.to_pandas()
        if self.columns is not None:
            columns = set(dataframe.columns)
            dataframe = dataframe[[h for h in self.columns if h in columns]]
        string = dataframe.to_csv(
            index=False,
            sep=self.sep,
            quoting=self.quoting,
            header=False,
            escapechar=self.escapechar,
        )
        self.output_file.write(string)
        self.output_file.flush()


@register_writer("core/writer/parquet")
class GeneralParquetWriter(GenericWriter):
    def __init__(
        self,
        output_file: str,
        nrows_per_sample: Optional[int] = None,
        columns: Optional[List[str]] = None,
        schema: Optional[str] = None,
        compression: Optional[str] = "snappy",
    ):
        """
        Initialize GeneralParquetWriter.

        Args:
            output_file (str): The path to the output file.
            nrows_per_sample (int, optional): The number of rows per sample. Defaults to None.
            columns (List[str], optional): The list of columns to include in the output file. Defaults to None.
            schema (str, optional): The Parquet schema in string format. Defaults to None.
            compression (str, optional): The compression algorithm to use. Defaults to "snappy".
        """
        self.columns = columns
        self.skip_n_samples = 0
        self.output_file = output_file
        self.pq_writer = None
        self.pq_schema = None if schema is None else eval(schema)
        self.compression = compression

    @classmethod
    @add_default_section_for_init("core/writer/parquet")
    def from_core_configure(cls, config, **kwargs):
        """
        Create an instance of GeneralParquetWriter from a core configuration.

        Args:
            config: The core configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            GeneralParquetWriter: An instance of GeneralParquetWriter.
        """
        pass

    def process_start(self, outputs: WriterOutputs):
        """
        Process the start of the writing process.

        Args:
            outputs (WriterOutputs): The writer outputs.
        """
        dataframe = outputs.to_pandas()
        if self.columns is not None:
            columns = set(dataframe.columns)
            dataframe = dataframe[[h for h in self.columns if h in columns]]
        pa_table = pa.Table.from_pandas(dataframe)
        if self.pq_schema is None:
            self.pq_schema = pa_table.schema
        pa_table = pa.Table.from_pandas(dataframe, schema=self.pq_schema)
        self.pq_writer = pq.ParquetWriter(
            self.output_file,
            self.pq_schema,
            version="1.0",
            use_dictionary=False,
            flavor="spark",
            compression=self.compression,
            use_compliant_nested_type=True,
        )
        self.pq_writer.write_table(pa_table)

    def process_end(self):
        """Process the end of the writing process."""
        self.pq_writer.close()

    def process_chunk(self, outputs: WriterOutputs):
        """
        Process a chunk of data during the writing process.

        Args:
            outputs (WriterOutputs): The writer outputs.
        """
        assert self.pq_writer is not None
        dataframe = outputs.to_pandas()
        if self.columns is not None:
            columns = set(dataframe.columns)
            dataframe = dataframe[[h for h in self.columns if h in columns]]
        pa_table = pa.Table.from_pandas(dataframe, schema=self.pq_schema)
        self.pq_writer.write_table(pa_table)


# more writers
