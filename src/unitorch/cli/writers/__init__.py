# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Optional
from unitorch.utils.io import GenericWriter
from unitorch.cli import config_defaults_init, register_writer


class WriterMixin:
    def from_pandas(self, df: pd.DataFrame):
        self.__pandas_dataframe__ = df

    def to_pandas(self):
        if hasattr(self, "__pandas_dataframe__"):
            return pd.DataFrame(self.__pandas_dataframe__)
        return pd.DataFrame()


class WriterOutputs:
    def __init__(self, process_outputs: pd.DataFrame):
        self.outputs = process_outputs
        self.schema = {}

    def to_pandas(self):
        return self.outputs


@register_writer("core/writer/jsonl")
class GeneralJsonlWriter(GenericWriter):
    """Write outputs to a JSONL file."""

    def __init__(
        self,
        output_file: str,
        nrows_per_sample: Optional[int] = None,
        header: Optional[bool] = None,
        columns: Optional[List[str]] = None,
    ):
        self.header = header
        self.columns = columns
        self.skip_n_samples = (
            0
            if nrows_per_sample is None or not os.path.exists(output_file)
            else sum(1 for _ in open(output_file)) // nrows_per_sample
        )
        mode = "a" if self.skip_n_samples > 0 else "w"
        self.output_file = open(output_file, mode, encoding="utf-8")

    @classmethod
    @config_defaults_init("core/writer/jsonl")
    def from_config(cls, config, **kwargs):
        pass

    def _write(self, outputs: "WriterOutputs"):
        dataframe = outputs.to_pandas()
        if self.columns is not None:
            cols = set(dataframe.columns)
            dataframe = dataframe[[c for c in self.columns if c in cols]]
        self.output_file.write(dataframe.to_json(orient="records", lines=True))
        self.output_file.flush()

    def process_start(self, outputs: "WriterOutputs"):
        self._write(outputs)

    def process_chunk(self, outputs: "WriterOutputs"):
        self._write(outputs)

    def process_end(self):
        self.output_file.close()


@register_writer("core/writer/csv")
class GeneralCsvWriter(GenericWriter):
    """Write outputs to a CSV/TSV file."""

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
        mode = "a" if self.skip_n_samples > 0 else "w"
        self.output_file = open(output_file, mode, encoding="utf-8")

    @classmethod
    @config_defaults_init("core/writer/csv")
    def from_config(cls, config, **kwargs):
        pass

    def _write(self, outputs: "WriterOutputs", include_header: bool = False):
        dataframe = outputs.to_pandas()
        if self.columns is not None:
            cols = set(dataframe.columns)
            dataframe = dataframe[[c for c in self.columns if c in cols]]
        self.output_file.write(
            dataframe.to_csv(
                index=False,
                sep=self.sep,
                quoting=self.quoting,
                header=include_header,
                escapechar=self.escapechar,
            )
        )
        self.output_file.flush()

    def process_start(self, outputs: "WriterOutputs"):
        self._write(outputs, include_header=bool(self.header) and self.skip_n_samples == 0)

    def process_chunk(self, outputs: "WriterOutputs"):
        self._write(outputs, include_header=False)

    def process_end(self):
        self.output_file.close()


@register_writer("core/writer/parquet")
class GeneralParquetWriter(GenericWriter):
    """Write outputs to a Parquet file."""

    def __init__(
        self,
        output_file: str,
        nrows_per_sample: Optional[int] = None,
        columns: Optional[List[str]] = None,
        schema: Optional[str] = None,
        compression: Optional[str] = "snappy",
    ):
        self.columns = columns
        self.skip_n_samples = 0
        self.output_file = output_file
        self.pq_writer = None
        self.pq_schema = None if schema is None else eval(schema)
        self.compression = compression

    @classmethod
    @config_defaults_init("core/writer/parquet")
    def from_config(cls, config, **kwargs):
        pass

    def process_start(self, outputs: "WriterOutputs"):
        dataframe = outputs.to_pandas()
        if self.columns is not None:
            cols = set(dataframe.columns)
            dataframe = dataframe[[c for c in self.columns if c in cols]]
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
        self.pq_writer.close()

    def process_chunk(self, outputs: "WriterOutputs"):
        assert self.pq_writer is not None
        dataframe = outputs.to_pandas()
        if self.columns is not None:
            cols = set(dataframe.columns)
            dataframe = dataframe[[c for c in self.columns if c in cols]]
        pa_table = pa.Table.from_pandas(dataframe, schema=self.pq_schema)
        self.pq_writer.write_table(pa_table)
