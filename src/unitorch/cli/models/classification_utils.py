# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from unitorch.cli import (
    add_default_section_for_init,
    register_process,
)
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models import TensorOutputs, TensorTargets, ACT2FN


@dataclass
class ClassificationOutputs(TensorOutputs, WriterMixin):
    """Outputs for classification models."""

    outputs: torch.Tensor


@dataclass
class EmbeddingOutputs(TensorOutputs, WriterMixin):
    """Outputs for embedding models, with up to four optional auxiliary embeddings."""

    embedding: torch.Tensor
    embedding1: Optional[torch.Tensor] = torch.empty(0)
    embedding2: Optional[torch.Tensor] = torch.empty(0)
    embedding3: Optional[torch.Tensor] = torch.empty(0)
    embedding4: Optional[torch.Tensor] = torch.empty(0)


@dataclass
class ClassificationTargets(TensorTargets):
    """Targets for classification models."""

    targets: torch.Tensor
    sample_weight: Optional[torch.Tensor] = torch.tensor(1.0)


class ClassificationProcessor:
    """Processor for classification models."""

    def __init__(
        self,
        act_fn: Optional[str] = None,
        return_scores: Optional[bool] = False,
        id2label: Optional[Dict[int, str]] = None,
    ):
        self.act_fn = ACT2FN.get(act_fn, None)
        self.return_scores = return_scores
        self.id2label = id2label

    @classmethod
    @add_default_section_for_init("core/process/classification")
    def from_core_configure(cls, config, **kwargs):
        pass

    @register_process("core/postprocess/classification/binary_score")
    def _binary_score(
        self,
        outputs: ClassificationOutputs,
    ):
        """Return positive-class score for binary classification."""
        assert outputs.outputs.dim() == 2

        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.outputs.shape[0]

        outputs = outputs.outputs.numpy()
        if self.act_fn is not None:
            outputs = self.act_fn(outputs)

        if outputs.ndim == 2:
            pscore = outputs[:, 1] if outputs.shape[-1] > 1 else outputs[:, 0]
            results["pscore"] = pscore.tolist()
        else:
            results["pscore"] = outputs.tolist()
        return WriterOutputs(results)

    @register_process("core/postprocess/classification/score")
    def _classifier_score(
        self,
        outputs: ClassificationOutputs,
    ):
        """Return argmax class and max score for multi-class classification."""
        assert outputs.outputs.dim() == 2

        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.outputs.shape[0]
        outputs = outputs.outputs.numpy()
        if self.act_fn is not None:
            outputs = self.act_fn(outputs)

        results["pscore"] = outputs.max(-1)
        results["pclass"] = outputs.argmax(-1)
        if self.id2label is not None:
            results["pclass"] = results["pclass"].map(self.id2label)

        if self.return_scores:
            results["scores"] = outputs.tolist()

        return WriterOutputs(results)

    @register_process("core/postprocess/classification/embedding")
    def _embedding(
        self,
        outputs: EmbeddingOutputs,
    ):
        """Postprocess embedding outputs, writing each embedding field to results."""
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.embedding.shape[0]

        embedding = outputs.embedding.numpy()
        if embedding.ndim > 2:
            embedding = embedding.reshape(embedding.size(0), -1)
        results["embedding"] = embedding.tolist()

        embedding1 = outputs.embedding1.numpy()
        if embedding1.size > 0:
            if embedding1.ndim > 2:
                embedding1 = embedding1.reshape(embedding1.size(0), -1)
            results["embedding1"] = embedding1.tolist()

        embedding2 = outputs.embedding2.numpy()
        if embedding2.size > 0:
            if embedding2.ndim > 2:
                embedding2 = embedding2.reshape(embedding2.size(0), -1)
            results["embedding2"] = embedding2.tolist()

        embedding3 = outputs.embedding3.numpy()
        if embedding3.size > 0:
            if embedding3.ndim > 2:
                embedding3 = embedding3.reshape(embedding3.size(0), -1)
            results["embedding3"] = embedding3.tolist()

        embedding4 = outputs.embedding4.numpy()
        if embedding4.size > 0:
            if embedding4.ndim > 2:
                embedding4 = embedding4.reshape(embedding4.size(0), -1)
            results["embedding4"] = embedding4.tolist()

        return WriterOutputs(results)

    @register_process("core/postprocess/classification/embedding/string")
    def _embedding_string(
        self,
        outputs: EmbeddingOutputs,
    ):
        """Postprocess embedding outputs as space-joined string representations."""
        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.embedding.shape[0]

        embedding = outputs.embedding.numpy()
        if embedding.ndim > 2:
            embedding = embedding.reshape(embedding.size(0), -1)
        results["embedding"] = embedding.tolist()
        results["embedding"] = results["embedding"].map(
            lambda x: " ".join([str(i) for i in x])
        )

        embedding1 = outputs.embedding1.numpy()
        if embedding1.size > 0:
            if embedding1.ndim > 2:
                embedding1 = embedding1.reshape(embedding1.size(0), -1)
            results["embedding1"] = embedding1.tolist()
            results["embedding1"] = results["embedding1"].map(
                lambda x: " ".join([str(i) for i in x])
            )

        embedding2 = outputs.embedding2.numpy()
        if embedding2.size > 0:
            if embedding2.ndim > 2:
                embedding2 = embedding2.reshape(embedding2.size(0), -1)
            results["embedding2"] = embedding2.tolist()
            results["embedding2"] = results["embedding2"].map(
                lambda x: " ".join([str(i) for i in x])
            )

        embedding3 = outputs.embedding3.numpy()
        if embedding3.size > 0:
            if embedding3.ndim > 2:
                embedding3 = embedding3.reshape(embedding3.size(0), -1)
            results["embedding3"] = embedding3.tolist()
            results["embedding3"] = results["embedding3"].map(
                lambda x: " ".join([str(i) for i in x])
            )

        embedding4 = outputs.embedding4.numpy()
        if embedding4.size > 0:
            if embedding4.ndim > 2:
                embedding4 = embedding4.reshape(embedding4.size(0), -1)
            results["embedding4"] = embedding4.tolist()
            results["embedding4"] = results["embedding4"].map(
                lambda x: " ".join([str(i) for i in x])
            )

        return WriterOutputs(results)
