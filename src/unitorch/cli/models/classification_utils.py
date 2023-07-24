# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import pyarrow as pa
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli import WriterMixin, WriterOutputs
from unitorch.cli.models import TensorsOutputs, TensorsTargets, ACT2FN


@dataclass
class ClassificationOutputs(TensorsOutputs, WriterMixin):
    """
    Outputs for classification models.

    Args:
        outputs (torch.Tensor): Output tensor containing the classification results.
    """

    outputs: torch.Tensor


@dataclass
class EmbeddingOutputs(TensorsOutputs, WriterMixin):
    """
    Outputs for embedding models.

    Args:
        embedding (torch.Tensor): The embedding tensor.
        embedding1 (Optional[torch.Tensor]): Additional embedding tensor 1. Defaults to an empty tensor.
        embedding2 (Optional[torch.Tensor]): Additional embedding tensor 2. Defaults to an empty tensor.
        embedding3 (Optional[torch.Tensor]): Additional embedding tensor 3. Defaults to an empty tensor.
        embedding4 (Optional[torch.Tensor]): Additional embedding tensor 4. Defaults to an empty tensor.
    """

    embedding: torch.Tensor
    embedding1: Optional[torch.Tensor] = torch.empty(0)
    embedding2: Optional[torch.Tensor] = torch.empty(0)
    embedding3: Optional[torch.Tensor] = torch.empty(0)
    embedding4: Optional[torch.Tensor] = torch.empty(0)


@dataclass
class ClassificationTargets(TensorsTargets):
    """
    Targets for classification models.

    Args:
        targets (torch.Tensor): The target tensor.
        sample_weight (Optional[torch.Tensor]): The weight associated with each target. Defaults to a tensor with a value of 1.0.
    """

    targets: torch.Tensor
    sample_weight: Optional[torch.Tensor] = torch.tensor(1.0)


class ClassificationProcessor:
    """Processor for classification models."""

    def __init__(
        self,
        act_fn: Optional[str] = None,
        return_scores: Optional[bool] = False,
    ):
        """
        Initialize the ClassificationProcessor.

        Args:
            act_fn (Optional[str]): Activation function to apply to the model outputs.
            return_scores (Optional[bool]): Whether to return the scores in addition to the predictions.
        """
        self.act_fn = ACT2FN.get(act_fn, None)
        self.return_scores = return_scores

    @classmethod
    @add_default_section_for_init("core/process/classification")
    def from_core_configure(cls, config, **kwargs):
        """
        Create a ClassificationProcessor instance from core configuration.

        Args:
            config: Configuration object.
            **kwargs: Additional keyword arguments.

        Returns:
            ClassificationProcessor: The initialized ClassificationProcessor instance.
        """
        pass

    @register_process("core/postprocess/classification/binary_score")
    def _binary_score(
        self,
        outputs: ClassificationOutputs,
    ):
        """
        Postprocess the classification outputs for binary classification with scores.

        Args:
            outputs (ClassificationOutputs): Outputs from the classification model.

        Returns:
            WriterOutputs: Processed outputs with scores.
        """
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
        """
        Postprocess the classification outputs for multi-classes classification with scores.

        Args:
            outputs (ClassificationOutputs): Outputs from the classification model.

        Returns:
            WriterOutputs: Processed outputs with scores and predicted classes.
        """
        assert outputs.outputs.dim() == 2

        results = outputs.to_pandas()
        assert results.shape[0] == 0 or results.shape[0] == outputs.outputs.shape[0]
        outputs = outputs.outputs.numpy()
        if self.act_fn is not None:
            outputs = self.act_fn(outputs)

        results["pscore"] = outputs.max(-1)
        results["pclass"] = outputs.argmax(-1)
        if self.return_scores:
            results["scores"] = outputs.tolist()
        return WriterOutputs(results)

    @register_process("core/postprocess/classification/embedding")
    def _embedding(
        self,
        outputs: EmbeddingOutputs,
    ):
        """
        Postprocess the embedding outputs.

        Args:
            outputs (EmbeddingOutputs): Outputs from the embedding model.

        Returns:
            WriterOutputs: Processed outputs with embeddings.
        """
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
        """
        Postprocess the embedding outputs as string representations.

        Args:
            outputs (EmbeddingOutputs): Outputs from the embedding model.

        Returns:
            WriterOutputs: Processed outputs with string representations of embeddings.
        """
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
