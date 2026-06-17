# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import abc
from dataclasses import dataclass, fields
from itertools import chain
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist


def _merge_object_batches(values: List[Any]) -> Any:
    if values[0] is None:
        return None
    if all(isinstance(value, list) for value in values):
        return list(chain.from_iterable(values))
    return values


class TensorBatchMixin:
    """Holds a flat dict of tensors with collective batch operations."""

    def __init__(self, tensors: Optional[Dict] = None, **kwargs):
        self.__tensors__: Dict = {}
        for k, v in {**(tensors or {}), **kwargs}.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            self.__tensors__[k] = v
        self.__post_init__()

    def __post_init__(self):
        if not hasattr(self, "__tensors__"):
            self.__tensors__ = {}
        for key, tensor in self.__tensors__.items():
            assert isinstance(tensor, torch.Tensor)
            setattr(self, key, tensor)

    @classmethod
    def union(cls, *items, dim: int = 0):
        if not items:
            return cls()
        keys = list(items[0].__tensors__.keys())
        for item in items:
            assert list(item.__tensors__.keys()) == keys
        merged = {}
        for key in keys:
            parts = [item.__tensors__[key] for item in items]
            merged[key] = torch.cat(parts, dim=dim) if parts[0] is not None else None
        return cls(**merged)

    @classmethod
    def stack(cls, *items, dim: int = 0):
        if not items:
            return cls()
        keys = list(items[0].__tensors__.keys())
        for item in items:
            assert list(item.__tensors__.keys()) == keys
        merged = {}
        for key in keys:
            parts = [item.__tensors__[key] for item in items]
            merged[key] = torch.stack(parts, dim=dim) if parts[0] is not None else None
        return cls(**merged)

    def add(self, tensors: Union[Dict, "TensorBatchMixin"]):
        src = tensors if isinstance(tensors, dict) else tensors.__tensors__
        for key, value in src.items():
            assert isinstance(value, torch.Tensor)
            if key in self.__tensors__:
                raise ValueError(f"{key!r} already exists in tensors.")
            self.__tensors__[key] = value
            setattr(self, key, value)

    def cpu(self, inplace: bool = False):
        updated = {
            k: (v.cpu() if v is not None else None) for k, v in self.__tensors__.items()
        }
        if inplace:
            self.__tensors__.update(updated)
            return
        return TensorBatchMixin(**updated)

    def cuda(self, inplace: bool = False):
        updated = {
            k: (v.cuda() if v is not None else None)
            for k, v in self.__tensors__.items()
        }
        if inplace:
            self.__tensors__.update(updated)
            return
        return TensorBatchMixin(**updated)

    def sync(self, dim: int = 0, inplace: bool = False):
        updated = {}
        for key, value in self.__tensors__.items():
            if value is not None:
                buckets = [value.clone() for _ in range(dist.get_world_size())]
                dist.all_gather(buckets, value)
                updated[key] = torch.cat(buckets, dim=dim)
            else:
                updated[key] = None
        if inplace:
            self.__tensors__.update(updated)
            return
        return TensorBatchMixin(**updated)

    def dict(self) -> Dict:
        return self.__tensors__


class TensorSeqMixin:
    """Holds a flat dict of tensor *lists* (variable-length sequences per sample)."""

    def __init__(self, list_tensors: Optional[Dict] = None, **kwargs):
        self.__list_tensors__: Dict = {}
        for k, v in {**(list_tensors or {}), **kwargs}.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            self.__list_tensors__[k] = v
        self.__post_init__()

    def __post_init__(self):
        if not hasattr(self, "__list_tensors__"):
            self.__list_tensors__ = {}
        normalised = {}
        for key, value in self.__list_tensors__.items():
            if isinstance(value, torch.Tensor):
                value = [value]
            normalised[key] = value
        self.__list_tensors__ = normalised
        for key, lst in self.__list_tensors__.items():
            assert isinstance(lst, list) and all(
                isinstance(t, torch.Tensor) for t in lst
            )
            setattr(self, key, lst)

    @classmethod
    def union(cls, *items, dim: int = 0):
        if not items:
            return cls()
        keys = list(items[0].__list_tensors__.keys())
        for item in items:
            assert list(item.__list_tensors__.keys()) == keys
        merged = {}
        for key in keys:
            parts = [item.__list_tensors__[key] for item in items]
            merged[key] = (
                list(chain.from_iterable(parts)) if parts[0] is not None else None
            )
        return cls(**merged)

    @classmethod
    def stack(cls, *items, dim: int = 0):
        if not items:
            return cls()
        keys = list(items[0].__list_tensors__.keys())
        for item in items:
            assert list(item.__list_tensors__.keys()) == keys
        merged = {}
        for key in keys:
            parts = [item.__list_tensors__[key] for item in items]
            if parts[0] is not None:
                assert all(len(p) == 1 for p in parts)
                merged[key] = list(chain.from_iterable(parts))
            else:
                merged[key] = None
        return cls(**merged)

    def add(self, list_tensors: Union[Dict, "TensorSeqMixin"]):
        src = (
            list_tensors
            if isinstance(list_tensors, dict)
            else list_tensors.__list_tensors__
        )
        for key, value in src.items():
            if key in self.__list_tensors__:
                raise ValueError(f"{key!r} already exists in list tensors.")
            if isinstance(value, torch.Tensor):
                value = [value]
            self.__list_tensors__[key] = value
            setattr(self, key, value)

    def cpu(self, inplace: bool = False):
        updated = {}
        for key, value in self.__list_tensors__.items():
            updated[key] = (
                [v.cpu() for v in value]
                if value is not None and value[0] is not None
                else None
            )
        if inplace:
            self.__list_tensors__.update(updated)
            return
        return TensorSeqMixin(**updated)

    def cuda(self, inplace: bool = False):
        updated = {}
        for key, value in self.__list_tensors__.items():
            updated[key] = (
                [v.cuda() for v in value]
                if value is not None and value[0] is not None
                else None
            )
        if inplace:
            self.__list_tensors__.update(updated)
            return
        return TensorSeqMixin(**updated)

    def sync(self, inplace: bool = False):
        updated = {}
        for key, value in self.__list_tensors__.items():
            if value is not None:
                buckets = [[] for _ in range(dist.get_world_size())]
                dist.all_gather_object(buckets, value)
                updated[key] = list(chain.from_iterable(buckets))
            else:
                updated[key] = None
        if inplace:
            self.__list_tensors__.update(updated)
            return
        return TensorSeqMixin(**updated)

    def dict(self) -> Dict:
        return self.__list_tensors__


class ObjectBatchMixin:
    """Holds a flat dict of arbitrary Python objects."""

    def __init__(self, objects: Optional[Dict] = None, **kwargs):
        self.__objects__: Dict = {}
        for k, v in {**(objects or {}), **kwargs}.items():
            if not hasattr(self, k):
                setattr(self, k, v)
            self.__objects__[k] = v
        self.__post_init__()

    def __post_init__(self):
        if not hasattr(self, "__objects__"):
            self.__objects__ = {}
        for key, value in self.__objects__.items():
            setattr(self, key, value)

    @classmethod
    def union(cls, *items, dim: int = 0):
        if not items:
            return cls()
        keys = list(items[0].__objects__.keys())
        for item in items:
            assert list(item.__objects__.keys()) == keys
        merged = {}
        for key in keys:
            parts = [item.__objects__[key] for item in items]
            merged[key] = _merge_object_batches(parts)
        return cls(**merged)

    @classmethod
    def stack(cls, *items, dim: int = 0):
        if not items:
            return cls()
        keys = list(items[0].__objects__.keys())
        for item in items:
            assert list(item.__objects__.keys()) == keys
        merged = {}
        for key in keys:
            parts = [item.__objects__[key] for item in items]
            merged[key] = parts if parts[0] is not None else None
        return cls(**merged)

    def add(self, objects: Union[Dict, "ObjectBatchMixin"]):
        src = objects if isinstance(objects, dict) else objects.__objects__
        for key, value in src.items():
            if key in self.__objects__:
                raise ValueError(f"{key!r} already exists in objects.")
            self.__objects__[key] = value
            setattr(self, key, value)

    def cpu(self, inplace: bool = False):
        if inplace:
            return
        return ObjectBatchMixin(**self.__objects__)

    def cuda(self, inplace: bool = False):
        if inplace:
            return
        return ObjectBatchMixin(**self.__objects__)

    def sync(self, inplace: bool = False):
        updated = {}
        for key, value in self.__objects__.items():
            if value is not None:
                buckets = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(buckets, value)
                updated[key] = _merge_object_batches(buckets)
            else:
                updated[key] = None
        if inplace:
            self.__objects__.update(updated)
            return
        return ObjectBatchMixin(**updated)

    def dict(self) -> Dict:
        return self.__objects__


class TensorMixMixin(TensorBatchMixin, TensorSeqMixin, ObjectBatchMixin):
    """Combines tensor, tensor-sequence, and object batch dictionaries."""

    def __init__(
        self,
        dict_of_tensors: Optional[Dict[str, torch.Tensor]] = None,
        dict_of_list_tensors: Optional[Dict[str, List[torch.Tensor]]] = None,
        dict_of_objects: Optional[Dict[str, Any]] = None,
    ):
        self.__tensors__ = dict_of_tensors or {}
        self.__list_tensors__ = dict_of_list_tensors or {}
        self.__objects__ = dict_of_objects or {}
        self.__post_init__()

    def __post_init__(self):
        TensorBatchMixin.__post_init__(self)
        TensorSeqMixin.__post_init__(self)
        ObjectBatchMixin.__post_init__(self)

    @classmethod
    def union(cls, *items, dim: int = 0):
        if not items:
            return cls()
        t_keys = list(items[0].__tensors__.keys())
        l_keys = list(items[0].__list_tensors__.keys())
        o_keys = list(items[0].__objects__.keys())
        for item in items:
            if list(item.__tensors__.keys()) != t_keys:
                raise ValueError("Tensor key mismatch in union.")
            if list(item.__list_tensors__.keys()) != l_keys:
                raise ValueError("List-tensor key mismatch in union.")
            if list(item.__objects__.keys()) != o_keys:
                raise ValueError("Object key mismatch in union.")
        new_t = {
            k: (
                torch.cat([i.__tensors__[k] for i in items], dim=dim)
                if items[0].__tensors__[k] is not None
                else None
            )
            for k in t_keys
        }
        new_l = {
            k: (
                list(chain.from_iterable(i.__list_tensors__[k] for i in items))
                if items[0].__list_tensors__[k] is not None
                else None
            )
            for k in l_keys
        }
        new_o = {
            k: _merge_object_batches([i.__objects__[k] for i in items])
            for k in o_keys
        }
        return cls(
            dict_of_tensors=new_t,
            dict_of_list_tensors=new_l,
            dict_of_objects=new_o,
        )

    @classmethod
    def stack(cls, *items, dim: int = 0):
        if not items:
            return cls()
        t_keys = list(items[0].__tensors__.keys())
        l_keys = list(items[0].__list_tensors__.keys())
        o_keys = list(items[0].__objects__.keys())
        for item in items:
            if list(item.__tensors__.keys()) != t_keys:
                raise ValueError("Tensor key mismatch in stack.")
            if list(item.__list_tensors__.keys()) != l_keys:
                raise ValueError("List-tensor key mismatch in stack.")
            if list(item.__objects__.keys()) != o_keys:
                raise ValueError("Object key mismatch in stack.")
        new_t = {
            k: (
                torch.stack([i.__tensors__[k] for i in items], dim=dim)
                if items[0].__tensors__[k] is not None
                else None
            )
            for k in t_keys
        }
        new_l = {
            k: (
                [i.__list_tensors__[k] for i in items]
                if items[0].__list_tensors__[k] is not None
                else None
            )
            for k in l_keys
        }
        new_o = {
            k: (
                [i.__objects__[k] for i in items]
                if items[0].__objects__[k] is not None
                else None
            )
            for k in o_keys
        }
        return cls(
            dict_of_tensors=new_t,
            dict_of_list_tensors=new_l,
            dict_of_objects=new_o,
        )

    def add(self, other: Union[TensorBatchMixin, TensorSeqMixin, ObjectBatchMixin]):
        if isinstance(other, TensorBatchMixin):
            TensorBatchMixin.add(self, other.__tensors__)
        if isinstance(other, TensorSeqMixin):
            TensorSeqMixin.add(self, other.__list_tensors__)
        if isinstance(other, ObjectBatchMixin):
            ObjectBatchMixin.add(self, other.__objects__)

    def cpu(self, inplace: bool = False):
        t = TensorBatchMixin.cpu(self, inplace=inplace)
        l = TensorSeqMixin.cpu(self, inplace=inplace)
        o = ObjectBatchMixin.cpu(self, inplace=inplace)
        if inplace:
            return
        return TensorMixMixin(
            dict_of_tensors=t.__tensors__,
            dict_of_list_tensors=l.__list_tensors__,
            dict_of_objects=o.__objects__,
        )

    def cuda(self, inplace: bool = False):
        t = TensorBatchMixin.cuda(self, inplace=inplace)
        l = TensorSeqMixin.cuda(self, inplace=inplace)
        o = ObjectBatchMixin.cuda(self, inplace=inplace)
        if inplace:
            return
        return TensorMixMixin(
            dict_of_tensors=t.__tensors__,
            dict_of_list_tensors=l.__list_tensors__,
            dict_of_objects=o.__objects__,
        )

    def sync(self, inplace: bool = False):
        t = TensorBatchMixin.sync(self, inplace=inplace)
        l = TensorSeqMixin.sync(self, inplace=inplace)
        o = ObjectBatchMixin.sync(self, inplace=inplace)
        if inplace:
            return
        return TensorMixMixin(
            dict_of_tensors=t.__tensors__,
            dict_of_list_tensors=l.__list_tensors__,
            dict_of_objects=o.__objects__,
        )

    def dict(self) -> Dict:
        return {**self.__tensors__, **self.__list_tensors__, **self.__objects__}


# ---------------------------------------------------------------------------
# Abstract base markers
# ---------------------------------------------------------------------------


class ModelInputs(abc.ABC):
    pass


class ModelOutputs(abc.ABC):
    pass


class ModelTargets(abc.ABC):
    pass


# ---------------------------------------------------------------------------
# Concrete Input / Output / Target types
# ---------------------------------------------------------------------------


@dataclass(init=False)
class TensorInputs(ModelInputs, TensorBatchMixin):
    def __init__(self, inputs: Optional[Dict] = None, **kwargs):
        TensorBatchMixin.__init__(self, tensors=inputs, **kwargs)

    def cpu(self, inplace=False):
        r = TensorBatchMixin.cpu(self, inplace=inplace)
        if inplace:
            return
        return TensorInputs(**r.__tensors__)

    def cuda(self, inplace=False):
        r = TensorBatchMixin.cuda(self, inplace=inplace)
        if inplace:
            return
        return TensorInputs(**r.__tensors__)

    def sync(self, inplace=False):
        r = TensorBatchMixin.sync(self, inplace=inplace)
        if inplace:
            return
        return TensorInputs(**r.__tensors__)


@dataclass
class TensorOutputs(ModelOutputs, TensorBatchMixin):
    def __post_init__(self):
        self.__tensors__ = {f.name: getattr(self, f.name) for f in fields(self)}
        TensorBatchMixin.__post_init__(self)

    def cpu(self, inplace=False):
        r = TensorBatchMixin.cpu(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__tensors__)

    def cuda(self, inplace=False):
        r = TensorBatchMixin.cuda(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__tensors__)

    def sync(self, inplace=False):
        r = TensorBatchMixin.sync(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__tensors__)


@dataclass
class TensorTargets(ModelTargets, TensorBatchMixin):
    def __post_init__(self):
        self.__tensors__ = {f.name: getattr(self, f.name) for f in fields(self)}
        TensorBatchMixin.__post_init__(self)

    def cpu(self, inplace=False):
        r = TensorBatchMixin.cpu(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__tensors__)

    def cuda(self, inplace=False):
        r = TensorBatchMixin.cuda(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__tensors__)

    def sync(self, inplace=False):
        r = TensorBatchMixin.sync(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__tensors__)


@dataclass(init=False)
class ObjectInputs(ModelInputs, ObjectBatchMixin):
    def __init__(self, inputs: Optional[Dict] = None, **kwargs):
        ObjectBatchMixin.__init__(self, objects=inputs, **kwargs)

    def cpu(self, inplace=False):
        r = ObjectBatchMixin.cpu(self, inplace=inplace)
        if inplace:
            return
        return ObjectInputs(**r.__objects__)

    def cuda(self, inplace=False):
        r = ObjectBatchMixin.cuda(self, inplace=inplace)
        if inplace:
            return
        return ObjectInputs(**r.__objects__)

    def sync(self, inplace=False):
        r = ObjectBatchMixin.sync(self, inplace=inplace)
        if inplace:
            return
        return ObjectInputs(**r.__objects__)


@dataclass
class ObjectOutputs(ModelOutputs, ObjectBatchMixin):
    def __post_init__(self):
        self.__objects__ = {f.name: getattr(self, f.name) for f in fields(self)}
        ObjectBatchMixin.__post_init__(self)

    def cpu(self, inplace=False):
        r = ObjectBatchMixin.cpu(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__objects__)

    def cuda(self, inplace=False):
        r = ObjectBatchMixin.cuda(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__objects__)

    def sync(self, inplace=False):
        r = ObjectBatchMixin.sync(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__objects__)


@dataclass
class ObjectTargets(ModelTargets, ObjectBatchMixin):
    def __post_init__(self):
        self.__objects__ = {f.name: getattr(self, f.name) for f in fields(self)}
        ObjectBatchMixin.__post_init__(self)

    def cpu(self, inplace=False):
        r = ObjectBatchMixin.cpu(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__objects__)

    def cuda(self, inplace=False):
        r = ObjectBatchMixin.cuda(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__objects__)

    def sync(self, inplace=False):
        r = ObjectBatchMixin.sync(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__objects__)


@dataclass(init=False)
class TensorSeqInputs(ModelInputs, TensorSeqMixin):
    def __init__(self, inputs: Optional[Dict] = None, **kwargs):
        TensorSeqMixin.__init__(self, list_tensors=inputs, **kwargs)

    def cpu(self, inplace=False):
        r = TensorSeqMixin.cpu(self, inplace=inplace)
        if inplace:
            return
        return TensorSeqInputs(**r.__list_tensors__)

    def cuda(self, inplace=False):
        r = TensorSeqMixin.cuda(self, inplace=inplace)
        if inplace:
            return
        return TensorSeqInputs(**r.__list_tensors__)

    def sync(self, inplace=False):
        r = TensorSeqMixin.sync(self, inplace=inplace)
        if inplace:
            return
        return TensorSeqInputs(**r.__list_tensors__)


@dataclass
class TensorSeqOutputs(ModelOutputs, TensorSeqMixin):
    def __post_init__(self):
        self.__list_tensors__ = {f.name: getattr(self, f.name) for f in fields(self)}
        TensorSeqMixin.__post_init__(self)

    def cpu(self, inplace=False):
        r = TensorSeqMixin.cpu(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__list_tensors__)

    def cuda(self, inplace=False):
        r = TensorSeqMixin.cuda(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__list_tensors__)

    def sync(self, inplace=False):
        r = TensorSeqMixin.sync(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__list_tensors__)


@dataclass
class TensorSeqTargets(ModelTargets, TensorSeqMixin):
    def __post_init__(self):
        self.__list_tensors__ = {f.name: getattr(self, f.name) for f in fields(self)}
        TensorSeqMixin.__post_init__(self)

    def cpu(self, inplace=False):
        r = TensorSeqMixin.cpu(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__list_tensors__)

    def cuda(self, inplace=False):
        r = TensorSeqMixin.cuda(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__list_tensors__)

    def sync(self, inplace=False):
        r = TensorSeqMixin.sync(self, inplace=inplace)
        if inplace:
            return
        return type(self)(**r.__list_tensors__)


class TensorMixInputs(ModelInputs, TensorMixMixin):
    def __init__(
        self,
        dict_of_tensors: Optional[Dict[str, torch.Tensor]] = None,
        dict_of_list_tensors: Optional[Dict[str, List[torch.Tensor]]] = None,
        dict_of_objects: Optional[Dict[str, Any]] = None,
    ):
        TensorMixMixin.__init__(
            self,
            dict_of_tensors=dict_of_tensors,
            dict_of_list_tensors=dict_of_list_tensors,
            dict_of_objects=dict_of_objects,
        )


class TensorMixOutputs(ModelOutputs, TensorMixMixin):
    def __init__(
        self,
        dict_of_tensors: Optional[Dict[str, torch.Tensor]] = None,
        dict_of_list_tensors: Optional[Dict[str, List[torch.Tensor]]] = None,
        dict_of_objects: Optional[Dict[str, Any]] = None,
    ):
        TensorMixMixin.__init__(
            self,
            dict_of_tensors=dict_of_tensors,
            dict_of_list_tensors=dict_of_list_tensors,
            dict_of_objects=dict_of_objects,
        )


class TensorMixTargets(ModelTargets, TensorMixMixin):
    def __init__(
        self,
        dict_of_tensors: Optional[Dict[str, torch.Tensor]] = None,
        dict_of_list_tensors: Optional[Dict[str, List[torch.Tensor]]] = None,
        dict_of_objects: Optional[Dict[str, Any]] = None,
    ):
        TensorMixMixin.__init__(
            self,
            dict_of_tensors=dict_of_tensors,
            dict_of_list_tensors=dict_of_list_tensors,
            dict_of_objects=dict_of_objects,
        )


@dataclass
class LossOutputs(TensorOutputs):
    loss: torch.Tensor
