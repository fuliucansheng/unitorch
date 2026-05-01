# `@replace` Decorator — Detailed Reference

Defined in `src/unitorch/utils/decorators.py`.

## Signature

```python
from unitorch.utils.decorators import replace

@replace(TargetClass)
class ReplacementClass(TargetClass):
    ...
```

## Mechanism

1. Records `TargetClass → ReplacementClass` in the module-level `OPTIMIZED_CLASSES` dict.
2. Sets `ReplacementClass.__replaced_class__ = TargetClass`.
3. Walks `sys.modules` and:
   - Replaces every module-level name that equals `TargetClass` with `ReplacementClass`.
   - Rewrites `__bases__` of any class that inherits from `TargetClass`.

## When to use

Use `@replace` when you need to override upstream library behaviour (e.g. HuggingFace `diffusers`, `datasets`) without forking the library or changing call sites. The replacement class typically:

- Inherits from the target to reuse its logic.
- Overrides specific methods to fix bugs, skip validation, or add features.

## Conventions in this codebase

| Location | Pattern |
|---|---|
| `src/unitorch/modules/replace/diffusers_v2.py` | Override diffusers pipeline `__call__` / `check_inputs` |
| `src/unitorch/modules/replace/datasets_v2.py` | Override HuggingFace datasets iterables for fast skip support |

Replacement classes are named `<Original>V2` by convention and are decorated immediately after their definition.

## Important constraints

- The `@replace` call must happen at **module import time** — never inside a function or conditional block.
- The replacement runs once when the module is first imported; re-importing has no additional effect (a warning is logged if the same target is replaced twice).
- The replacement is **process-global** — it affects every consumer of the patched module in the same Python process.
- Always inherit from the target class so that `isinstance` checks and `super()` calls continue to work correctly.
