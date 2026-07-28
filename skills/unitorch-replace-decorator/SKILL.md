---
name: replace-decorator
description: Reference for using, reviewing, or modifying unitorch's @replace decorator from the installed unitorch package. Use when overriding upstream classes, working with replacement modules, reasoning about process-global monkey patches, or debugging import-time replacement and subclass __bases__ rewriting.
---

# `@replace` Decorator Reference

`@replace` is available from `unitorch.utils.decorators` after installing unitorch.

## Signature

```python
from unitorch.utils.decorators import replace

@replace(TargetClass)
class ReplacementClass(TargetClass):
    ...
```

## Mechanism

1. Record `TargetClass -> ReplacementClass` in the module-level
   `OPTIMIZED_CLASSES` dict.
2. Set `ReplacementClass.__replaced_class__ = TargetClass`.
3. Walk `sys.modules` and:
   - Replace every module-level name that equals `TargetClass` with
     `ReplacementClass`.
   - Rewrite `__bases__` of any class that inherits from `TargetClass`.

## When To Use

Use `@replace` when you need to override upstream library behavior such as
HuggingFace `diffusers` or `datasets` without forking the library or changing
call sites. The replacement class typically:

- Inherits from the target to reuse its logic.
- Overrides specific methods to fix bugs, skip validation, or add features.

## Conventions In This Codebase

| Location | Pattern |
|----------|---------|
| `unitorch.modules.replace.diffusers_v2` | Override diffusers pipeline `__call__` / `check_inputs`. |
| `unitorch.modules.replace.datasets_v2` | Override HuggingFace datasets iterables for fast skip support. |

Replacement classes are named `<Original>V2` by convention and are decorated
immediately after their definition.

## Important Constraints

- The `@replace` call must happen at module import time. Do not put it inside a
  function or conditional block.
- The replacement runs once when the module is first imported. Re-importing has
  no additional effect, although a warning is logged if the same target is
  replaced twice.
- The replacement is process-global. It affects every consumer of the patched
  module in the same Python process.
- Always inherit from the target class so that `isinstance` checks and
  `super()` calls continue to work correctly.
