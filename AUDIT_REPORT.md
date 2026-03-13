# LIONHEART Code Audit Report

Comprehensive audit of the LIONHEART Python package. Findings are categorized by severity.

---

## CRITICAL: Bugs That Cause Incorrect Behavior

### 1. Broken model name assertion — `global_vars.py:7`

```python
assert all([(m.split("_"[0] in ["detect", "subtype"]) for m in INCLUDED_MODELS)])
```

**Two compounding bugs on this single line:**

1. **Misplaced parenthesis changes semantics entirely.** `"_"[0] in ["detect", "subtype"]` evaluates to `"_" in ["detect", "subtype"]` → `False`. So `m.split(False)` would be called, which is a `TypeError`.

2. **But the `TypeError` is never raised** because the expression `(... for m in INCLUDED_MODELS)` creates a **generator object**, and that generator is wrapped in a list: `[<generator>]`. `all([<generator>])` checks whether the single element (the generator object) is truthy — it always is. **The generator body is never evaluated.**

**Result:** The assertion passes for *any* model name, providing zero validation. A model named `"invalid_garbage"` would pass silently.

**Intended code:**
```python
assert all([m.split("_")[0] in ["detect", "subtype"] for m in INCLUDED_MODELS])
```

Note: list comprehension `[...]` vs generator expression `(...)` inside `all()` matters here.

---

### 2. Undefined variable `resources_dir` — `validate.py:269, 315`

When `--custom_model_dir` is specified without `--resources_dir`:

```python
# Line 241-245 (else branch: custom_model_dir path)
else:
    if args.resources_dir is not None:          # Only sometimes assigned
        resources_dir = pathlib.Path(args.resources_dir)
    model_dir = pathlib.Path(args.custom_model_dir)
    model_name = model_dir.stem

# ...

if resources_dir is not None:     # Line 269 — NameError if not assigned above
    paths.set_path(...)

# ...

paths.set_path("resources_dir", resources_dir, "in_dirs")  # Line 315 — always reached, NameError
```

If `args.resources_dir is None` in the `else` branch, `resources_dir` is never defined. Line 269 raises `NameError`. Even if line 269 were guarded, line 315 unconditionally references `resources_dir`.

**Fix:** Initialize `resources_dir = None` before the `if/else` block, and guard line 315 with `if resources_dir is not None:`.

---

### 3. Youden's J formula is wrong in help text — `validate.py:134`, `predict.py:136`

Both files contain:
```
'max_j' is the threshold at the max. of Youden's J (`sensitivity + specificity + 1`).
```

Youden's J statistic is defined as `J = sensitivity + specificity - 1`, not `+ 1`. This is a documentation error that could mislead researchers interpreting the metric.

---

### 4. Help text says "specificity" when it means "sensitivity" — `validate.py:137-138`, `predict.py:139-140`

```python
"\nPrefix a sensitivity-based threshold with <b>'sens_'</b>. \n  The first threshold "
"that should lead to a specificity above this level is chosen. "
#                      ^^^^^^^^^^^
#                      Should be "sensitivity"
```

The description of `sens_` thresholds incorrectly says "specificity" instead of "sensitivity". This appears in both `validate.py` and `predict.py`.

---

## HIGH: Bugs in Error Paths / Edge Cases

### 5. Typo in error message — `train_model.py:299`

```python
"`--required_lionheart_version` was never than "
"the currently installed version of LIONHEART."
```

`"was never than"` should be `"was newer than"`.

---

### 6. Inverted "..." ellipsis logic — `poisson.py:202`

```python
example_negs = x[x < 0].flatten()[:5]
dots = ", ..." if len(example_negs) < 5 else ""
```

The condition is inverted:
- If there are 3 negatives (all shown): adds `", ..."` suggesting more exist (WRONG)
- If there are 8 negatives (only 5 shown): no `", ..."` (WRONG — 3 are hidden)

**Fix:** `dots = ", ..." if np.sum(x < 0) > 5 else ""`

---

### 7. `prepare_modeling` crashes for regression tasks — `prepare_modeling.py:544`

```python
new_label_idx_to_new_label = None  # Line 387, initial value

# ... classification block may set it, regression block does not ...

return {
    ...
    "new_label_to_new_label_idx": {
        lab: idx for idx, lab in new_label_idx_to_new_label.items()  # Line 544-545
    },
}
```

For regression tasks, `new_label_idx_to_new_label` stays `None`, causing `AttributeError: 'NoneType' object has no attribute 'items'`. The package only does classification currently, but this is a latent crash for any future regression support.

---

## MEDIUM: Incorrect/Misleading Documentation

### 8. Copy-pasted docstring — `customize_thresholds.py:1-3`

```python
"""
Script that validates a model on one or more specified validation datasets.
"""
```

This file is the `customize_thresholds` command, not validation. The docstring was copied from `validate.py`.

---

### 9. Typo in `predict.py:2`

```python
"""
Script that applies the model to the features of a singe sample ...
"""
```

`"singe"` should be `"single"`.

---

### 10. Ambiguous operator precedence — `cross_validate.py:408`

```python
if args.k_inner < 0 or len(dataset_paths) - len(train_only) >= 4 and not args.loco:
```

Python parses this as:
```python
if (args.k_inner < 0) or ((len(dataset_paths) - len(train_only) >= 4) and (not args.loco)):
```

While technically correct due to Python's precedence rules, the mixing of `or` and `and` without explicit parentheses is error-prone. Add parentheses for clarity.

---

## LOW: Code Quality / Robustness Concerns

### 11. Bare `except:` clauses — multiple files

`run_predict_single_model.py` (lines 88, 97, 112, 142, 157, 217) and `predict.py` (line 319) use bare `except:` clauses that catch `KeyboardInterrupt`, `SystemExit`, etc. These should use `except Exception:` at minimum.

### 12. Mutable default arguments — multiple files

Several functions use mutable default arguments (lists/dicts), e.g.:
- `correction.py:13-17`: `smoothing_settings: dict = {"kernel_size": 5, ...}`
- `transformers.py:11`: `min_var_thresh: List[float] = [0.0]`
- `transformers.py:12`: `scale_rows: List[str] = ["mean", "std"]`

These are shared across calls and could be mutated. In practice the current code doesn't mutate them, but it's a latent hazard.

### 13. `RunningStats` edge case with single-element input — `running_stats.py:59,92`

When `add_data` is called with a single-element array as the very first call:
```python
self.mean, self.var, self.min, self.max = self._compute_stats(x=x)
# _compute_stats calls np.var(x, ddof=1) which returns NaN for len(x)==1
```

The variance becomes `NaN` and propagates through all subsequent calculations.

### 14. `RunningPearsonR`/`RunningStats` type detection fails on empty arrays — `running_stats.py:107`, `running_pearson_r.py:159`

```python
dtype = type(1.0 + x[0])  # IndexError if x is empty after NaN removal
```

If all elements are NaN and `ignore_nans=True`, the filtered array is empty. The `len(x) == 0` check returns early in `RunningStats.add_data`, but `_check_data` accesses `x[0]` *before* the length check happens in `add_data`. The length check after NaN removal occurs in `add_data`, but `_check_data` has already accessed `x[0]`.

Actually, looking more carefully: `_check_data` is called first, and it accesses `x[0]` on line 107. The `new_n = len(x)` check on line 54 happens *after* `_check_data` returns. So if all values are NaN, `_check_data` will try `x[0]` on an empty array → `IndexError`.

### 15. Smoothing with NaNs — `correction.py:309-312`

```python
smooth_x = np.convolve(x, kernel, mode="same")
smooth_x[np.isnan(smooth_x)] = x[np.isnan(smooth_x)]
```

`np.convolve` propagates NaNs: any window touching a NaN produces NaN in the output. The fallback on line 312 only restores *originally* NaN positions, but adjacent non-NaN values that became NaN due to convolution with a NaN neighbor are lost (become NaN silently). This means data near NaN bins is quietly corrupted.

---

## Summary

| Severity | Count | Key Issues |
|----------|-------|------------|
| Critical | 4 | Broken assertion (global_vars), undefined variable (validate), wrong formula (Youden's J), wrong term (sens/spec) |
| High | 3 | Typo in error msg, inverted ellipsis logic, regression crash |
| Medium | 3 | Wrong docstrings, ambiguous precedence |
| Low | 5 | Bare excepts, mutable defaults, edge cases, NaN handling |
