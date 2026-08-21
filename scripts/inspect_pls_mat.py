"""One-shot: dump the structure of PAA_PLS_model.mat for porting reference."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io


def describe(name: str, val) -> str:
    if isinstance(val, np.ndarray):
        return f"  {name}: ndarray shape={val.shape} dtype={val.dtype}"
    return f"  {name}: {type(val).__name__} = {val!r}"


def main(path: Path) -> int:
    raw = scipy.io.loadmat(str(path), squeeze_me=False, struct_as_record=True)
    print(f"# {path.name}\n")
    user_keys = [k for k in raw if not k.startswith("__")]
    meta_keys = [k for k in raw if k.startswith("__")]

    print("## meta")
    for k in meta_keys:
        print(describe(k, raw[k]))

    print("\n## user-level variables")
    for k in user_keys:
        v = raw[k]
        print(describe(k, v))
        # Recurse one level into structs
        if isinstance(v, np.ndarray) and v.dtype.names:
            for fname in v.dtype.names:
                fval = v[fname]
                print(f"    .{fname}: shape={fval.shape} dtype={fval.dtype}")
                # Unwrap nested array-of-array
                if fval.dtype == object:
                    inner = fval[0, 0] if fval.shape == (1, 1) else None
                    if inner is not None:
                        print(f"       inner: shape={getattr(inner, 'shape', None)} "
                              f"dtype={getattr(inner, 'dtype', None)}")

    print("\n## sample numerical content (first var)")
    if user_keys:
        v = raw[user_keys[0]]
        if isinstance(v, np.ndarray) and v.dtype != object:
            arr = np.asarray(v).squeeze()
            print(f"  shape={arr.shape}")
            if arr.size <= 30:
                print(f"  values: {arr}")
            else:
                print(f"  first 5: {arr.flat[:5]}")
                print(f"  last 5:  {arr.flat[-5:]}")
                print(f"  min={arr.min():.6g} max={arr.max():.6g} mean={arr.mean():.6g}")
    return 0


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("indpensim/data/PAA_PLS_model.mat")
    sys.exit(main(p))
