#!/usr/bin/env python3
"""Check that the code qrjax and qrjax_ood share is still identical.

The two repositories keep their own copies of a common core (decision of
Nov 2026: copies + this check, no shared package). This script compares those
copies: whole modules where the files are meant to be identical, single
functions where only part of a module is shared. Docstrings, comments and the
package name (qrjax / qrood) are ignored; everything else must match.

    python tools/check_shared.py                      # sibling checkouts ~/qrjax, ~/qrjax_ood
    python tools/check_shared.py --qrjax PATH --qrood PATH

Exit status 1 if anything differs. The same file lives in both repositories;
when you change a shared unit in one, port it to the other and run this.
"""

import argparse
import ast
import sys
from pathlib import Path

# (path inside the package, functions to compare or None for the whole module)
SHARED = [
    ("core/so3.py", None),
    ("rl/sac.py", None),
    ("rl/networks.py", None),
    ("core/controller.py", ["_unit_derivs", "attitude_from_thrust", "integrator_projection"]),
]
# Same equations but NOT identical code (left alone on purpose): core/dynamics.py
# (helper names differ), core/mixer.py (different uncertainty model; the
# prioritized allocator is the same algorithm with a different signature).


def _strip(node):
    """Drop docstrings recursively so only code is compared."""
    for n in ast.walk(node):
        body = getattr(n, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            n.body = body[1:] or [ast.Pass()]
    return node


def _norm(src):
    return src.replace("qrood", "PKG").replace("qrjax", "PKG")


def _units(path, funcs):
    tree = ast.parse(_norm(path.read_text()))
    if funcs is None:
        return {"<module>": ast.unparse(_strip(tree))}
    found = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    return {f: (ast.unparse(_strip(found[f])) if f in found else None) for f in funcs}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qrjax", default=str(Path.home() / "qrjax"))
    p.add_argument("--qrood", default=str(Path.home() / "qrjax_ood"))
    a = p.parse_args()
    roots = {"qrjax": Path(a.qrjax) / "src" / "qrjax", "qrood": Path(a.qrood) / "src" / "qrood"}
    bad = 0
    for rel, funcs in SHARED:
        fa, fb = roots["qrjax"] / rel, roots["qrood"] / rel
        if not (fa.is_file() and fb.is_file()):
            print(f"MISSING    {rel}")
            bad += 1
            continue
        ua, ub = _units(fa, funcs), _units(fb, funcs)
        for name in ua:
            label = rel if name == "<module>" else f"{rel}::{name}"
            if ua[name] is None or ub[name] is None:
                print(f"MISSING    {label}")
                bad += 1
            elif ua[name] == ub[name]:
                print(f"identical  {label}")
            else:
                print(f"DIFFERENT  {label}")
                bad += 1
    print("\nall shared code identical" if not bad else f"\n{bad} unit(s) differ or are missing")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
