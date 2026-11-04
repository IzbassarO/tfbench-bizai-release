"""Apply a per-cell physics post-processing setting (a tfb.t3.grid name, fd_* only) to a validation+private state file.

usage: python -m tfb.t3.apply <in_state.csv> <out_state.csv> <setting, e.g. fd_v_a0.5> [skip_panel,skip_panel] [alphas.json]
alphas.json (tfb.t3.tune_pp choice): {panel: {"free": a, "cong": a}} overrides the setting's alpha per panel, separately for
free and congested (v < 0.6 v_f) cells; panels not in it use the setting's alpha.
Only template target cells exist in the file, so only they change; rows/keys/order are preserved (write_state).
Values are per link (stations on one link share a prediction), as everywhere in tfb.t1.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from .. import cache
from ..t1.base import read_state, write_state
from . import post


def main(src, dst, setting, skip="", alphas=""):
    kind, keep, a = setting.split("_")
    assert kind == "fd", "only per-cell fd_* settings need no extra model or field"
    per = json.loads(Path(alphas).read_text()) if alphas else {}
    preds, changed = {}, 0
    for (panel, split), (sp, fl, tgt) in read_state(src).items():
        if panel in skip.split(","):
            preds[(panel, split)] = (sp, fl); continue
        pn = cache.load(panel, split); al = float(a[1:])
        if panel in per:
            obs = pn.melig & np.isfinite(pn.mspeed) & ~tgt if "cong_e" in per[panel] else None   # edge alphas only
            al = post.alpha_cells(sp, pn, per[panel], obs)
        s2, f2 = post.fd_soft(sp, fl, tgt, pn, al, keep)
        changed += int(((np.abs(s2 - sp) > 1e-3) | (np.abs(f2 - fl) > 1e-2))[tgt].sum())
        preds[(panel, split)] = (s2, f2)
    write_state(preds, Path(dst), meta={"post": setting, "skip_panels": skip, "alphas": per, "input": str(src)})
    print(f"wrote {dst}; target cells changed {changed}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
