"""Average several state submissions (same template order) -> outputs/t1/state_submission_ens.csv (or -o <path>)
usage: python -m tfb.t1.blend [-o out.csv] outputs/t1/state_submission.csv outputs/t1/state_submission_seed7.csv ...
       weighted: path@w (e.g. ens_v6t1s3.csv@0.65 ens_v5t1.csv@0.35; weights must sum to 1); no '@' = plain mean as before
"""
import json, sys
from pathlib import Path
import pandas as pd
from .. import io

def main(paths, out=io.OUTPUTS / "t1" / "state_submission_ens.csv"):
    keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
    weighted = any("@" in p for p in paths)
    ws = [float(p.split("@")[1]) for p in paths] if weighted else [1.0] * len(paths)
    paths = [p.split("@")[0] for p in paths]
    assert not weighted or abs(sum(ws) - 1) < 1e-9, "weights must sum to 1"
    base = pd.read_csv(paths[0], dtype={k: str for k in keys})
    sp, fl = ws[0] * base.speed_kmh.to_numpy(), ws[0] * base.flow_vph.to_numpy()
    for p, w in zip(paths[1:], ws[1:]):
        d = pd.read_csv(p, dtype={k: str for k in keys})
        assert (d[keys].to_numpy() == base[keys].to_numpy()).all(), f"{p}: row order differs"
        sp = sp + w * d.speed_kmh.to_numpy(); fl = fl + w * d.flow_vph.to_numpy()
    n = 1 if weighted else len(paths)
    base["speed_kmh"] = (sp / n).round(3); base["flow_vph"] = (fl / n).round(2)
    out = Path(out); base.to_csv(out, index=False)
    out.with_suffix(".meta.json").write_text(json.dumps({"model": "weighted mean" if weighted else "mean of seeds", "inputs": paths, "weights": ws}, indent=2)); print("wrote", out)

if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[2:], a[1]) if a[0] == "-o" else main(a)
