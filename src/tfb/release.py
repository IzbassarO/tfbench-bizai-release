"""run.py stage bodies that are not a module CLI: shipped artifacts -> outputs, Task 1 members trained or predicted.

artifacts/ (release assets, see README):
  components/  the four Task 1 member predictions (template order, gzip) + the final queue file      (--from-components)
  weights/     lgbm/*.txt, mlp/seed*/model.pt, unet/seed*/weights.pt, unet2/seed*/weights.pt, t2/ final probabilities
"""
from __future__ import annotations
import gzip, json, shutil
from pathlib import Path
import numpy as np
from . import io

ART = io.PROJECT / "artifacts"
COMP = io.OUTPUTS / "components"
T2_FILES = {"final_mix_clean.pkl": "t2b/hybrid", "final_probs_v2l.pkl": "t2b/v2l", "final_probs.pkl": "t2b",
            "probs_v2.parquet": "t2", "probs_v4_onset.parquet": "t2", "queue_submission_v2.csv": "t2"}
MLP_CONFIG = dict(width=512, depth=2, dropout=0.1, loss="huber", lr=0.001, epochs=40, scheduler="onecycle", batch=16384)
MLP_TAG = "bc973ed531fb"
UNET2_ARCH = dict(epochs=50, batch=8, width=48, lr=0.002, weight_decay=0.0001, win=576, ctx=144)
COMPONENT = dict(unet="state_unet_s5.csv", unet2="state_unet2_s3.csv")


def _copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz" and dst.suffix != ".gz":
        with gzip.open(src, "rb") as a, open(dst, "wb") as b: shutil.copyfileobj(a, b, 1 << 24)
    else:
        shutil.copy2(src, dst)


def unpack_components():
    """--from-components: member predictions -> outputs/components, the final queue file -> outputs/t2."""
    for f in sorted((ART / "components").glob("state_*")):
        _copy(f, COMP / (f.name if f.name.endswith(".csv.gz") and "mlp" in f.name else f.name.removesuffix(".gz")))
    q = "queue_onset_Ahyb_mix_clean_E50v2l.csv"; _copy(ART / "components" / q, io.OUTPUTS / "t2" / q)


def unpack_t2():
    """--skip-train: the Task 2 members' final probabilities on the released windows."""
    for name, d in T2_FILES.items(): _copy(ART / "weights" / "t2" / name, io.OUTPUTS / d / name)


def lgbm_predict(seeds):
    """LightGBM 8x members from the shipped boosters (tfb.t1.train.run_final without the fit)."""
    import lightgbm as lgb
    from . import cache
    from .t1 import train as T
    from .t1.base import write_state
    for s in seeds:
        suf = "_v6t1" + (f"_seed{s}" if s != 42 else "")
        models = {k: lgb.Booster(model_file=str(ART / "weights" / "lgbm" / f"lgbm_{k}{suf}.txt")) for k in ("speed", "flow_lane")}
        cols = models["speed"].feature_name(); preds = {}
        for panel in io.panels():
            pn_train = cache.load(panel, "train")
            for split in ("validation", "private"):
                pn = cache.load(panel, split)
                preds[(panel, split)] = T.predict_panel(models, cols, pn, pn_train, None, np.ones(len(pn.mspeed), bool))
        out = io.OUTPUTS / "t1" / f"state_submission{suf}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
        write_state(preds, out, meta=dict(model="pooled LGBM (shipped booster)", seed=s)); print("wrote", out, flush=True)


def mlp(seeds, train: bool):
    """MLP random_12: feature export, then 3 seeds trained or predicted."""
    import torch
    from .t1 import train as T
    from .t1.mlp_colab import runner, workflow as W
    from .t1.mlp_colab.export import export_final
    assert W.resolve_config(MLP_CONFIG)[1] == MLP_TAG
    T.SEED, T.TRANS, T.TW, T.CW = 42, 8, 1.0, 1.0
    root = io.OUTPUTS / "mlp"; dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not (root / "manifest.json").exists():
        meta = export_final(root, io.panels())
        (root / "manifest.json").write_text(json.dumps(dict(kind="final", bundle_id="reproduce", holdouts=dict(final=meta))))
    out = COMP / "state_submission_mlp_r12_s3.csv.gz"
    if train:
        shutil.copy2(W.final_train(root, root / "runs", MLP_CONFIG, tuple(seeds), dev), out); return
    meta = json.loads((root / "manifest.json").read_text())["holdouts"]["final"]; dirs = []
    for s in seeds:
        z = torch.load(ART / "weights" / "mlp" / f"seed{s}" / "model.pt", map_location="cpu", weights_only=False); c = z["config"]
        net = runner.Net(meta["n_num"], meta["n_flag"], meta["n_panel"], c["width"], c["depth"], c["dropout"])
        net.load_state_dict(z["state"]); net.to(dev); d = root / "infer" / f"seed{s}"; dirs.append(d)
        for split in meta["test_splits"]:
            (d / split).mkdir(parents=True, exist_ok=True); runner.predict(net, root, "final", meta, torch.device(dev), d / split, split=split)
    runner.write_final_csv(root, dirs, out); print("wrote", out, flush=True)


def unet(kind: str, seeds, train: bool):
    """U-Net x5 (kind unet) / U-Net2 x3 (unet2) on all train days + validation/private re-mask targets -> seed-mean component.
    train=False: each run dir gets the shipped weights and only predicts (tfb.t1.unet.train_run's prediction part)."""
    import torch
    from .t1 import unet as U
    from .t2b.postproc import atomic_json
    cfg = dict(U.DEFAULTS, **(UNET2_ARCH if kind == "unet2" else {}))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); runs = [(s, io.OUTPUTS / "t1" / f"{kind}_final" / f"seed{s}") for s in seeds]
    if not train:
        panels, jobs = U.load_all("final", tuple(io.panels()))
        for s, r in runs:
            if (r / "done.json").exists(): continue
            r.mkdir(parents=True, exist_ok=True); shutil.copy2(ART / "weights" / kind / f"seed{s}" / "weights.pt", r / "weights.pt")
            net = U.UNet(U.C_IN, cfg["width"]).to(dev); net.load_state_dict(torch.load(r / "weights.pt", map_location=dev)["net"])
            (r / "pred").mkdir(exist_ok=True)
            for i, split in jobs:
                sp, fl = U.predict(net, panels[i], np.arange(len(panels[i]["ms"]) // U.STEP), dev, win=cfg.get("win", U.WIN), ctx=cfg.get("ctx", U.CTX))
                np.savez(r / "pred" / f"{panels[i]['panel']}_{split}.npz", sp=sp, fl=fl)
            atomic_json(r / "done.json", dict(hold="final", seed=s, cfg=cfg, inference_only=True, device=str(dev)))
    U.final_component(cfg, dev, runs, COMP / COMPONENT[kind])
    if dev.type == "cuda": print(f"{kind}: max GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB", flush=True)
