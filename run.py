"""TrafficFlowBench 2026 (IEEE Big Data Cup), team BizAI: one entry point for both selected finals.

  python run.py --final v3 [--data DIR] [--skip-train | --from-components] [--stages a,b,...] [--force]
    --final v3   cand_combo_v3 (public LB 0.88378, sha256 7ea29080...), final no. 1
    --final v1   cand_combo_v1 (public LB 0.88327, sha256 d7d0912b...), final no. 2
    --data       the unzipped competition release (folder with config/, corridors/, task1/ ... submission_key.csv);
                 default data/kaggle_public
    full run (default): every model is trained from the training split, then predicted and assembled
    --skip-train inference only: Task 1 members predicted from the shipped weights (artifacts/weights), Task 2 decoded
                 from the shipped final probabilities; no training anywhere
    --from-components  the shipped Task 1 member predictions (artifacts/components) + Task 2 queue file: minutes on a
                 CPU, byte-identical to the submitted file
  -> outputs/submission_<final>.csv (+ .sha256), outputs/run_log.tsv (stage, seconds, peak RSS of the stage process).

Stages run as separate processes, in order; a finished stage leaves outputs/.done/<mode>/<stage>.json and is skipped on the
next call (resume after a lost session); --force reruns the named --stages. Seeds are fixed in every module.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
ART = ROOT / "artifacts"
PY = sys.executable
# LightGBM's deterministic mode is fixed per thread count: num_threads is part of the recipe (8 for Task 1, 2 for V4).
ENV = dict(PYTHONPATH=str(ROOT / "src"), PYTHONWARNINGS="ignore", PYTHONDONTWRITEBYTECODE="1",
           OMP_NUM_THREADS="8", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def py(code: str) -> list[str]:
    return [PY, "-c", code]


def m(module: str, *args: str) -> list[str]:
    return [PY, "-m", module, *args]


# ------------------------------------------------------------------------------------------------ Task 2 (CPU)
T2_CODEX = [   # family A v1 dataset + v2 (ongoing member) + v4 (onset member) on the training split
    m("tfb.t2.run", "--stage", "all", "--feature-set", "enhanced", "--window-policy", "release", "--refresh", "--median-experiment",
      "--release-root", "{data}"),
    m("tfb.t2.v2_experiment", "--stage", "all", "--root", "{data}"),
    m("tfb.t2.v4_export", "--root", "{data}"),
    m("tfb.t2.v4_data", "--fold", "final", "--root", "{data}"),
    m("tfb.t2.v4_model", "--fold", "final", "--stage", "normal", "--threads", "2", "--root", "{data}"),
    m("tfb.t2.v4_predict", "--threads", "2", "--skip-shared-check", "--root", "{data}"),
]
T2B_BASE = [m("tfb.t2b.run", "--stage", "windows"), m("tfb.t2b.run", "--stage", "final")]
IMPUTER = [py("from tfb.t1 import imputer_clean as IC; IC.fit('outputs/t1/imputer_clean', rounds=20000, seed=42)")]
ONSET_TRAIN = [
    m("tfb.t2.v4_hyb", "--stage", "released"),
    py("from tfb.t2b import hybrid as H, hyb_clean as HC\n"
       "H.HYB_MIX = H.io.OUTPUTS / 't1' / 'train_hybrid_speed_clean'\n"
       "HC.labels(H.io.OUTPUTS / 't1' / 'imputer_clean' / 'lgbm_speed.txt', 4000, H.HYB_MIX, lambda *a, **k: None)\n"
       "H.build_cache('mix_clean'); H.stage_final('mix_clean')"),
]
ONSET_DECODE = [   # A_hyb = decode30(0.5 v4_hyb + 0.5 t2b noloc_hyb), spliced over E25's ongoing rows (replaced below)
    py("from tfb.t2b import hybrid as H, ladder; from tfb.t2b.ladder import write_e25\n"
       "ladder.write()   # queue_onset_A.csv: the row template of write_a_hyb\n"
       "H.write_a_hyb('mix_clean'); write_e25('../t2/queue_onset_E25.csv', 0.25)\n"
       "H.splice('queue_onset_A_hyb_mix_clean.csv', 'queue_onset_E25.csv', 'queue_onset_Ahyb_mix_clean_E25.csv', ref='queue_onset_A_hyb_mix_clean.csv')"),
]
ONGOING_TRAIN = [   # v2l: family B noloc ongoing + slow-link features, trained on the selector's admissible ongoing windows,
    py("import pickle; from tfb.t2b import harness as HS, hybrid as H, ongoing_v2l as V\n"   # selected on clean labels (rule B)
       "H.HYB_MIX = H.io.OUTPUTS / 't1' / 'train_hybrid_speed_clean'   # written by stage onset (clean imputer)\n"
       "W = H.io.OUTPUTS / 't2b' / 'harness_clean'; W.mkdir(parents=True, exist_ok=True)\n"
       "for p in HS.io.T2_PANELS:\n"
       "    f = W / f'windows_{p}.parquet'\n"
       "    f.exists() or HS.select(p, 'mix').to_parquet(f)\n"
       "ms, P, keep = V.fit_final(cache='cache_hw_clean', windows=W); V.OUT.mkdir(parents=True, exist_ok=True)\n"
       "pickle.dump(dict(v2l=P, keep=keep), open(V.OUT / 'final_probs_v2l.pkl', 'wb'))"),
]
ONGOING_DECODE = [   # ongoing rows = exact expected-IoU top-m of 0.5 family A v2 + 0.5 v2l over the 6 horizons
    py("import pickle; from tfb.t2b import ongoing_v2l as V\n"
       "P = pickle.load(open(V.OUT / 'final_probs_v2l.pkl', 'rb'))['v2l']\n"
       "print(V.write_queue(P, 0.5, 'queue_onset_Ahyb_mix_clean_E50v2l.csv')[0])"),
]

# ------------------------------------------------------------------------------------------------ Task 1 members
LGBM_SEEDS, MLP_SEEDS, UNET_SEEDS, UNET2_SEEDS = (42, 7, 123), (42, 7, 123), (42, 7, 123, 2031, 137), (42, 7, 123)
LGBM = [[*m("tfb.t1.train", "final"), "@env", f"TFB_ROUNDS=20000 TFB_TRANS=8 TFB_TAG=v6t1 TFB_SEED={s}"] for s in LGBM_SEEDS] + [
    m("tfb.t1.blend", "-o", "outputs/components/state_submission_ens_v6t1s3.csv",
      *[f"outputs/t1/state_submission_v6t1{'' if s == 42 else f'_seed{s}'}.csv" for s in LGBM_SEEDS])]
LGBM_INFER = [py(f"from tfb import release as R; R.lgbm_predict({list(LGBM_SEEDS)})")] + LGBM[-1:]
MLP = [py(f"from tfb import release as R; R.mlp({list(MLP_SEEDS)}, train=True)")]
MLP_INFER = [py(f"from tfb import release as R; R.mlp({list(MLP_SEEDS)}, train=False)")]
UNET = [py(f"from tfb import release as R; R.unet('unet', {list(UNET_SEEDS)}, train=True)")]
UNET_INFER = [py(f"from tfb import release as R; R.unet('unet', {list(UNET_SEEDS)}, train=False)")]
UNET2 = [py(f"from tfb import release as R; R.unet('unet2', {list(UNET2_SEEDS)}, train=True)")]
UNET2_INFER = [py(f"from tfb import release as R; R.unet('unet2', {list(UNET2_SEEDS)}, train=False)")]

CACHE = [m("tfb.cache")]
T4 = [m("tfb.t4.project")]


def plan(final: str, mode: str) -> list[tuple[str, list, str]]:
    """(stage, commands, lane). Lanes: prep -> {t2, lgbm, gpu} (independent: no shared outputs) -> final.
    Task 2, Task 4, cache and LightGBM run on the CPU; mlp / unet / unet2 use CUDA when present."""
    fin = (f"final_{final}", [m("tfb.finals", final, "--components", "outputs/components", "--out", "outputs/final")], "final")
    prep = [("cache", CACHE, "prep"), ("t4", T4, "prep")]
    if mode == "components":
        return prep + [("components", [py("from tfb import release as R; R.unpack_components()")], "prep"), fin]
    train = mode == "full"
    t2 = ([("t2_codex", T2_CODEX), ("t2b_base", T2B_BASE), ("imputer", IMPUTER), ("onset", ONSET_TRAIN), ("ongoing", ONGOING_TRAIN)]
          if train else [("t2_probs", [py("from tfb import release as R; R.unpack_t2()")])])
    t2 += [("onset_decode", ONSET_DECODE), ("ongoing_decode", ONGOING_DECODE)]
    gpu = [("mlp", MLP if train else MLP_INFER), ("unet", UNET if train else UNET_INFER), ("unet2", UNET2 if train else UNET2_INFER)]
    return (prep + [(n, c, "t2") for n, c in t2] + [("lgbm", LGBM if train else LGBM_INFER, "lgbm")]
            + [(n, c, "gpu") for n, c in gpu] + [fin])


def run_cmd(cmd: list[str], data: Path, log) -> tuple[float, float]:
    env = dict(os.environ, **ENV, TFB_RELEASE=str(data))
    if "@env" in cmd:
        i = cmd.index("@env"); env.update(kv.split("=", 1) for kv in cmd[i + 1].split()); cmd = cmd[:i]
    cmd = [c.replace("{data}", str(data)) for c in cmd]
    log(f"  $ {' '.join(cmd if cmd[1] != '-c' else cmd[:2] + [cmd[2].splitlines()[0] + ' ...'])}")
    t0 = time.time(); p = subprocess.Popen(cmd, cwd=ROOT, env=env)
    _, status, ru = os.wait4(p.pid, 0)
    rss = ru.ru_maxrss / (1024 ** 3 if platform.system() == "Darwin" else 1024 ** 2)   # bytes on macOS, KiB on Linux -> GiB
    if os.waitstatus_to_exitcode(status): raise SystemExit(f"stage command failed ({os.waitstatus_to_exitcode(status)}): {cmd}")
    return time.time() - t0, rss


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", choices=["v3", "v1"], required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "kaggle_public")
    g = ap.add_mutually_exclusive_group(); g.add_argument("--skip-train", action="store_true"); g.add_argument("--from-components", action="store_true")
    ap.add_argument("--stages", help="comma-separated subset (default: all, in order)"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="3: run the Task 2, LightGBM and GPU lanes concurrently (same results)")
    a = ap.parse_args()
    data = a.data.resolve()
    if not (data / "submission_key.csv").is_file(): raise SystemExit(f"--data {data}: submission_key.csv not found")
    mode = "components" if a.from_components else "infer" if a.skip_train else "full"
    done = OUT / ".done" / mode; done.mkdir(parents=True, exist_ok=True); (OUT / "components").mkdir(exist_ok=True)
    logf = open(OUT / "run.log", "a"); lock = threading.Lock()

    def log(s):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
        with lock: print(line, flush=True); logf.write(line + "\n"); logf.flush()
    tsv = OUT / "run_log.tsv"
    if not tsv.exists(): tsv.write_text("final\tmode\tstage\tseconds\tpeak_rss_gib\thost\n")
    only = set(a.stages.split(",")) if a.stages else None
    log(f"run.py --final {a.final} mode {mode} data {data} python {platform.python_version()} on {platform.node()}")
    def stage(name, cmds):
        marker = done / f"{name}.json"
        if only and name not in only: return
        if marker.exists() and not (a.force and only):
            log(f"[{name}] done earlier ({json.loads(marker.read_text())['seconds']:.0f} s), skipped"); return
        log(f"[{name}] start"); secs, peak = 0.0, 0.0
        for c in cmds:
            s, r = run_cmd(c, data, log); secs += s; peak = max(peak, r)
        marker.write_text(json.dumps(dict(stage=name, mode=mode, seconds=secs, peak_rss_gib=peak, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))))
        with lock, open(tsv, "a") as f: f.write(f"{a.final}\t{mode}\t{name}\t{secs:.0f}\t{peak:.2f}\t{platform.node()}\n")
        log(f"[{name}] done in {secs / 60:.1f} min, peak RSS {peak:.2f} GiB")

    st = plan(a.final, mode); lanes = [l for l in ("t2", "lgbm", "gpu") if any(x[2] == l for x in st)]
    lane = lambda l: [stage(n, c) for n, c, x in st if x == l]
    lane("prep")
    if a.jobs > 1:   # the three lanes at once (>= 20 cores and >= 48 GB RAM: LightGBM 8 threads ~18 GB, MLP export, Task 2)
        with ThreadPoolExecutor(a.jobs) as ex:
            for f in [ex.submit(lane, l) for l in lanes]: f.result()
    else:
        for l in lanes: lane(l)
    lane("final")
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "finals.yaml").read_text())[a.final]
    built = OUT / "final" / f"{cfg['name']}.csv"
    if built.exists():
        dst = OUT / f"submission_{a.final}.csv"; shutil.copy2(built, dst); h = sha(dst)
        (OUT / f"submission_{a.final}.sha256").write_text(f"{h}  {dst.name}\n")
        log(f"{dst}: sha256 {h} " + ("== the submitted file" if h == cfg["sha256"] else
            f"!= the submitted {cfg['sha256'][:8]} (expected unless --from-components: GPU/CPU float rounding; run scripts/verify.py)"))


if __name__ == "__main__":
    main()
