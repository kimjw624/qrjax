"""Stage 2 -- training queue (resumable, deadline-aware).

Groups (all with every bug fix on, paper protocol otherwise: 200 Hz, flat
training over the full box, 2M env steps, 256 envs, UTD 1, history 10):

  A  02_retrain_fixed : {PID, PD} x {thrust LPF beta 0.2, no LPF}, paper reward
                        (tau_pos 0.5) and authority 0.2            -> Q1 + Q2
  B  03_reward_sweep  : PID + LPF, tau_pos {0.5, 0.2, 0.1, 0.05} x authority
                        {0.2, 0.4, 0.6}  ((0.5, 0.2) is group A's run) -> Q3
  C  03_reward_sweep  : PID, NO LPF, tau_pos 0.05 x authority {0.2, 0.6}:
                        is the LPF still needed with a sharp reward?    -> Q2/Q3
  H  talk ablation    : {PID, PD} + thrust LPF, history {1, 20} (10 = group A),
                        paper reward and authority: does the history help?
  M  talk ablation    : {PID, PD}, LPF on thrust AND moments (beta 0.2 each),
                        paper reward and authority: why filter thrust only?

Order: group A for every seed first, then B+C seed by seed, then H, M (the
talk ablations), so a short night
still finishes the LPF question with all seeds. A job is skipped if a finished
run (DONE.json) exists; a job is not started if it would not finish before the
deadline minus the reserve kept for evaluation and the report.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TAUS = (0.5, 0.2, 0.1, 0.05)
AUTHS = (0.2, 0.4, 0.6)
HISTORIES = (1, 20)          # group H; history 10 is group A's run


def job_list(seeds):
    jobs = []
    for s in seeds:
        for base in ("pid", "pd"):
            for beta in (0.2, 1.0):
                jobs.append({"name": f"A_{base}_{'lpf' if beta < 1 else 'nolpf'}_s{s}",
                             "group": "A", "seed": s, "base": base, "beta": beta,
                             "tau_pos": 0.5, "auth": 0.2})
    for s in seeds:
        for auth in AUTHS:
            for tau in TAUS:
                if (tau, auth) == (0.5, 0.2):
                    continue                     # = A_pid_lpf
                jobs.append({"name": f"B_pid_lpf_tau{tau:g}_auth{auth:g}_s{s}",
                             "group": "B", "seed": s, "base": "pid", "beta": 0.2,
                             "tau_pos": tau, "auth": auth})
        for auth in (0.2, 0.6):
            jobs.append({"name": f"C_pid_nolpf_tau0.05_auth{auth:g}_s{s}",
                         "group": "C", "seed": s, "base": "pid", "beta": 1.0,
                         "tau_pos": 0.05, "auth": auth})
    for s in seeds:
        for base in ("pid", "pd"):
            for h in HISTORIES:
                jobs.append({"name": f"H_{base}_lpf_h{h}_s{s}", "group": "H", "seed": s,
                             "base": base, "beta": 0.2, "tau_pos": 0.5, "auth": 0.2,
                             "history": h})
    for s in seeds:
        for base in ("pid", "pd"):
            jobs.append({"name": f"M_{base}_lpfall_s{s}", "group": "M", "seed": s,
                         "base": base, "beta": 0.2, "beta_m": 0.2, "tau_pos": 0.5,
                         "auth": 0.2})
    return jobs


def finished_run(results, name):
    """Trial dir of a finished run of this job, or None."""
    root = Path(results) / "runs" / name / "train"
    done = sorted(root.glob("trial_*/DONE.json"))
    return done[-1].parent if done else None


def train_cmd(py, job, results, steps, smoke):
    cmd = [py, "-m", "scripts.train", "--no_curriculum",
           "--base_controller", job["base"], "--seed", job["seed"],
           "--total_steps", steps, "--control_hz", 200,
           "--utd", 1.0, "--history", job.get("history", 10), "--hidden", "256,256",
           "--thrust_filter_beta", job["beta"],
           "--moment_filter_beta", job.get("beta_m", 1.0),
           "--residual_authority", job["auth"],
           "--tau_pos", job["tau_pos"], "--fixed",
           "--runs_root", Path(results) / "runs" / job["name"], "--run_name", "train"]
    if smoke:
        cmd += ["--num_envs", 32, "--eval_episodes", 16, "--eval_every_iters", 5,
                "--buffer_size", 20000, "--learning_starts", 2000]
    else:
        cmd += ["--num_envs", 256, "--eval_episodes", 128, "--eval_every_iters", 10,
                "--buffer_size", 300000]
    return [str(c) for c in cmd]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="study_results")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--deadline", type=float, default=0.0,
                   help="unix time; 0 = no deadline")
    p.add_argument("--reserve_min", type=float, default=45.0,
                   help="minutes kept free before the deadline for eval + report")
    p.add_argument("--default_job_min", type=float, default=25.0)
    p.add_argument("--groups", default="A,B,C")
    p.add_argument("--max_jobs", type=int, default=0, help="0 = no limit")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    groups = set(a.groups.split(","))
    jobs = [j for j in job_list(seeds) if j["group"] in groups]
    if a.max_jobs > 0:
        jobs = jobs[:a.max_jobs]
    results = Path(a.results)
    logs = results / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (results / "jobs.json").write_text(json.dumps(jobs, indent=1))
    status_path = results / "queue_status.csv"
    py = sys.executable

    durations = []
    for j in jobs:
        r = finished_run(results, j["name"])
        if r is not None:
            try:
                durations.append(json.loads((r / "DONE.json").read_text())["wallclock_s"] / 60)
            except Exception:                                # noqa: BLE001
                pass

    status = []

    def save_status():
        with status_path.open("w") as f:
            f.write("name,group,seed,state,minutes,run_dir\n")
            for s in status:
                f.write(",".join(str(s[k]) for k in
                                 ("name", "group", "seed", "state", "minutes", "run_dir")) + "\n")

    n_todo = sum(finished_run(results, j["name"]) is None for j in jobs)
    print(f"{len(jobs)} jobs, {len(jobs) - n_todo} already finished, {n_todo} to run", flush=True)
    stop = False
    for i, j in enumerate(jobs):
        done = finished_run(results, j["name"])
        if done is not None:
            status.append({**j, "state": "done (earlier)", "minutes": "", "run_dir": done})
            continue
        est = statistics.median(durations) if durations else a.default_job_min
        if a.deadline > 0 and not stop:
            left_min = (a.deadline - time.time()) / 60 - a.reserve_min
            if left_min < 1.1 * est:
                print(f"deadline: {left_min:.0f} min left before the reserve, a job takes "
                      f"~{est:.0f} min; not starting more jobs", flush=True)
                stop = True
        if stop:
            status.append({**j, "state": "deferred", "minutes": "", "run_dir": ""})
            continue
        cmd = train_cmd(py, j, results, a.steps, a.smoke)
        log = logs / f"train_{j['name']}.log"
        print(f"[{i + 1}/{len(jobs)}] {time.strftime('%H:%M')} {j['name']}  "
              f"(est {est:.0f} min)  log: {log}", flush=True)
        t0 = time.time()
        timeout = max(3 * est, 60) * 60
        with log.open("w") as f:
            f.write(" ".join(cmd) + "\n\n")
            f.flush()
            try:
                rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        minutes = (time.time() - t0) / 60
        run = finished_run(results, j["name"])
        if run is not None:
            durations.append(minutes)
            state = "done"
        else:
            state = f"FAILED ({rc})"
            print(f"    FAILED ({rc}); last lines of {log}:", flush=True)
            print("".join(log.read_text().splitlines(True)[-15:]), flush=True)
        status.append({**j, "state": state, "minutes": f"{minutes:.1f}",
                       "run_dir": run or ""})
        save_status()
        print(f"    {state} in {minutes:.1f} min", flush=True)
    save_status()
    n_done = sum(s["state"].startswith("done") for s in status)
    print(f"queue finished: {n_done}/{len(jobs)} jobs done", flush=True)


if __name__ == "__main__":
    main()
