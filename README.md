# Ledger Lab · RL Financial Audit Environment

**Genesis Hackathon · FinTech / 2026**

A reproducible, offline-first sandbox where an agent inspects synthetic company records, cites evidence, and submits audit findings. This is an environment for evaluating audit policies; it is not a banking product, an audit opinion, or a trained model.

## Quick start

```bash
python3 server.py
```

Open <http://127.0.0.1:8000>. Requires Python 3.11+ and no third-party packages. For the environment API directly:

```python
from audit_env import AuditEnv

env = AuditEnv()
observation, info = env.reset(seed=42, config={"scenario": "interacting_fx_allocation", "mode": "open"})
page, reward, terminated, truncated, info = env.step({"action": "list_records", "table": "ledger", "cursor": 0, "limit": 20})
```

The hosted GitHub Pages build is a **public open demo** with a client-side sample generator. The actual Python API and evaluator isolation are exercised by running the app locally or using `audit_env.py` directly. GitHub Pages cannot host the Python backend. On Pages, entries are not persisted server-side.

## Run a paired baseline benchmark

```bash
python3 benchmark.py --seeds 20 --out artifacts/benchmark.json --write-trajectories
```

Policies share each seed and scenario: `random` is a fixed-seed explorer, `reconcile` conservatively checks duplicate invoice identities and payment allocation, `flag_all` attacks the scorer with unsupported claims, and `oracle` supplies generated truth only to measure the evaluator ceiling. The oracle is not evidence of agent capability.

## Generate open train/dev examples

```bash
python3 generate_data.py
```

Creates `train_episodes.jsonl` and `dev_episodes.jsonl` (10 each), separate `train_truth.jsonl` / `dev_truth.jsonl` and matching provenance files, plus `trajectories.jsonl` and `dataset_manifest.json`. Those references never appear in `AuditEnv.step()` observations. All dataset files use synthetic companies, transactions, and documents only.

## Environment contract

`AuditEnv.reset(seed, config)` returns the task, manifest, policy, action names, and remaining budget. The seed is not returned. `step(action)` returns JSON-compatible observations and follows a 100-step budget.

Actions:

- `{"action":"list_records","table":"ledger|invoices|payments|fx","cursor":0,"limit":20}`
- `{"action":"get_record","id":"..."}`
- `{"action":"get_document","id":"..."}`
- `{"action":"calculator","expression":"1200 / 3"}` — numeric arithmetic only; no Python execution, files, shell, SQL, or network
- `{"action":"submit","findings":[...]}`

Tables are read-only; pages are capped at 50 rows. Invalid actions return a bounded error and consume a step. The schema, tables, limit, and terminal behavior are covered in `audit_env.py` and `test_audit_env.py`.

A finding contains `type`, canonical `affected_ids`, integer `claimed_amount_minor`, `currency`, `evidence_ids`, and a short `explanation`. The matcher requires an exact type/ID/currency match, amount within one minor unit, and at least one complete allowed evidence set. Maximum one-to-one matching means duplicate claims remain false positives. Prose does not affect reward.

The shared reward is:

```text
precision = TP / (TP + FP)                 # 0 when denominator is 0
recall    = TP / (TP + FN)                 # 0 when denominator is 0
F1        = 2*TP / (2*TP + FP + FN)        # 1 only when truth and findings are both empty
return    = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1)
```

Open mode returns the terminal score for training. Closed mode returns zero to the policy and withholds metrics from both observation and `info`. The evaluator-only result is accessible after submission through `evaluator_result()`; the HTTP server never serializes the private truth or provenance. Benchmark access is disabled while the server is in closed mode. For formal evaluation, keep the evaluator in a separate process and use unseen seeds/configurations.

## Synthetic scenario coverage

- Duplicate invoice identity, with a second source document and sufficient document evidence.
- Missing or partial payment allocation, including a foreign-currency interaction.
- FX rate-date mismatch with a recorded causal amount change and daily-rate evidence.
- Reporting-period cutoff with a posting moved beyond the inclusive period boundary.
- Two interactions: FX × allocation and cutoff × duplicate.
- Clean books include repeated values and split payments as harmless distractors.

Journal entries use balanced debit/credit pairs. Dates are ISO-8601 with offsets; amounts are integer minor units; FX rates are decimal strings. `provenance` records causal edits separately from the agent-facing episode.

## Docker

```bash
docker build -t ledger-lab .
docker run --rm -p 8000:8000 ledger-lab
```

The container binds its service for the explicitly mapped local demo port. There are no paid APIs, model weights, GPU dependencies, or outbound service calls in the environment engine.

## Scope and limitations

- Every company, person, invoice, account, and amount is synthetic.
- This is not a professional audit conclusion and must not be used for real financial data.
- The baseline is intentionally conservative and incomplete; no learning or model-quality improvement is claimed.
- The browser preview is an open-mode demonstration. Do not expose the evaluator process or use real records.

See [DESIGN.md](DESIGN.md) for the design note and visual system, and [LICENSE](LICENSE) for the MIT license.
