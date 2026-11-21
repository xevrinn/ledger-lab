#!/usr/bin/env python3
"""Run four deterministic policies on a shared seed/scenario set."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from audit_env import SCENARIOS, run_policy

POLICIES=("random","reconcile","flag_all","oracle")
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--out',default='artifacts/benchmark.json')
    parser.add_argument('--write-trajectories',action='store_true')
    args=parser.parse_args(); runs=[]; trajectories=[]
    for i in range(max(1,args.seeds)):
        # Same scenario and seed for each policy in a paired comparison.
        seed=8400+i; scenario=SCENARIOS[i%len(SCENARIOS)]
        for policy in POLICIES:
            result=run_policy(policy,seed,scenario)
            runs.append({'seed':seed,'scenario':scenario,'policy':policy,**result['metrics']})
            if args.write_trajectories: trajectories.extend(result['trace'])
    grouped={}
    for row in runs:grouped.setdefault(row['policy'],[]).append(row)
    summary={name:{key:round(statistics.mean(r[key] for r in rows),4) for key in ('precision','recall','f1','return','inspection_steps')} for name,rows in grouped.items()}
    output={'version':'0.3.0','paired_episodes':args.seeds,'runs':runs,'summary':summary}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    if args.write_trajectories:
        p=path.parent/'trajectories.jsonl';p.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in trajectories),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
