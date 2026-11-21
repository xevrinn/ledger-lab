#!/usr/bin/env python3
"""Build 10 open training + 10 open dev episodes, evaluator truth, and trajectories."""
from __future__ import annotations
import json
from pathlib import Path
from audit_env import SCENARIOS, VERSION, generate_episode, run_policy

def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows),encoding='utf-8')

def main():
    trajectories=[]
    for split,first_seed in [('train',1100),('dev',2200)]:
        episodes=[]; truths=[]; provenances=[]
        for i in range(10):
            seed=first_seed+i; scenario=SCENARIOS[i%len(SCENARIOS)]
            episode,truth,provenance=generate_episode(seed,{'scenario':scenario})
            episodes.append(episode);truths.append(truth);provenances.append(provenance)
            run=run_policy('reconcile',seed,scenario)
            trajectories.extend(run['trace'])
        write_jsonl(Path(f'{split}_episodes.jsonl'),episodes)
        write_jsonl(Path(f'{split}_truth.jsonl'),truths)
        write_jsonl(Path(f'{split}_provenance.jsonl'),provenances)
    write_jsonl(Path('trajectories.jsonl'),trajectories)
    Path('dataset_manifest.json').write_text(json.dumps({'schema_version':'1.0','environment_version':VERSION,'train_episodes':10,'dev_episodes':10,'trajectory_count':len(trajectories),'note':'All records are synthetic. truth/provenance files are evaluator-side and never returned by AuditEnv.step.'},indent=2),encoding='utf-8')
    print('Wrote 10 train episodes, 10 dev episodes, evaluator references, and baseline trajectories.')
if __name__=='__main__':main()
