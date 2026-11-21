#!/usr/bin/env python3
"""Local HTTP bridge for the synthetic audit environment. Standard library only."""
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os
from urllib.parse import urlparse, parse_qs
import json
from audit_env import AuditEnv, VERSION, SCENARIOS, run_policy

ROOT = Path(__file__).resolve().parent
STATE = {"env": AuditEnv(), "seed": 42, "config": {"mode":"open"}, "trace": []}

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)
    def send_json(self, data, status=200):
        body=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(body)
    def read_json(self):
        length=int(self.headers.get('Content-Length','0'))
        if length>1_000_000: raise ValueError('payload too large')
        return json.loads(self.rfile.read(length) or b'{}')
    def do_GET(self):
        route=urlparse(self.path)
        if route.path=='/api/health': return self.send_json({'ok':True,'version':VERSION,'scenarios':list(SCENARIOS)})
        if route.path=='/api/state': return self.send_json(STATE['env']._initial_observation())
        if route.path=='/api/trajectory': return self.send_json(STATE['trace'])
        if route.path=='/api/benchmark':
            if STATE['config'].get('mode') == 'closed':
                return self.send_json({'error':'benchmark is disabled during closed evaluation'},403)
            query=parse_qs(route.query); seed=int(query.get('seed',[STATE['seed']])[0]); scenario=query.get('scenario',[None])[0]
            results=[]
            for policy in ('random','reconcile','flag_all','oracle'):
                run=run_policy(policy,seed,scenario)
                results.append({'policy':policy,**run['metrics']})
            return self.send_json({'seed_label':f'OPEN-{seed:04d}','results':results,'note':'Open demo benchmark; oracle is an evaluator ceiling.'})
        return super().do_GET()
    def do_POST(self):
        route=urlparse(self.path)
        try: payload=self.read_json()
        except Exception: return self.send_json({'error':'invalid JSON payload'},400)
        if route.path=='/api/reset':
            try:
                seed=int(payload.get('seed',42));scenario=payload.get('scenario')
                if scenario not in (None,*SCENARIOS): return self.send_json({'error':'unknown scenario'},400)
                mode='closed' if payload.get('mode')=='closed' else 'open'
                config={'mode':mode}
                if scenario:config['scenario']=scenario
                obs,_=STATE['env'].reset(seed,config);STATE['seed']=seed;STATE['config']=config;STATE['trace']=[]
                return self.send_json(obs)
            except Exception as e:return self.send_json({'error':str(e)},400)
        if route.path=='/api/step':
            obs,reward,terminated,truncated,info=STATE['env'].step(payload.get('action'))
            item={'episode_id':STATE['env'].episode['manifest']['episode_id'],'step':STATE['env'].steps,
                  'action':payload.get('action'),'observation':obs,'reward':reward,'terminated':terminated,'truncated':truncated}
            STATE['trace'].append(item)
            # Private evaluator diagnostics are never returned through this endpoint.
            return self.send_json({'observation':obs,'reward':reward,'terminated':terminated,'truncated':truncated,'info':{'schema_version':VERSION}})
        return self.send_json({'error':'not found'},404)
    def log_message(self, fmt, *args):
        if self.path.startswith('/api/'): return
        super().log_message(fmt,*args)

if __name__=='__main__':
    host=os.environ.get('AUDIT_HOST','127.0.0.1');port=int(os.environ.get('AUDIT_PORT','8000'))
    print(f'Ledger Lab {VERSION} at http://{host}:{port}')
    HTTPServer((host,port),Handler).serve_forever()
