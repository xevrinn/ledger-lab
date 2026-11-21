"""Attack-oriented contract checks for the evaluator and environment."""
import unittest
from audit_env import AuditEnv, SCENARIOS, generate_episode, oracle_findings, score_findings

class AuditEnvironmentTests(unittest.TestCase):
    def test_seed_and_actions_reproduce_exact_episode(self):
        a=generate_episode(8101,{'scenario':'interacting_fx_allocation'})
        b=generate_episode(8101,{'scenario':'interacting_fx_allocation'})
        self.assertEqual(a,b)
    def test_clean_silence_beats_unsupported_accusations(self):
        _,truth,_=generate_episode(8102,{'scenario':'clean'})
        silence=score_findings([],truth,0)
        false=[{'type':'duplicate_invoice','affected_ids':['fake'],'claimed_amount_minor':10,'currency':'KZT','evidence_ids':['fake']}]
        accusation=score_findings(false,truth,0)
        self.assertGreater(silence['reward'],accusation['reward'])
    def test_duplicates_never_create_extra_true_positives(self):
        _,truth,_=generate_episode(8103,{'scenario':'duplicate_invoice'})
        one=oracle_findings(truth); two=one+one
        self.assertEqual(score_findings(one,truth,0)['tp'],1)
        self.assertEqual(score_findings(two,truth,0)['tp'],1)
        self.assertEqual(score_findings(two,truth,0)['fp'],1)
    def test_missing_evidence_is_not_a_match(self):
        _,truth,_=generate_episode(8104,{'scenario':'period_cutoff'})
        finding=oracle_findings(truth)[0];finding['evidence_ids']=[]
        self.assertEqual(score_findings([finding],truth,0)['tp'],0)
    def test_closed_mode_withholds_score_from_policy(self):
        env=AuditEnv();env.reset(8105,{'scenario':'duplicate_invoice','mode':'closed'})
        obs,reward,done,truncated,info=env.step({'action':'submit','findings':oracle_findings(env._truth)})
        self.assertEqual(reward,0);self.assertNotIn('metrics',obs);self.assertNotIn('evaluator',info)
        self.assertIn('metrics',env.evaluator_result())
    def test_every_scenario_is_recognized(self):
        for scenario in SCENARIOS:
            episode,truth,_=generate_episode(8106,{'scenario':scenario})
            self.assertEqual(episode['manifest']['ledger_rows']%2,0)
            self.assertLessEqual(len(truth['issues']),30)

if __name__=='__main__':unittest.main()
