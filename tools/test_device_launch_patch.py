"""Ensure other Quasar roles and third-party preload settings are preserved or rejected."""
import copy,json,unittest
from package_device import patch_launch,ROOT
class LaunchPatchTests(unittest.TestCase):
    def setUp(self):
        self.original=json.loads((ROOT/'profiles/plum/launch.json').read_text())
    def test_only_platform_environment_changes(self):
        before=copy.deepcopy(self.original)
        after=patch_launch(self.original)
        self.assertEqual(self.original,before)
        changed=[]
        for n,node in enumerate(before['nodes']):
            for t,task in enumerate(node['tasks']):
                other=after['nodes'][n]['tasks'][t]
                if task!=other:
                    changed.append(other['arg'])
                    restored=copy.deepcopy(other);restored['envs']=restored['envs'][:-2]
                    self.assertEqual(restored,task)
        self.assertEqual(changed,[['--platformd']])
    def test_existing_preload_is_rejected(self):
        for task in self.original['nodes'][0]['tasks']:
            if task['arg']==['--platformd']:task.setdefault('envs',[]).append('LD_PRELOAD=/data/other.so')
        with self.assertRaises(ValueError):patch_launch(self.original)
    def test_ambiguous_platform_is_rejected(self):
        platform=next(t for t in self.original['nodes'][0]['tasks'] if t['arg']==['--platformd'])
        self.original['nodes'][0]['tasks'].append(copy.deepcopy(platform))
        with self.assertRaises(ValueError):patch_launch(self.original)
if __name__=='__main__':unittest.main()
