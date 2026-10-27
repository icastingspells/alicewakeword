"""Exercise the actual activation shell script without a speaker or its logs."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHELL = (Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git/bin/bash.exe') if os.name == 'nt' else shutil.which('sh')


@unittest.skipUnless(SHELL and Path(SHELL).is_file(), 'POSIX shell unavailable')
class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.run_dir = self.root / 'run'; self.run_dir.mkdir()
        self.maind = self.root / 'maind'; self.maind.write_bytes(b'fixture firmware')
        self.profile = self.root / 'profile.env'
        self.profile.write_text("FIRMWARE_SHA256='" + hashlib.sha256(self.maind.read_bytes()).hexdigest() + "'\n")
        source = (ROOT / 'device/package/activate.sh').read_text()
        source = source.replace('/system/vendor/quasar/maind', '"$TEST_MAIND"').replace('/tmp/eva', '"$TEST_RUN"')
        self.script = self.root / 'activate.sh'; self.script.write_text(source, newline='\n')
        helper = self.root / 'eva-native'
        helper.write_text('''#!/bin/sh
case "$1" in
--mic-state) [ "$TEST_MIC" != unknown ] || exit 2; printf 'MICS_MUTED=%s\\n' "$TEST_MIC" ;;
--activate-cli) echo CLI_CALLED ;;
*) exit 2 ;;
esac
''', newline='\n')
        helper.chmod(0o700)

    def invoke(self, state):
        env = os.environ.copy()
        env.update(TEST_MIC=state, TEST_MAIND=self.maind.as_posix(), TEST_RUN=self.run_dir.as_posix())
        return subprocess.run([str(SHELL), self.script.as_posix()], env=env, capture_output=True, text=True, timeout=10)

    def test_missing_or_rotated_quasar_logs_do_not_block_activation(self):
        result = self.invoke('0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CLI_CALLED', result.stdout)
        self.assertFalse((self.run_dir / 'activation.lock').exists())

    def test_hardware_mute_blocks_activation(self):
        result = self.invoke('1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('reason=muted', result.stdout)
        self.assertNotIn('CLI_CALLED', result.stdout)

    def test_unknown_hardware_state_blocks_activation(self):
        result = self.invoke('unknown')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('reason=mic_state_unavailable', result.stdout)
        self.assertNotIn('CLI_CALLED', result.stdout)

    def test_existing_activation_lock_blocks_second_request(self):
        (self.run_dir / 'activation.lock').mkdir()
        result = self.invoke('0')
        self.assertIn('reason=locked', result.stdout)
        self.assertNotIn('CLI_CALLED', result.stdout)

    def test_changed_firmware_is_rejected(self):
        self.maind.write_bytes(b'different firmware')
        result = self.invoke('0')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('CLI_CALLED', result.stdout)


if __name__ == '__main__':
    unittest.main()
