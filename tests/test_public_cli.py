import argparse
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch, Mock

from tools import public_cli as cli
from tools.package_device import release_id
from tools import package_device


def package(path, changed=None, extra=None):
    files = {name: b'fixture' for name in cli.FILES - {'SHA256SUMS'}}
    files['profile.env'] = b"RELEASE_ID='0123456789abcdef'\n"
    files['SHA256SUMS'] = ''.join(cli.checksum(value) + '  ' + name + '\n' for name, value in sorted(files.items())).encode()
    if changed:
        files.update(changed)
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in files.items():
            member = tarfile.TarInfo('eva-plum/' + name); member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        if extra:
            archive.addfile(extra)


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'package.tar.gz'

    def test_valid_package_and_release(self):
        package(self.path)
        files, release = cli.read_package(self.path)
        self.assertEqual(release, '0123456789abcdef')
        self.assertEqual(set(files), cli.FILES)

    def test_changed_binary_is_rejected(self):
        package(self.path, {'eva-native': b'changed'})
        with self.assertRaisesRegex(ValueError, 'checksum'):
            cli.read_package(self.path)

    def test_parent_traversal_is_rejected(self):
        package(self.path, extra=tarfile.TarInfo('../escape'))
        with self.assertRaisesRegex(ValueError, 'entry'):
            cli.read_package(self.path)

    def test_symlink_is_rejected(self):
        link = tarfile.TarInfo('eva-plum/extra'); link.type = tarfile.SYMTYPE; link.linkname = '/etc/passwd'
        package(self.path, extra=link)
        with self.assertRaisesRegex(ValueError, 'entry'):
            cli.read_package(self.path)

    def test_duplicate_entry_is_rejected(self):
        package(self.path, extra=tarfile.TarInfo('eva-plum/eva-native'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            cli.read_package(self.path)

    def test_missing_file_is_rejected(self):
        with tarfile.open(self.path, 'w:gz'):
            pass
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            cli.read_package(self.path)

    def test_preflight_does_not_install_or_reboot(self):
        package(self.path)
        args = argparse.Namespace(package=self.path, check=True, reboot=False)
        with patch.object(cli, 'Remote') as remote:
            cli.install(args)
        commands = [call.args[0] for call in remote.return_value.shell.call_args_list]
        self.assertTrue(any('/preflight.sh' in command for command in commands))
        self.assertFalse(any('/install.sh' in command for command in commands))
        remote.return_value.reboot.assert_not_called()


class ReleaseIdentityTests(unittest.TestCase):
    def test_classifier_metadata_changes_release(self):
        before = release_id({'models.json': b'threshold=0.5'}, {'firmware': 'same'})
        after = release_id({'models.json': b'threshold=0.7'}, {'firmware': 'same'})
        self.assertNotEqual(before, after)

    def test_firmware_changes_release(self):
        files = {'native': b'same'}
        self.assertNotEqual(release_id(files, {'firmware': 'one'}), release_id(files, {'firmware': 'two'}))

    def test_file_order_does_not_change_release(self):
        self.assertEqual(release_id({'one': b'1', 'two': b'2'}, {}), release_id({'two': b'2', 'one': b'1'}, {}))


class RemoteTests(unittest.TestCase):
    def test_host_shell_injection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'host'):
            cli.Remote(argparse.Namespace(host='host;id', adb=None))

    def test_old_adb_masked_exit_status_is_rejected(self):
        args = argparse.Namespace(host=None, adb='device', adb_path='adb')
        with patch.object(cli.subprocess, 'run', return_value=Mock(returncode=0, stdout='ALICEWAKEWORD_EXIT=1\n', stderr='')):
            with self.assertRaisesRegex(ValueError, 'ADB shell'):
                cli.Remote(args).shell('false')

    def test_adb_missing_marker_is_rejected(self):
        args = argparse.Namespace(host=None, adb='device', adb_path='adb')
        with patch.object(cli.subprocess, 'run', return_value=Mock(returncode=0, stdout='output only\n', stderr='')):
            with self.assertRaisesRegex(ValueError, 'ADB shell'):
                cli.Remote(args).shell('false')


class ProfileTests(unittest.TestCase):
    def test_public_launch_has_expected_bytes(self):
        import json
        profile = json.loads((cli.ROOT / 'profiles/plum/profile.json').read_text())
        self.assertEqual(cli.checksum((cli.ROOT / 'profiles/plum/launch.json').read_bytes()), profile['base_launch_sha256'])

    def test_package_needs_only_selected_native_binary(self):
        import json, shutil
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); build = root / 'analysis/device-build'
            build.mkdir(parents=True)
            (root / 'LICENSE').write_text('fixture license')
            shutil.copytree(cli.ROOT / 'device/package', root / 'device/package')
            license_path = root / '.cache/eva-engine/LICENSE'
            license_path.parent.mkdir(parents=True); license_path.write_text('fixture license')
            (build / 'candidate64').write_bytes(b'candidate')
            (build / 'libeva_tap.so').write_bytes(b'hook')
            (build / 'models.json').write_text('{}')
            argv = ['package_device', '--native', str(build / 'candidate64'), '--profile', str(cli.ROOT / 'profiles/plum/profile.json')]
            with patch.object(package_device, 'ROOT', root), patch.object(package_device, 'BUILD', build), patch('sys.argv', argv), patch('sys.stdout', new_callable=io.StringIO):
                package_device.main()
            self.assertFalse((build / 'eva-native').exists())
            manifest = json.loads((build / 'package.json').read_text())
            self.assertEqual(manifest['files']['eva-native']['sha256'], cli.checksum(b'candidate'))


if __name__ == '__main__':
    unittest.main()
