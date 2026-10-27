"""Public workflows; standard library only except during model compilation."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, io, json, os, platform, re, shutil, subprocess, sys
import tarfile, tempfile, urllib.request, venv, zipfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'analysis/device-build'
CACHE = ROOT / '.cache'
FILES = set('eva-native libeva_tap.so evactl preflight.sh activate.sh profile.env launch.json S98eva SHA256SUMS config.env install.sh uninstall.sh models.json LICENSE LICENSE.openwakeword NOTICE'.split())
CONTROL = '/data/eva/current/evactl'


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'alicewakeword'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def run(argv, **kwargs):
    return subprocess.run([str(a) for a in argv], check=True, **kwargs)


def build_python():
    path = CACHE / 'eva-venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not path.exists():
        raise ValueError('Run: python alicewakeword.py setup')
    return path


def zig_binary():
    configured = os.environ.get('ALICEWAKEWORD_ZIG')
    if configured:
        return configured
    candidates = list((CACHE / 'zig').glob('*/zig.exe')) + list((CACHE / 'zig').glob('*/zig'))
    pinned = [path for path in candidates if '0.15.2' in path.parent.name]
    if pinned:
        return pinned[0]
    if shutil.which('zig'):
        return shutil.which('zig')
    raise ValueError('Zig 0.15.2 not found. Run setup, or set ALICEWAKEWORD_ZIG.')


def setup():
    destination = CACHE / 'eva-venv'
    interpreter = destination / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not interpreter.exists():
        venv.EnvBuilder(with_pip=True).create(destination)
    run([interpreter, '-m', 'pip', 'install', '-r', ROOT / 'requirements-build.txt'])
    try:
        if subprocess.check_output([str(zig_binary()), 'version'], text=True).strip() != '0.15.2':
            raise ValueError('Zig version mismatch')
    except ValueError:
        if os.environ.get('ALICEWAKEWORD_ZIG'):
            raise ValueError('ALICEWAKEWORD_ZIG must select Zig 0.15.2.')
        machine = platform.machine().lower()
        arch = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine)
        system = {'win32': 'windows', 'linux': 'linux', 'darwin': 'macos'}[sys.platform]
        index = json.loads(fetch('https://ziglang.org/download/index.json'))
        entry = index['0.15.2'][f'{arch}-{system}']
        if not entry['tarball'].startswith('https://ziglang.org/'):
            raise ValueError('Unexpected Zig download host')
        data = fetch(entry['tarball'])
        if checksum(data) != entry['shasum']:
            raise ValueError('Zig checksum mismatch')
        folder = CACHE / 'zig'; folder.mkdir(parents=True, exist_ok=True)
        if entry['tarball'].endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.namelist():
                    if not (folder / member).resolve().is_relative_to(folder.resolve()):
                        raise ValueError('Unsafe ZIP path')
                archive.extractall(folder)
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as archive:
                archive.extractall(folder, filter='data')
    print('Build environment ready.')


def frontend():
    assets = json.loads((ROOT / 'profiles/openwakeword.json').read_text())
    folder = CACHE / 'eva-engine'; folder.mkdir(parents=True, exist_ok=True)
    for name, item in assets.items():
        path = folder / name
        data = path.read_bytes() if path.exists() else fetch(item['url'])
        if checksum(data) != item['sha256']:
            raise ValueError('Frontend checksum mismatch: ' + name)
        path.write_bytes(data)


def build(args):
    model = args.model.resolve()
    if model.suffix.lower() != '.tflite' or not model.is_file():
        raise ValueError('Select an existing .tflite classifier.')
    py = build_python(); zig = zig_binary(); frontend()
    if subprocess.check_output([str(zig), 'version'], text=True).strip() != '0.15.2':
        raise ValueError('This build is pinned to Zig 0.15.2.')
    env = os.environ.copy()
    env['ZIG_GLOBAL_CACHE_DIR'] = str(CACHE / 'zig-global')
    env['ZIG_LOCAL_CACHE_DIR'] = str(CACHE / 'zig-local')
    run([py, ROOT / 'tools/compile_device_models.py', '--model', model, '--threshold', str(args.threshold)], cwd=ROOT)
    common = ['-O3', '-ffast-math', '-fno-finite-math-only', '-fno-strict-aliasing', '-Idevice/native', '-Ianalysis/device-build']
    run([zig, 'cc', '-target', 'aarch64-linux-musl', '-mcpu=cortex_a53', *common, '-static', '-s', '-pthread', 'device/native/engine.c', 'analysis/device-build/models.c', '-lm', '-o', BUILD / 'eva-native-aarch64'], cwd=ROOT, env=env)
    run([zig, 'cc', '-target', 'arm-linux-gnueabihf.2.30', '-mcpu=cortex_a53', '-Os', '-s', '-fPIC', '-shared', 'device/native/alsa_tap.c', '-ldl', '-o', BUILD / 'libeva_tap.so'], cwd=ROOT, env=env)
    library = BUILD / ('models.dll' if os.name == 'nt' else 'libmodels.so')
    host_target = ['-target', 'x86_64-windows-gnu'] if os.name == 'nt' else []
    run([zig, 'cc', *host_target, *common, '-shared', '-fPIC', 'analysis/device-build/models.c', '-lm', '-o', library], cwd=ROOT, env=env)
    run([py, ROOT / 'tools/verify_device_models.py', '--model', model], cwd=ROOT)
    run([py, ROOT / 'tools/verify_device_stream.py'], cwd=ROOT)
    run([py, ROOT / 'tools/package_device.py', '--native', BUILD / 'eva-native-aarch64', '--profile', args.profile.resolve()], cwd=ROOT)
    metadata = json.loads((BUILD / 'package.json').read_text())
    destination = ROOT / 'dist/alicewakeword-plum.tar.gz'
    shutil.copyfile(ROOT / metadata['archive'], destination)
    (destination.parent / 'alicewakeword-plum.sha256').write_text(checksum(destination.read_bytes()) + '  alicewakeword-plum.tar.gz\n')
    shutil.copyfile(BUILD / 'package.json', destination.parent / 'package.json')
    print('READY:', destination)
    print('Model SHA256:', metadata['weights']['models']['eva']['sha256'])
    print('New models require a new on-device measurement before activation.')


def read_package(path):
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('Package exceeds 32 MiB')
    files = {}; total = 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if member.isdir() and parts == ('eva-plum',):
                continue
            if len(parts) != 2 or parts[0] != 'eva-plum' or parts[1] not in FILES or not member.isfile():
                raise ValueError('Unexpected archive entry: ' + member.name)
            if parts[1] in files:
                raise ValueError('Duplicate package entry')
            total += member.size
            if total > 32 * 1024 * 1024:
                raise ValueError('Expanded package exceeds 32 MiB')
            files[parts[1]] = archive.extractfile(member).read()
    if set(files) != FILES:
        raise ValueError('Incomplete package')
    hashes = {}
    for line in files['SHA256SUMS'].decode('ascii').splitlines():
        digest, name = line.split()
        if name in hashes or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Invalid checksum manifest')
        hashes[name] = digest
    if set(hashes) != FILES - {'SHA256SUMS'} or any(checksum(files[n]) != h for n, h in hashes.items()):
        raise ValueError('Package checksum mismatch')
    match = re.search(rb"^RELEASE_ID='([0-9a-f]{16})'$", files['profile.env'], re.M)
    if not match:
        raise ValueError('Invalid release ID')
    return files, match[1].decode()


class Remote:
    def __init__(self, args):
        self.adb = args.adb
        if args.host:
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', args.host):
                raise ValueError('Invalid host')
            self.command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-p', str(args.port), 'root@' + args.host]
        else:
            self.command = [args.adb_path, '-s', args.adb]

    def shell(self, command):
        if not self.adb:
            run(self.command + [command], timeout=120)
            return
        marker = 'ALICEWAKEWORD_EXIT'
        wrapped = f'( {command} ); code=$?; printf "\\n{marker}=%s\\n" "$code"'
        result = subprocess.run(self.command + ['shell', wrapped], capture_output=True, text=True, timeout=120)
        output = result.stdout.replace('\r', '')
        print(re.sub(r'(?m)^ALICEWAKEWORD_EXIT=\d+\s*$', '', output), end='')
        if result.returncode or not re.search(r'(?m)^ALICEWAKEWORD_EXIT=0\s*$', output):
            raise ValueError('ADB shell operation failed: ' + result.stderr.strip())

    def push(self, local, remote):
        if self.adb:
            run(self.command + ['push', str(local), remote], timeout=120)
        else:
            run(self.command + [f"umask 077; test ! -L '{remote}' && cat > '{remote}'"], input=local.read_bytes(), timeout=120)

    def reboot(self):
        self.shell('sync')
        result = subprocess.run(self.command + (['shell', '/sbin/reboot'] if self.adb else ['/sbin/reboot']), timeout=20)
        if result.returncode not in (0, 255):
            raise ValueError('Reboot request failed')
        print('Reboot requested. Reconnect before activating.')


def install(args):
    path = args.package
    if path is None:
        repo = json.loads((ROOT / 'project.json').read_text())['repository']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
            raise ValueError('Invalid repository')
        folder = CACHE / 'downloads'; folder.mkdir(parents=True, exist_ok=True)
        url = f'https://github.com/{repo}/releases/latest/download/'
        data = fetch(url + 'alicewakeword-plum.tar.gz')
        expected = fetch(url + 'alicewakeword-plum.sha256').decode().split()[0]
        if checksum(data) != expected:
            raise ValueError('Release checksum mismatch')
        path = folder / 'alicewakeword-plum.tar.gz'; path.write_bytes(data)
    files, release = read_package(path)
    remote = Remote(args); stage = '/tmp/eva-upload-' + release
    remote.shell(f"umask 077; test ! -L '{stage}' && mkdir -p '{stage}' && chmod 700 '{stage}'")
    with tempfile.TemporaryDirectory() as temp:
        for name in ['preflight.sh', 'profile.env']:
            local = Path(temp) / name; local.write_bytes(files[name]); remote.push(local, stage + '/' + name)
    remote.shell(f"sh '{stage}/preflight.sh'")
    if args.check:
        remote.shell(f"rm -f '{stage}/preflight.sh' '{stage}/profile.env'; rmdir '{stage}'")
        return
    incoming = '/data/eva-upload-' + release + '.tar.gz'; remote.push(path, incoming)
    cleanup = ' '.join(f"'{stage}/eva-plum/{n}'" for n in sorted(FILES))
    digest = checksum(path.read_bytes())
    remote.shell(f"set -e; printf '%s  %s\\n' '{digest}' '{incoming}' | sha256sum -c -; gzip -dc '{incoming}' | tar -xf - -C '{stage}'; sh '{stage}/eva-plum/install.sh'; rm -f '{incoming}' {cleanup}; rmdir '{stage}/eva-plum'; rm -f '{stage}/preflight.sh' '{stage}/profile.env'; rmdir '{stage}'")
    print('Installed in observe mode. First installation needs a reboot to attach the audio tap.')
    if args.reboot:
        remote.reboot()


def main():
    parser = argparse.ArgumentParser(description='AliceWakeWord: install, run and build custom openWakeWord models.')
    commands = parser.add_subparsers(dest='action', required=True)
    commands.add_parser('setup', help='Prepare Python build environment and Zig 0.15.2')
    item = commands.add_parser('build', help='Compile a compatible .tflite classifier and verify its output')
    item.add_argument('--model', type=Path, required=True)
    item.add_argument('--threshold', type=float, default=0.5)
    item.add_argument('--profile', type=Path, default=ROOT / 'profiles/plum/profile.json')
    for action in ['install', 'status', 'start', 'stop', 'activate', 'observe', 'measure', 'disable', 'enable', 'uninstall', 'reboot']:
        item = commands.add_parser(action)
        connection = item.add_mutually_exclusive_group(required=True)
        connection.add_argument('--host', help='Root SSH host, key-based authentication')
        connection.add_argument('--adb', help='Connected ADB serial, e.g. HOST:5555')
        item.add_argument('--port', type=int, choices=range(1, 65536), default=2222, metavar='PORT')
        item.add_argument('--adb-path', default='adb')
        if action == 'install':
            item.add_argument('--package', type=Path, help='Local archive; otherwise download latest GitHub release')
            item.add_argument('--check', action='store_true', help='Only preflight, no persistent installation')
            item.add_argument('--reboot', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'setup': setup()
        elif args.action == 'build': build(args)
        elif args.action == 'install':
            if args.check and args.reboot: raise ValueError('--check cannot reboot')
            install(args)
        elif args.action == 'reboot': Remote(args).reboot()
        else:
            command = {
                'status': f'{CONTROL} status', 'start': f'{CONTROL} start', 'stop': f'{CONTROL} stop',
                'observe': f'{CONTROL} mode observe', 'measure': f'{CONTROL} measure',
                'activate': f'{CONTROL} mode observe && {CONTROL} measure && {CONTROL} mode active',
                'disable': f'{CONTROL} disable', 'enable': f'{CONTROL} enable', 'uninstall': 'sh /data/eva/current/uninstall.sh',
            }[args.action]
            Remote(args).shell(command)
    except (ValueError, OSError, subprocess.SubprocessError, KeyError) as error:
        parser.exit(1, 'Error: ' + str(error) + '\n')
