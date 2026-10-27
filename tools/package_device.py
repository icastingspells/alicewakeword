"""Build a firmware-pinned, reversible BusyBox deployment package; never connect to a device."""
from pathlib import Path
import argparse,copy,gzip,hashlib,io,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'analysis/device-build'

def digest(data):return hashlib.sha256(data).hexdigest()

def release_id(files,profile):
    identity={'files':{name:digest(data) for name,data in files.items()},'profile':profile}
    return digest(json.dumps(identity,sort_keys=True,separators=(',',':')).encode())[:16]

def patch_launch(original):
    config=copy.deepcopy(original);matches=[]
    for node in config.get('nodes',[]):
        for task in node.get('tasks',[]):
            if task.get('path')=='/system/vendor/quasar/maind' and task.get('arg')==['--platformd']:
                matches.append(task)
    if len(matches)!=1:raise ValueError('Exactly one platformd task is required')
    env=matches[0].setdefault('envs',[])
    if not isinstance(env,list) or not all(isinstance(v,str) for v in env):raise ValueError('Unexpected task env format')
    if any(v.split('=',1)[0] in {'LD_PRELOAD','EVA_SOCKET'} for v in env):
        raise ValueError('Existing preload/EVA environment must be reviewed instead of overwritten')
    env.extend(['LD_PRELOAD=/data/eva/current/libeva_tap.so','EVA_SOCKET=/tmp/eva/pcm.sock'])
    return config

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',type=Path,default=ROOT/'profiles/plum/profile.json')
    parser.add_argument('--launch',type=Path)
    parser.add_argument('--native',type=Path,default=BUILD/'eva-native')
    args=parser.parse_args()
    selected=json.loads(args.profile.read_text())
    launch=args.launch or args.profile.parent/selected['launch']
    files={p.name:p.read_bytes().replace(b'\r\n',b'\n') for p in (ROOT/'device/package').iterdir() if p.is_file()}
    files['eva-native']=args.native.read_bytes()
    files['libeva_tap.so']=(BUILD/'libeva_tap.so').read_bytes()
    files['LICENSE.openwakeword']=(ROOT/'.cache/eva-engine/LICENSE').read_bytes()
    files['LICENSE']=(ROOT/'LICENSE').read_bytes()
    files['NOTICE']=b'Original detector and installer code: MIT; see LICENSE.\nIncludes AOT exports of OpenWakeWord v0.5.1 frontend weights and a user-selected classifier.\nAudioFeatures reference: dscripka/openWakeWord commit 368c03716d1e92591906a84949bc477f3a834455.\nCopyright 2022 David Scripka; Apache-2.0 license included.\nSelected classifier weights retain their own licensing terms.\n'
    original=launch.read_bytes()
    if not args.launch and digest(original)!=selected['base_launch_sha256']:
        raise ValueError('Profile launch checksum mismatch')
    files['launch.json']=(json.dumps(patch_launch(json.loads(original)),indent=2)+'\n').encode()
    files['models.json']=(BUILD/'models.json').read_bytes()
    profile={'FIRMWARE_SHA256':selected['firmware_sha256'],'BASE_LAUNCH_SHA256':digest(original),
             'PATCHED_LAUNCH_SHA256':digest(files['launch.json']),'MIN_AVAILABLE_KB':'32768','MIN_DISK_KB':'12000'}
    release=release_id(files,profile)
    profile={'RELEASE_ID':release,**profile}
    files['profile.env']=''.join(f"{key}='{value}'\n" for key,value in profile.items()).encode()
    files['SHA256SUMS']=''.join(digest(files[name])+'  '+name+'\n' for name in sorted(files)).encode()
    stage=BUILD/'package'
    stage.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():(stage/name).write_bytes(data)
    dist=ROOT/'dist';dist.mkdir(exist_ok=True)
    target=dist/f'eva-plum-{release}.tar.gz'
    with target.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as archive:
                for name in sorted(files):
                    item=tarfile.TarInfo('eva-plum/'+name);item.size=len(files[name]);item.mtime=0
                    item.mode=0o700 if name in {'eva-native','evactl','S98eva'} or name.endswith('.sh') else 0o600
                    archive.addfile(item,io.BytesIO(files[name]))
    report={'archive':str(target.relative_to(ROOT)),'archive_bytes':target.stat().st_size,'archive_sha256':digest(target.read_bytes()),
            'payload_bytes':sum(map(len,files.values())),'firmware_profile':profile,
            'default_mode':'observe','installs_python':False,'modifies_vendor_binary':False,'installs_readiness_patch':False,
            'weights':json.loads(files['models.json']),'files':{name:{'bytes':len(data),'sha256':digest(data)} for name,data in files.items()}}
    (BUILD/'package.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['archive','archive_bytes','archive_sha256','payload_bytes','default_mode']},indent=2))

if __name__=='__main__':main()
