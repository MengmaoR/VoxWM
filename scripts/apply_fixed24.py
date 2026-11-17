"""Apply the opt-in timing patch to the project's fixed Craftium source."""
import argparse,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PIN='ca8b2762aea46d10a88c53f6411ff1f0dbdf3b49'

def apply(source):
    source=Path(source).resolve();patch=ROOT/'voxwm_gt/patches/fixed24.patch'
    capture=ROOT/'voxwm_gt/patches/region_capture.patch'
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=PIN:raise ValueError('Use the fixed Craftium revision; source was not changed')
    command=['git','-C',str(source),'apply']
    # The capture delta changes a base-patch hunk. Recognize the complete new
    # layer first, so existing v1 installs upgrade and repeated builds stay safe.
    if subprocess.run([*command,'--reverse','--check',str(capture)],capture_output=True).returncode:
        if subprocess.run([*command,'--reverse','--check',str(patch)],capture_output=True).returncode:
            subprocess.run([*command,'--check',str(patch)],check=True)
            subprocess.run([*command,str(patch)],check=True)
        subprocess.run([*command,'--check',str(capture)],check=True)
        subprocess.run([*command,str(capture)],check=True)
    # The previous macOS setup used a temporary toolchain. Restore that file only
    # when its recorded path disappeared, so an incremental build can reconfigure.
    cache=source/'CMakeCache.txt'
    if sys.platform=='darwin'and cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith('CMAKE_TOOLCHAIN_FILE:FILEPATH='):
                path=Path(line.split('=',1)[1])
                if not path.exists()and path.name=='macos-gl-png.cmake':
                    path.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(ROOT/'voxwm_gt/patches/macos-native.cmake',path)
    print('Opt-in fixed24 + integer-origin capture ready; timing and native rules are unchanged.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--craftium-dir',type=Path,default=ROOT/'third_party/craftium')
    apply(p.parse_args().craftium_dir)
