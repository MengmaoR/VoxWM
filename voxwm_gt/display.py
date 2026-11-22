"""A private, automatically numbered Xvfb for one software-rendered session."""
import os,select,shutil,subprocess,time
from pathlib import Path
from .runtime import InfrastructureError,stop_process


class SoftwareDisplay:
    def __init__(self,folder,timeout=15):
        self.folder=Path(folder);self.timeout=timeout;self.proc=None;self.log=None;self.name=None
    def start(self):
        binary=shutil.which('Xvfb')
        if not binary:
            raise InfrastructureError('Xvfb not found on PATH. Add its bin directory to PATH or install Xvfb.')
        read_fd,write_fd=os.pipe()
        self.log=open(self.folder/'xvfb.log','w+')
        try:
            args=[binary,'-displayfd',str(write_fd),'-screen','0','640x480x24','-nolisten','tcp','-noreset']
            xkb_dir=os.environ.get('X11_XKBDIR')
            if xkb_dir:
                args.extend(['-xkbdir',xkb_dir])
            self.proc=subprocess.Popen(args,
                pass_fds=(write_fd,),start_new_session=True,stdout=self.log,stderr=self.log)
            os.close(write_fd);write_fd=None
            deadline=time.monotonic()+self.timeout;number=b''
            while b'\n'not in number:
                if self.proc.poll()is not None:raise InfrastructureError('Private Xvfb exited before reporting a display')
                remaining=deadline-time.monotonic()
                if remaining<=0:raise InfrastructureError('Private Xvfb startup timed out')
                if select.select([read_fd],[],[],min(.2,remaining))[0]:
                    part=os.read(read_fd,64)
                    if not part:raise InfrastructureError('Private Xvfb closed its display-number pipe')
                    number+=part
            self.name=':'+str(int(number.strip()));return self.name
        except BaseException as exc:
            details=''
            if self.log:
                self.log.flush();self.log.seek(0);details=self.log.read()[-2000:]
            self.close()
            if isinstance(exc,InfrastructureError)and details:raise InfrastructureError(str(exc)+'\n'+details)from exc
            raise
        finally:
            os.close(read_fd)
            if write_fd is not None:os.close(write_fd)
    def close(self):
        try:stop_process(self.proc)
        finally:
            self.proc=None
            if self.log:self.log.close();self.log=None
