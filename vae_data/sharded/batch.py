"""One in-memory native RAW batch: full first tick, lossless sparse later ticks."""
import os,socket,struct,time
import numpy as np
from .clock import RAW

VERSION='voxwm-advance-batch-v1'
CELLS=27*4096
DELTA=np.dtype([('index','<u4'),('raw',RAW)])
CLOCK=np.dtype([('tick','<u8'),('dt','<f4')])
MAX_BYTES=12+CELLS*4+47*(4+CELLS*8)+48*CLOCK.itemsize


def receive(channel,deadline):
    # Craftium's fixed-size receive consumes only its ordinary final observation;
    # the native extension then sends this length-prefixed payload, before waiting
    # for the NEXT action. Duplicate the fd so ownership stays with Craftium.
    blocking=os.get_blocking(channel.connfd)
    try:
        with socket.socket(fileno=os.dup(channel.connfd))as stream:
            def exact(size):
                data=bytearray(size);view=memoryview(data);at=0
                while at<size:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise TimeoutError('Native batch transfer exceeded wall budget')
                    stream.settimeout(remaining)
                    n=stream.recv_into(view[at:])
                    if not n:raise ConnectionError('Native batch socket closed before completion')
                    at+=n
                return data
            length=struct.unpack('<Q',exact(8))[0]
            if not 12<=length<=MAX_BYTES:raise ValueError('Invalid native batch byte length')
            return exact(length)
    finally:os.set_blocking(channel.connfd,blocking)


class Batch:
    def __init__(self,payload,requested):
        self.payload=payload;self.offsets=[]
        if len(payload)<12 or payload[:4]!=b'VWB1':raise ValueError('Unknown native batch encoding')
        self.count,cells=struct.unpack_from('<II',payload,4)
        if not 1<=self.count<=requested<=48 or cells!=CELLS:raise ValueError('Invalid native batch capture count/shape')
        at=12+CELLS*RAW.itemsize
        if at>len(payload):raise ValueError('Truncated native first state')
        for _ in range(1,self.count):
            if at+4>len(payload):raise ValueError('Truncated native delta header')
            count=struct.unpack_from('<I',payload,at)[0];at+=4
            if count>CELLS or at+count*DELTA.itemsize>len(payload):raise ValueError('Invalid native delta length')
            changes=np.frombuffer(payload,DELTA,count,at);indices=changes['index']
            if count and(indices[-1]>=CELLS or np.any(indices[1:]<=indices[:-1])):raise ValueError('Invalid native delta indices')
            self.offsets.append((at,count));at+=count*DELTA.itemsize
        if at+self.count*CLOCK.itemsize!=len(payload):raise ValueError('Missing native client clocks/trailing bytes')
        self.clocks=np.frombuffer(payload,CLOCK,self.count,at)
    def states(self):
        # Exactly one reconstructed RAW state. Token/light arrays retained by the
        # Collector are independent copies; discarded tails never enter vocabulary.
        current=np.frombuffer(self.payload,RAW,CELLS,12).copy()
        yield current.reshape(27,16,16,16)
        for at,count in self.offsets:
            changes=np.frombuffer(self.payload,DELTA,count,at);current[changes['index']]=changes['raw']
            yield current.reshape(27,16,16,16)
