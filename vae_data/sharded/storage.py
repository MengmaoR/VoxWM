"""Shard-owned writes, a short global vocabulary lock, and atomic READY publication."""
from contextlib import contextmanager
from pathlib import Path
from collections import Counter
import fcntl,hashlib,json,os,sqlite3,tempfile,time
import numpy as np
import zarr
from numcodecs import Blosc
import pyarrow as pa
import pyarrow.parquet as pq
from .protocol import CONFIG,VERSION,CATEGORIES,STRATA,BIOMES,Vocabulary,Frame,Chain,dump,d4_targets,STATIC_HEIGHT_POLICY_VERSION

@contextmanager
def lock(path,blocking=True,timeout=30.):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a+')as stream:
        deadline=time.monotonic()+timeout
        while True:
            try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if not blocking:raise
                if time.monotonic()>=deadline:raise TimeoutError('File lock wait exceeded '+str(timeout)+'s: '+str(path))
                time.sleep(min(.05,max(0,deadline-time.monotonic())))
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def stage_json(path,value,sync=True):
    path=Path(path);fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w')as stream:
            json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n');stream.flush()
            if sync:os.fsync(stream.fileno())
        return name
    except BaseException:
        if os.path.exists(name):os.unlink(name)
        raise


def atomic_json(path,value,sync=True):
    name=stage_json(path,value,sync)
    try:os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def initialize(root,fixture=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with lock(root/'.dataset.lock'):
        path=root/'dataset_manifest.json'
        if path.exists():
            current=json.loads(path.read_text())
            if current.get('version')!=VERSION or current.get('source_mode')!=('unit_fixture'if fixture else'native'):raise ValueError('Choose a new dataset directory for this format/source mode')
            if current.get('static_height_policy_version')!=STATIC_HEIGHT_POLICY_VERSION:raise ValueError('Static height policy changed; use a new dataset directory, do not mix old samples')
        else:
            if any(p.name not in('.dataset.lock','.locks')for p in root.iterdir()):raise ValueError('New sharded format requires an empty dataset directory')
            atomic_json(path,dict(CONFIG,source_mode='unit_fixture'if fixture else'native',vocabulary='append-only global Node Name/raw param2'))
            Vocabulary().save(root/'vocabulary.json')
        (root/'shards').mkdir(exist_ok=True)
        with sqlite3.connect(root/'static_positions.sqlite')as positions:
            positions.execute('CREATE TABLE IF NOT EXISTS positions(position_key TEXT PRIMARY KEY,shard INTEGER NOT NULL,snapshot INTEGER NOT NULL)')
    return root

def globalize(root,frames,local_vocabulary,timeout=30.):
    """Durable append-only IDs; known tokens never acquire the publication lock.

    New snapshots are written/fsynced BEFORE taking the short compare/replace
    lock. A slow filesystem sync cannot hold every shard behind vocabulary.lock.
    Concurrent publishers retry against the newly published prefix, never expose
    tentative IDs and never overwrite another worker's appended tokens.
    """
    root=Path(root);path=root/'vocabulary.json';deadline=time.monotonic()+timeout
    used=np.unique(np.concatenate([f.tokens.ravel()for f in frames]))
    wanted={int(token):local_vocabulary.entries[int(token)]for token in used}
    while True:
        v=Vocabulary(path);base=list(v.entries);mapping=np.zeros(len(local_vocabulary.entries),np.uint16)
        for token,pair in wanted.items():mapping[token]=v.add(*pair)
        if len(v.entries)==len(base):
            return [Frame(mapping[f.tokens],f.light,dict(f.meta))for f in frames],len(v.entries)
        data=dict(version=1,representation='VocabularyID(Node Name, raw param2)',K=len(v.entries),light_in_token=False,
            tokens=[dict(id=i,node_name=name,param2=param)for i,(name,param)in enumerate(v.entries)])
        # Expensive serialization and fsync occur outside any cross-shard lock.
        staged=stage_json(path,data);published=False
        try:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Vocabulary publication exceeded time budget')
            with lock(root/'.vocabulary.lock',timeout=remaining):
                if Vocabulary(path).entries==base:
                    os.replace(staged,path);published=True
        finally:
            if os.path.exists(staged):os.unlink(staged)
        if published:return [Frame(mapping[f.tokens],f.light,dict(f.meta))for f in frames],len(v.entries)
        if time.monotonic()>=deadline:raise TimeoutError('Vocabulary changed repeatedly during publication')


def source_key(frame,vocabulary):
    # Used only for same-source duplicate detection, not a provenance/hash audit.
    names,param=vocabulary.decode(frame.center)
    content=''.join(f'{name}\0{int(p)}\n'for name,p in zip(names.ravel(),param.ravel())).encode()
    key=json.dumps([frame.meta['world_seed'],frame.meta['center_origin']]).encode()+content
    return hashlib.blake2b(key,digest_size=20).hexdigest()

def static_position_key(frame):
    # The real sampling-window center, not a token/content hash or engine block ID.
    return json.dumps([int(frame.meta['world_seed']),*[int(v)+8 for v in frame.meta['center_origin']]],separators=(',',':'))


def rows_at(path):
    db=sqlite3.connect(f'file:{Path(path)/"journal.sqlite"}?mode=ro',uri=True)
    try:
        rows=[json.loads(row[0])for row in db.execute('SELECT metadata FROM samples ORDER BY idx')]
        chains=[json.loads(row[0])for row in db.execute('SELECT data FROM chains ORDER BY id')]
        return rows,chains
    finally:db.close()

class Writer:
    def __init__(self,root,shard,resume=False,fixture=False):
        self.root=initialize(root,fixture);self.shard=int(shard);self.fixture=fixture;self.name=f'shard_{shard:04d}'
        if not 0<=self.shard<100:raise ValueError('Shard index must be 0..99')
        self.ownership=lock(self.root/'.locks'/(self.name+'.lock'),False);self.ownership.__enter__()
        self.closed=False
        try:
            final=self.root/'shards'/self.name;partial=self.root/'shards'/('.'+self.name+'.partial')
            if (final/'READY').exists():raise FileExistsError('READY shard is immutable')
            self.path=final if final.exists()else partial
            if self.path.exists()and not resume:raise FileExistsError('Incomplete shard exists; use --resume')
            self.path.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(self.path/'journal.sqlite')
            self.db.execute('ATTACH DATABASE ? AS static_registry',(str(self.root/'static_positions.sqlite'),))
            self.db.executescript('''CREATE TABLE IF NOT EXISTS samples(idx INTEGER PRIMARY KEY,category TEXT,source_key TEXT UNIQUE,metadata TEXT);
                CREATE TABLE IF NOT EXISTS chains(id TEXT PRIMARY KEY,data TEXT,anchor_idx INTEGER);
                CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,data TEXT);''')
            n=self.db.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
            a=self.db.execute('SELECT COUNT(*) FROM chains WHERE anchor_idx IS NOT NULL').fetchone()[0]
            self.arrays={}
            for pool,size in(('snapshots',n),('anchors',a)):
                group=zarr.open_group(str(self.path/(pool+'.zarr')),mode='a')
                for name,dtype in(('tokens','<u2'),('light','u1')):
                    if name not in group:group.create_dataset(name,shape=(0,48,48,48),chunks=(1,48,48,48),dtype=dtype,compressor=Blosc(cname='zstd',clevel=3))
                    array=group[name]
                    if len(array)<size:raise ValueError('Committed array data missing')
                    if len(array)>size:array.resize((size,48,48,48))
                self.arrays[pool]=group
        except BaseException:
            self.ownership.__exit__(None,None,None);raise
    def rows(self):return [json.loads(row[0])for row in self.db.execute('SELECT metadata FROM samples ORDER BY idx')]
    def counts(self):return Counter(dict(self.db.execute('SELECT category,COUNT(*) FROM samples GROUP BY category')))
    def missing(self):
        rows=self.rows();static=Counter((r['biome_group'],r['stratum'])for r in rows if r['category']=='Static')
        return dict(categories={k:q-self.counts()[k]for k,q in CATEGORIES.items()},
            static=[dict(biome=biome,stratum=stratum,remaining=quota-static[biome,stratum])for biome in BIOMES for stratum,quota in STRATA.items()if static[biome,stratum]<quota],
            d4={subtype:q-self.db.execute("SELECT COUNT(*) FROM chains WHERE json_extract(data,'$.subtype')=?",(subtype,)).fetchone()[0]for subtype,q in d4_targets(self.shard).items()})
    def attempt(self,row):
        self.db.execute('INSERT INTO attempts(data)VALUES(?)',(json.dumps(row),));self.db.commit()
    def submit(self,chain,vocabulary):
        category=chain.category;frames=chain.frames;n=len(frames)
        if category not in CATEGORIES or n!=(1 if category=='Static'else 5):raise ValueError('A dynamic commit must contain exactly five states')
        if self.counts()[category]+n>CATEGORIES[category]:return False
        if any(f.meta.get('data_source')!=('synthetic_fixture'if self.fixture else'native')for f in frames):raise ValueError('Fixture/native provenance mismatch')
        rows=self.rows();seed=frames[0].meta['world_seed']
        if sum(r['world_seed']==seed for r in rows)+n>CONFIG['max_samples_per_seed_per_shard']:return False
        position_key=static_position_key(frames[0])if category=='Static'else None
        if category=='Static':
            if not self.fixture:
                from .validate import validate_native
                from .static_height import validate_static
                validate_native(frames[0]);validate_static(frames[0],vocabulary)
            if self.db.execute('SELECT 1 FROM static_registry.positions WHERE position_key=?',(position_key,)).fetchone():return False
            biome=frames[0].meta['biome_group'];stratum=frames[0].meta['stratum']
            if biome not in BIOMES or stratum not in STRATA:return False
            if sum(r['category']=='Static'and r['biome_group']==biome and r['stratum']==stratum for r in rows)>=STRATA[stratum]:return False
        if category=='D4':
            if chain.subtype not in d4_targets(self.shard):raise ValueError('Missing D4 subtype')
            have=self.db.execute("SELECT COUNT(*) FROM chains WHERE json_extract(data,'$.subtype')=?",(chain.subtype,)).fetchone()[0]
            if have>=d4_targets(self.shard)[chain.subtype]:return False
        if category!='Static':
            if any('candidate_index'not in f.meta for f in frames):raise ValueError('Missing candidate-window metadata')
            from .validate import validate_chain
            validate_chain(chain,vocabulary,require_native=not self.fixture)
        keys=[source_key(f,vocabulary)for f in frames]
        if len(set(keys))!=len(keys)or any(self.db.execute('SELECT 1 FROM samples WHERE source_key=?',(key,)).fetchone()for key in keys):return False
        all_frames=frames+([chain.anchor]if chain.anchor else[])
        mapped,vsize=globalize(self.root,all_frames,vocabulary)
        group=self.arrays['snapshots'];start=len(group['tokens']);anchor_group=self.arrays['anchors'];anchor_count=len(anchor_group['tokens'])
        chain_id=frames[0].meta['rollout_id']+'/'+category
        try:
            for name in('tokens','light'):
                group[name].resize((start+n,48,48,48));group[name][start:start+n]=np.stack([getattr(f,name)for f in mapped[:n]])
            anchor_idx=None
            if chain.anchor:
                anchor_idx=anchor_count
                for name in('tokens','light'):
                    anchor_group[name].resize((anchor_count+1,48,48,48));anchor_group[name][anchor_count]=getattr(mapped[-1],name)
            self.db.execute('BEGIN')
            if category=='Static':
                # Both journals participate in one SQLite transaction; a failed
                # local commit cannot leave a separately committed reservation.
                self.db.execute('INSERT INTO static_registry.positions VALUES(?,?,?)',(position_key,self.shard,start))
            for i,(frame,key)in enumerate(zip(frames,keys)):
                meta=dict(frame.meta,snapshot_id=start+i,category=category,chain_id=chain_id if category!='Static'else None,
                    chain_index=i if category!='Static'else None,physical_frame_id=frame.key,vocabulary_size_at_commit=vsize)
                if category=='Static':meta['static_position_key']=position_key
                self.db.execute('INSERT INTO samples VALUES(?,?,?,?)',(start+i,category,key,json.dumps(meta)))
            if category!='Static':
                data=dict(id=chain_id,category=category,snapshot_indices=list(range(start,start+n)),frame_ids=[f.key for f in frames],
                    events=chain.events,intervention=chain.intervention,subtype=chain.subtype,anchor_idx=anchor_idx,anchor_metadata=chain.anchor.meta if chain.anchor else None,anchor_vocabulary_size_at_commit=vsize if chain.anchor else None)
                self.db.execute('INSERT INTO chains VALUES(?,?,?)',(chain_id,json.dumps(data),anchor_idx))
            self.db.commit()
            try:(self.path/'progress').touch()
            except OSError:pass
            return True
        except BaseException as exc:
            self.db.rollback()
            for name in('tokens','light'):
                group[name].resize((start,48,48,48));anchor_group[name].resize((anchor_count,48,48,48))
            if category=='Static'and isinstance(exc,sqlite3.IntegrityError)and self.db.execute('SELECT 1 FROM static_registry.positions WHERE position_key=?',(position_key,)).fetchone():return False
            raise
    def export(self):
        rows=self.rows();data=[dict(snapshot_id=r['snapshot_id'],category=r['category'],world_seed=r['world_seed'],split=r['split'],
            biome_group=r.get('biome_group'),stratum=r.get('stratum'),chain_id=r['chain_id'],metadata_json=json.dumps(r))for r in rows]
        pq.write_table(pa.Table.from_pylist(data)if data else pa.table({'snapshot_id':pa.array([],pa.int64())}),self.path/'metadata.parquet',compression='zstd')
        from .centers import height_statistics
        heights=height_statistics([json.loads(row[0])for row in self.db.execute('SELECT data FROM attempts')])
        stats=dict(height_sampling=heights,counts=dict(self.counts()),samples=len(rows),missing=self.missing(),attempts=self.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0])
        dump(self.path/'statistics.json',stats)
        dump(self.path/'manifest.json',dict(version=VERSION,shard=self.shard,source_mode='unit_fixture'if self.fixture else'native',
            samples=len(rows),config=CONFIG,center_slice=[[16,32]]*3,region_shape=[48]*3,d4_chain_targets=d4_targets(self.shard)))
        return stats
    def publish(self):
        from .validate import check_shard
        self.export();result=check_shard(self.root,self.path,require_complete=True,require_native=not self.fixture)
        if not result['valid']:raise ValueError('Shard remains incomplete/invalid: '+json.dumps(result['errors']))
        self.db.close();self.db=None;final=self.root/'shards'/self.name
        if self.path!=final:os.rename(self.path,final)
        self.path=final;atomic_json(final/'READY',dict(version=VERSION,samples=1000,source_mode='unit_fixture'if self.fixture else'native'))
        return result
    def close(self):
        if self.closed:return
        if getattr(self,'db',None)is not None:self.export();self.db.close()
        self.closed=True;self.ownership.__exit__(None,None,None)
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
