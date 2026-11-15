"""Persistent global vocabulary, append-safe Zarr arrays, SQLite commit journal and scene index."""
import json
from pathlib import Path
import sqlite3
import time
import subprocess
import numpy as np
import zarr
from numcodecs import Blosc
import pyarrow as pa
import pyarrow.parquet as pq
from .core import Vocabulary,CONFIG,BIOMES,state_hash,dump
from vae_p1.storage import Trajectory

class Dataset:
    def __init__(self,root,resume=False):
        self.root=Path(root)
        if self.root.exists() and not resume:raise FileExistsError('Use --resume to continue this dataset')
        self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'collection.sqlite');self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY,pool TEXT,idx INTEGER,category TEXT,seed INTEGER,biome TEXT,origin TEXT,identity TEXT UNIQUE,metadata TEXT);
        CREATE TABLE IF NOT EXISTS scenes(id TEXT PRIMARY KEY,data TEXT);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,status TEXT,data TEXT);
        CREATE INDEX IF NOT EXISTS sample_category ON samples(pool,category,biome,seed);
        CREATE INDEX IF NOT EXISTS sample_seed ON samples(pool,category,seed);''')
        path=self.root/'vocabulary.json';self.v=Vocabulary(path if path.exists() else None)
        self.saved_vocabulary_size=len(self.v.entries);self.arrays={}
        for pool in ('standard','aux_short_event'):
            directory=self.root if pool=='standard' else self.root/pool
            g=zarr.open_group(str(directory/'snapshots.zarr'),mode='a');arrays=[]
            for name,dtype in (('tokens','<u2'),('light','u1')):
                if name not in g:g.create_dataset(name,shape=(0,16,16,16),chunks=(8,16,16,16),dtype=dtype,compressor=Blosc(cname='zstd',clevel=3))
                a=g[name];committed=self.db.execute('SELECT COUNT(*) FROM samples WHERE pool=?',(pool,)).fetchone()[0]
                if len(a)<committed:raise ValueError('Committed samples missing array data')
                if len(a)>committed:a.resize((committed,16,16,16))
                arrays.append(a)
            self.arrays[pool]=arrays
        cache=zarr.open_group(str(self.root/'discovery_cache.zarr'),mode='a')
        for name,dtype in (('tokens','<u2'),('light','u1')):
            if name not in cache:cache.create_dataset(name,shape=(0,16,16,16),chunks=(8,16,16,16),dtype=dtype,compressor=Blosc(cname='zstd',clevel=3))
        self.cache=cache
        self._counts=dict(self.db.execute("SELECT category,COUNT(*) FROM samples WHERE pool='standard' GROUP BY category"))
        self._last_export=time.monotonic()
        config_path=self.root/'collection_config.json'
        if not config_path.exists():
            source=Path(__file__).resolve().parents[1]
            commit=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True)
            dirty=subprocess.run(['git','-C',str(source),'status','--porcelain'],capture_output=True,text=True)
            dump(config_path,dict(config=CONFIG,collector_base_commit=commit.stdout.strip(),collector_worktree_dirty=bool(dirty.stdout.strip()),
                provenance_note='Dirty runs are not claimed to have a clean release revision. Engine/game commits are recorded per scene.'))
        for name in ('trajectories','visualizations'): (self.root/name).mkdir(exist_ok=True)
    def ensure_vocabulary(self):
        if len(self.v.entries)!=self.saved_vocabulary_size or not(self.root/'vocabulary.json').exists():
            self.v.save(self.root/'vocabulary.json');self.saved_vocabulary_size=len(self.v.entries)
    def cache_scene(self,scene,state):
        if state.meta.get('data_source')!='native':raise ValueError('Only native discovery observations may be cached')
        if self.db.execute('SELECT 1 FROM scenes WHERE id=?',(scene['scene_id'],)).fetchone():return
        self.ensure_vocabulary();index=len(self.cache['tokens'])
        for name,value in (('tokens',state.tokens),('light',state.light)):
            self.cache[name].resize((index+1,16,16,16));self.cache[name][index]=value
        scene['cache_index']=index;scene['cache_metadata']=state.meta;self.add_scene(scene)
    def cached_state(self,scene):
        from vae_p0.sampling import State
        i=scene['cache_index'];return State(self.cache['tokens'][i],self.cache['light'][i],dict(scene['cache_metadata']))
    def add_scene(self,scene):
        self.db.execute('INSERT OR REPLACE INTO scenes VALUES(?,?)',(scene['scene_id'],json.dumps(scene)));self.db.commit()
    def scenes(self):return [json.loads(r[0]) for r in self.db.execute('SELECT data FROM scenes')]
    def counts(self):return dict(self._counts)
    def append(self,state,category,scene,rollout,pool='standard',extra=None):
        if state.meta.get('data_source')!='native':raise ValueError('Synthetic fixtures cannot enter the native dataset')
        digest=state_hash(state,self.v);origin=state.meta['chunk_origin'];seed=int(scene['world_seed'])
        identity=json.dumps([seed,scene.get('region_id',scene['scene_id']),category,origin,digest])
        if self.db.execute('SELECT 1 FROM samples WHERE identity=?',(identity,)).fetchone():return False
        biome=state.meta.get('biome_group');quota=CONFIG['quotas'][category]
        if pool=='standard':
            if self.counts().get(category,0)>=quota:return False
            if category=='Ordinary':
                if biome not in BIOMES:return False
                per_biome=self.db.execute("SELECT COUNT(*) FROM samples WHERE pool='standard' AND category='Ordinary' AND biome=?",(biome,)).fetchone()[0]
                per_seed=self.db.execute("SELECT COUNT(*) FROM samples WHERE pool='standard' AND category='Ordinary' AND seed=?",(seed,)).fetchone()[0]
                if per_biome>=CONFIG['ordinary_biome_target'] or per_seed>=int(quota*CONFIG['max_ordinary_seed_fraction']):return False
        # Save append-only IDs before making array data commit-visible.
        self.ensure_vocabulary()
        a,b=self.arrays[pool];n=len(a)
        for array,value in ((a,state.tokens),(b,state.light)):
            array.resize((n+1,16,16,16));array[n]=value
        meta=dict(state.meta);meta.update(category=category,scene_id=scene['scene_id'],rollout_id=rollout,token_hash=digest,
                  phase=state.meta.get('phase','native_observation'));meta.update(extra or{})
        self.db.execute('INSERT INTO samples(pool,idx,category,seed,biome,origin,identity,metadata) VALUES(?,?,?,?,?,?,?,?)',
            (pool,n,category,seed,biome,json.dumps(origin),identity,json.dumps(meta)));self.db.commit()
        if pool=='standard':self._counts[category]=self._counts.get(category,0)+1
        return True
    def job(self,key,status,data):
        self.db.execute('INSERT OR REPLACE INTO jobs VALUES(?,?,?)',(key,status,json.dumps(data)));self.db.commit()
        row=dict(data);row.update(job_id=key,status=status)
        with open(self.root/'scenes.jsonl','a')as f:f.write(json.dumps(row)+'\n')
    def completed(self,key):
        row=self.db.execute('SELECT status FROM jobs WHERE id=?',(key,)).fetchone();return row and row[0]=='complete'
    def flush_if_due(self):
        # SQLite commits each sample; avoid rewriting every Parquet row after every scene.
        if time.monotonic()-self._last_export>=300:self.flush()
    def flush(self):
        self._last_export=time.monotonic()
        self.v.save(self.root/'vocabulary.json')
        for pool in self.arrays:
            rows=[dict(snapshot_id=r[0],**json.loads(r[1])) for r in self.db.execute('SELECT idx,metadata FROM samples WHERE pool=? ORDER BY idx',(pool,))]
            target=self.root if pool=='standard' else self.root/pool
            if rows:
                # JSON encode nested provenance to keep schema stable across event categories.
                keys=sorted(set().union(*(row.keys() for row in rows)))
                rows=[{k:json.dumps(row.get(k)) if isinstance(row.get(k),dict) else row.get(k) for k in keys} for row in rows]
                pq.write_table(pa.Table.from_pylist(rows),target/'metadata.parquet',compression='zstd')
            else:pq.write_table(pa.table({'snapshot_id':pa.array([],pa.int64())}),target/'metadata.parquet')
        scenes=self.scenes()
        if scenes:pq.write_table(pa.Table.from_pylist([dict(scene_id=s['scene_id'],world_seed=s['world_seed'],origin=s['origin'],data=json.dumps(s)) for s in scenes]),self.root/'scene_index.parquet',compression='zstd')
    def summary(self):
        from collections import Counter,defaultdict
        groups=defaultdict(lambda:dict(attempts=0,wall_seconds=0,simulation_seconds=0,native_changes=0,standard_snapshots=0))
        for status,data in self.db.execute('SELECT status,data FROM jobs'):
            r=json.loads(data);category=r.get('category')
            if not category:continue
            if 'standard_snapshots' not in r:
                r['standard_snapshots']=self.db.execute("SELECT COUNT(*) FROM samples WHERE pool='standard' AND json_extract(metadata,'$.rollout_id')=?",(r.get('rollout_id'),)).fetchone()[0]
            g=groups[category];g['attempts']+=1;g['wall_seconds']+=r.get('wall_seconds',0);g['simulation_seconds']+=r.get('observation_seconds',0)
            g['native_changes']+=r.get('native_changes',0);g['standard_snapshots']+=r.get('standard_snapshots',0)
        for g in groups.values():
            g['snapshots_per_sim_hour']=g['standard_snapshots']/g['simulation_seconds']*3600 if g['simulation_seconds'] else None
            g['snapshots_per_wall_hour']=g['standard_snapshots']/g['wall_seconds']*3600 if g['wall_seconds'] else None
        overlap=self.db.execute("SELECT COUNT(*) FROM (SELECT json_extract(metadata,'$.token_hash') h,seed,origin,COUNT(DISTINCT category) n FROM samples WHERE pool='standard' GROUP BY h,seed,origin HAVING n>1)").fetchone()[0]
        scenes=self.scenes()
        return dict(counts=self.counts(),indexed_scenes=len(scenes),category_rates=dict(groups),
            discovery_by_biome=dict(Counter(scene.get('biome_group') or 'unresolved' for scene in scenes)),
            discovery_by_stratum=dict(Counter(scene['stratum'] for scene in scenes)),
            ordinary_by_stratum=dict(self.db.execute("SELECT json_extract(metadata,'$.stratum'),COUNT(*) FROM samples WHERE category='Ordinary' AND pool='standard' GROUP BY json_extract(metadata,'$.stratum')")),
            ordinary_by_biome=dict(self.db.execute("SELECT biome,COUNT(*) FROM samples WHERE category='Ordinary' AND pool='standard' GROUP BY biome")),
            seed_counts=dict(self.db.execute("SELECT seed,COUNT(*) FROM samples WHERE pool='standard' GROUP BY seed")),
            exact_source_states_shared_across_categories=overlap,
            units='scene/rollout and world seed; correlated snapshots are not independent replicates',
            resume='Committed samples are retained. Incomplete scenes restart as new rollouts; native time is not rewound.')
    def close(self):self.flush();self.db.close()

class RawTrajectory(Trajectory):
    def __init__(self,root,dataset):super().__init__(root);self.dataset=dataset
    def append(self,state,scene,phase):
        self.dataset.ensure_vocabulary();return super().append(state,scene,phase)
