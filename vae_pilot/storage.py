"""SQLite is both the compact dataset and exact-content deduplication index."""
import json
import sqlite3
import zlib
from pathlib import Path
import numpy as np
from .sampling import split_for_seed,distinct_progress,progress_sample,clip_size


def js(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)

class Dataset:
    def __init__(self,root,config):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'dataset.sqlite')
        self.config=config;self.limit=None
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS settings(config TEXT);
        CREATE TABLE IF NOT EXISTS palette(id INTEGER PRIMARY KEY,name TEXT UNIQUE,groups_json TEXT);
        CREATE TABLE IF NOT EXISTS snapshots(
          id INTEGER PRIMARY KEY, category TEXT, split TEXT, seed INTEGER, event_id TEXT,
          origin_json TEXT, biome TEXT, biome_group TEXT, terrain TEXT, timestamp REAL,
          info_json TEXT, payload BLOB UNIQUE);
        CREATE INDEX IF NOT EXISTS snapshot_category ON snapshots(category);
        CREATE TABLE IF NOT EXISTS clips(event_id TEXT PRIMARY KEY,kind TEXT,seed INTEGER,
          source TEXT,info_json TEXT);
        CREATE TABLE IF NOT EXISTS worlds(seed INTEGER PRIMARY KEY,info_json TEXT);
        CREATE TABLE IF NOT EXISTS sites(seed INTEGER,site INTEGER,info_json TEXT,PRIMARY KEY(seed,site));
        """)
        old=self.db.execute('SELECT config FROM settings').fetchone()
        if old and old[0]!=js(config):raise ValueError('Dataset config differs; use a new output directory')
        if not old:self.db.execute('INSERT INTO settings VALUES(?)',(js(config),))
        self.names=[];self.groups=[];self.ids={}
        for ident,name,groups in self.db.execute('SELECT id,name,groups_json FROM palette ORDER BY id'):
            self.ids[name]=ident;self.names.append(name);self.groups.append(json.loads(groups))
        self.db.commit()
        self._counts=dict(self.db.execute('SELECT category,count(*) FROM snapshots GROUP BY category'))
        self._biome_counts=dict(self.db.execute("SELECT biome_group,count(*) FROM snapshots WHERE category='ordinary' GROUP BY biome_group"))

    def register_nodes(self,records):
        maximum=max(int(r['id']) for r in records)
        lookup=np.zeros(maximum+1,np.uint16)
        for r in records:
            name=r['name']
            if name not in self.ids:
                ident=len(self.names)
                if ident>65535:raise ValueError('Canonical node palette exceeds uint16')
                self.ids[name]=ident;self.names.append(name);self.groups.append(r.get('groups') or {})
                self.db.execute('INSERT INTO palette VALUES(?,?,?)',(ident,name,js(r.get('groups') or {})))
            lookup[int(r['id'])]=self.ids[name]
        self.db.commit()
        return lookup

    def counts(self):return dict(self._counts)
    def remaining(self,kind):
        counts=self.counts();remaining=self.config['quotas'][kind]-counts.get(kind,0)
        return min(remaining,self.limit-sum(counts.values())) if self.limit else remaining
    def exists(self,state):return self.db.execute('SELECT 1 FROM snapshots WHERE payload=?',(zlib.compress(state.payload()),)).fetchone() is not None

    def insert(self,state,category,event_id):
        i=state.info
        try:
            self.db.execute('INSERT INTO snapshots(category,split,seed,event_id,origin_json,biome,biome_group,terrain,timestamp,info_json,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (category,split_for_seed(i['seed'],self.config['split_percent']),i['seed'],event_id,
                 js(i['origin_enu']),i['biome'],i.get('biome_group'),i['terrain'],i['sim_time'],js(i),zlib.compress(state.payload())))
            self._counts[category]=self._counts.get(category,0)+1
            if category=='ordinary':
                group=i.get('biome_group');self._biome_counts[group]=self._biome_counts.get(group,0)+1
            return True
        except sqlite3.IntegrityError:return False

    def ordinary(self,state):
        if self.remaining('ordinary')<=0:return False
        group=state.info.get('biome_group')
        if group not in self.config['biome_groups']:return False
        per_group=self.config['quotas']['ordinary']//len(self.config['biome_groups'])
        n=self._biome_counts.get(group,0)
        if n>=per_group:return False
        return self.insert(state,'ordinary',None)

    def clip(self,event_id,kind,states,seed,source,extra=None):
        unique,total=distinct_progress(states)
        available=[s for s in unique if not self.exists(s)]
        extra=extra or {}
        n=clip_size(len(available),self.remaining(kind)) if extra.get('native_evolution_observed',True) else 0
        selected=progress_sample(available,n) if n else []
        bounds=self.config['change_bins']
        band='small' if total<bounds[0] else ('medium' if total<bounds[1] else 'large')
        info=dict(distinct_states=len(unique),consecutive_deduplicated_states=1+sum(changed(states[i-1],states[i]) for i in range(1,len(states))),
                  sampled_states=len(selected),cumulative_voxel_updates=total,change_band=band,
                  source=source,first_time=states[0].info['sim_time'],last_time=states[-1].info['sim_time'],
                  selected_times=[s.info['sim_time'] for s in selected],selected_progress=[s.progress for s in selected],
                  standard_clip=len(selected)>=5,duplicate_states_already_stored=len(unique)-len(available),**extra)
        self.db.execute('INSERT INTO clips VALUES(?,?,?,?,?)',(event_id,kind,seed,source,js(info)))
        for state in selected:
            if not self.insert(state,kind,event_id):raise RuntimeError('Clip state unexpectedly duplicated during single-writer insert')
        self.db.commit()
        return len(selected)

    def site(self,seed,site,info):
        self.db.execute('INSERT OR REPLACE INTO sites VALUES(?,?,?)',(seed,site,js(info)));self.db.commit()
    def world(self,seed,info):
        self.db.execute('INSERT OR REPLACE INTO worlds VALUES(?,?)',(seed,js(info)));self.db.commit()
    def close(self):self.db.commit();self.db.close()


def changed(a,b):return bool(np.any(a.nodes!=b.nodes) or np.any(a.param2!=b.param2))


def iter_snapshots(root,split='train'):
    db=sqlite3.connect(Path(root)/'dataset.sqlite')
    try:
        for row in db.execute('SELECT category,event_id,info_json,payload FROM snapshots WHERE split=? ORDER BY id',(split,)):
            category,event_id,info,payload=row;data=zlib.decompress(payload)
            yield dict(voxel_ids=np.frombuffer(data[:8192],'<u2').reshape(16,16,16).copy(),
                       param2=np.frombuffer(data[8192:],'u1').reshape(16,16,16).copy(),category=category,event_id=event_id,metadata=json.loads(info))
    finally:db.close()
