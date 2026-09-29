from __future__ import annotations
import hashlib,json
from pathlib import Path
from .utils import canonical_json, parameter_hash

class VerifiAudit:
    def __init__(self,path):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text('',encoding='utf-8'); self.prev_digest='GENESIS'
    def append(self,record,model_vector):
        rec=dict(record); rec['model_hash']=parameter_hash(model_vector); rec['previous_digest']=self.prev_digest
        payload=canonical_json(rec); digest=hashlib.sha256((self.prev_digest+payload).encode()).hexdigest(); rec['round_digest']=digest
        with self.path.open('a',encoding='utf-8') as f:f.write(canonical_json(rec)+'\n')
        self.prev_digest=digest; return digest
    def verify(self):
        prev='GENESIS'
        for line in self.path.read_text(encoding='utf-8').splitlines():
            rec=json.loads(line); got=rec.pop('round_digest'); expected=hashlib.sha256((prev+canonical_json(rec)).encode()).hexdigest()
            if got!=expected or rec.get('previous_digest')!=prev:return False
            prev=got
        return True
