import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def load_data(path=None):
    path=Path(path) if path else ROOT/'data'
    return tuple(read_jsonl(path/f'{name}.jsonl') for name in ('documents','claims','questions'))

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def write_jsonl(path,items):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items),encoding='utf-8')
