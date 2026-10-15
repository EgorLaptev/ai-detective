import argparse,json
from pathlib import Path
from src.io import read_jsonl,load_data,fingerprint
from src.experiment import export

def apply_review(rows,review):
    if len(set(review.get('reviewed_by',[])))<2:raise ValueError('Нужна проверка обоими участниками')
    if review.get('dataset_hash')!=fingerprint(list(load_data())):raise ValueError('Проверка другого досье')
    for r in rows:r['human_reviewed']=r['variant']!='base' and f"{r['question_id']}:{r['variant']}" in review.get('approved_variants',[])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--review');a=p.parse_args()
    output=Path(a.results);rows=read_jsonl(output/'predictions.jsonl')
    if a.review:apply_review(rows,json.loads(Path(a.review).read_text()))
    export(rows,output)
    from make_report import make_report
    make_report(rows,output)
