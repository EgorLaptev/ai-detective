import json
from src.io import ROOT,load_data,fingerprint
from src.stress import variants
if __name__=='__main__':
    ds,cs,qs=load_data()
    proposal=dict(dataset_hash=fingerprint([ds,cs,qs]),reviewed_by=[],approved_variants=[],candidate_variants=[f"{q['question_id']}:{v}" for q in qs for v,*_ in variants(ds,cs,q) if v!='base'])
    (ROOT/'docs/review_candidates.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2))
    print('Созданы кандидаты проверки: docs/review_candidates.json')
