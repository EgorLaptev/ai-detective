from collections import Counter
from src.io import load_data
from src.padic import validate_path

def validate(documents,claims,questions):
    assert 8<=len(documents)<=10
    words=sum(len(d['text'].split()) for d in documents);assert 5000<=words<=8000,words
    assert 30<=len(claims)<=40
    assert len(questions)==12
    assert set(Counter(q['question_type'] for q in questions).values())=={2}
    docs={d['document_id']:d for d in documents};cs={c['claim_id']:c for c in claims}
    assert len(docs)==len(documents) and len(cs)==len(claims)
    for c in claims:
        assert c['status'] in ('direct','reported','inferred','disputed')
        assert c['evidence'] in docs[c['document_id']]['text'],c['claim_id']
        validate_path(c['padic_path'])
    for q in questions:
        assert set(q['gold_evidence_ids']+q['gold_counterevidence_ids'])<=set(cs)
        assert q['gold_status'] in ('entailed','contradicted','both','insufficient')
    return dict(documents=len(docs),words=words,claims=len(cs),questions=len(questions))
if __name__=='__main__':print(validate(*load_data()))
