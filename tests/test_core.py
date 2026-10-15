import itertools,json
import pytest
from src.padic import distance,encode
from src.schemas import parse_answer
from src.io import load_data
from src.retrieval import retrieve
from src.context_builder import build_context,select
from src.stress import remove_premise,permutation,variants
from src.llm import OpenAICompatible,LLMUnavailable
from src.evaluation import score,summarize,stability
from validate_data import validate

@pytest.fixture
def data():return load_data()

def answer(**kw):
    a=dict(question_id='Q01',status='entailed',answer='Подтверждается C01',evidence_ids=['C01'],counterevidence_ids=[],confidence=.8,explanation='Основание C01')
    a.update(kw);return json.dumps(a,ensure_ascii=False)

def test_dataset(data):assert validate(*data)['words']>=5000

def test_identical_paths():assert distance([0,1,2],[0,1,2])==0

@pytest.mark.parametrize('x,y,w',[([0,0,0],[1,0,0],1),([0,0,0],[0,1,0],1/3),([0,0,0],[0,0,1],1/9)])
def test_level_distances(x,y,w):assert distance(x,y)==pytest.approx(w)

def test_encoding():assert encode([1,2,1])==16

def test_not_absolute_code_distance():assert distance([0,0,0],[0,0,1])!=abs(encode([0,0,0])-encode([0,0,1]))

def test_ultrametric_exhaustive():
    paths=list(itertools.product(range(3),repeat=3))
    for x,y,z in itertools.product(paths,repeat=3):assert distance(x,z)<=max(distance(x,y),distance(y,z))

@pytest.mark.parametrize('raw',['not JSON','```json\n{}\n```','{"question_id":'])
def test_invalid_json(raw):
    with pytest.raises(ValueError):parse_answer(raw,'Q01',['C01'])

def test_unknown_id():
    with pytest.raises(ValueError):parse_answer(answer(evidence_ids=['C99']),'Q01',['C01'])

def test_retrieved_id_required():
    with pytest.raises(ValueError):parse_answer(answer(evidence_ids=['C02']),'Q01',['C01'])

def test_both_requires_counter():
    with pytest.raises(ValueError):parse_answer(answer(status='both'),'Q01',['C01'])

def test_insufficient_cannot_assert_invented_fact():
    with pytest.raises(ValueError):parse_answer(answer(status='insufficient',answer='Лев тайно отправил пакет'),'Q01',['C01'])

def test_insufficient_null():assert parse_answer(answer(status='insufficient',answer=None),'Q01',['C01']).answer is None

def test_llm_unavailable(monkeypatch):
    import requests
    def fail(*a,**kw):raise requests.ConnectionError('offline')
    monkeypatch.setattr(requests,'post',fail)
    with pytest.raises(LLMUnavailable):OpenAICompatible()([])

def test_bm25_limit_and_stable_ties(data):
    _,cs,qs=data
    assert len(retrieve(qs[0]['question'],cs,k=100))==15
    assert retrieve('несуществующиеслова',cs)==retrieve('несуществующиеслова',list(reversed(cs)))

def test_bc_identical_ids_and_payload(data):
    ds,cs,qs=data;s=select(qs[0]['question'],cs)
    b,bi=build_context('B',ds,cs,s);c,ci=build_context('C',ds,cs,s)
    assert set(bi)==set(ci) and len(bi)==15
    from src.context_builder import claim_line
    for claim in s:assert claim_line(claim) in b and claim_line(claim) in c

def test_a_does_not_get_gold(data):
    ds,cs,_=data;text,ids=build_context('A',ds,cs)
    assert 'gold_status' not in text and set(ids)=={c['claim_id'] for c in cs}
    for c in cs:assert f"[{c['claim_id']}] {c['evidence']}" in text

def test_removal_reaches_full_context(data):
    ds,cs,qs=data
    for q in qs:
        if q['remove_claim_id']:
            cid=q['remove_claim_id'];orig=next(c for c in cs if c['claim_id']==cid)
            nd,nc=remove_premise(ds,cs,cid)
            assert orig['evidence'] not in '\n'.join(d['text'] for d in nd)
            assert cid not in {c['claim_id'] for c in nc}
            assert len(nd)==len(ds) and len(nc)==len(cs)-1

def test_permutation_only_order(data):
    ds,cs,_=data;nd,nc=permutation(ds,cs)
    assert nd!=ds and sorted(nd,key=lambda x:x['document_id'])==sorted(ds,key=lambda x:x['document_id']) and nc==cs
    assert nd==permutation(ds,cs)[0]

def test_variants_evidence_exists(data):
    ds,cs,qs=data;count=0
    for q in qs:
        for variant,nd,nc,gold,meta in variants(ds,cs,q):
            count+=3
            docs={d['document_id']:d for d in nd}
            for c in nc:assert c['evidence'] in docs[c['document_id']]['text']
            assert set(gold['gold_evidence_ids']+gold['gold_counterevidence_ids'])<={c['claim_id'] for c in nc}
    assert count==120

def test_invalid_predictions_count_as_wrong(data):
    q=data[2][0];row=dict(gold=q,parsed=None,variant='base',mode='A',seconds=1)
    assert score(row)['evidence_f1']==0
    _,s=summarize([row]);assert s.iloc[0].status_accuracy==0

def test_unreviewed_stress_excluded(data):
    q=data[2][0];row=dict(gold=q,parsed=None,variant='contradiction',mode='A',seconds=1,human_reviewed=False)
    assert summarize([row])[0].empty

def test_pipeline_checkpoint_and_resume(tmp_path):
    # Только тестовый транспорт в временном каталоге, не исследовательский прогон.
    from src.experiment import run
    class TestTransport:
        model='unit-test-only'
        def __init__(self):self.calls=0
        def __call__(self,messages):
            self.calls+=1
            return dict(raw=answer(),input_tokens=10,output_tokens=5)
    llm=TestTransport()
    first=run(llm,tmp_path,only_base=True,limit=1)
    assert len(first)==1 and llm.calls==1
    second=run(llm,tmp_path,only_base=True,limit=1)
    assert len(second)==2 and llm.calls==2
    assert (tmp_path/'variants/Q01/base/documents.jsonl').exists()
