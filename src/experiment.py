import json,time,platform,importlib.metadata
from pathlib import Path
from .io import ROOT,fingerprint,load_data,write_jsonl,read_jsonl
from .context_builder import build_context,select
from .schemas import parse_answer
from .stress import variants
from .evaluation import summarize,stability

def make_messages(system,question,context):
    return [{'role':'system','content':system},{'role':'user','content':f"ВОПРОС {question['question_id']}\n{question['question']}\n\nДОСЬЕ\n{context}"}]

def call_one(llm,system,q,mode,documents,claims,selected):
    context,ids=build_context(mode,documents,claims,selected)
    messages=make_messages(system,q,context)
    started=time.perf_counter();raw='';error=None;parsed=None;usage={}
    try:
        usage=llm(messages);raw=usage['raw']
        parsed=parse_answer(raw,q['question_id'],ids).model_dump()
    except Exception as e:
        error=f'{type(e).__name__}: {e}'
    return dict(parsed=parsed,raw=raw,error=error,seconds=time.perf_counter()-started,input_tokens=usage.get('input_tokens'),output_tokens=usage.get('output_tokens'),context_hash=fingerprint(context),available_claim_ids=ids,messages=messages)

def run(llm,output,only_base=False,review_path=None,limit=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    docs,claims,qs=load_data();system=(ROOT/'prompts/system.txt').read_text(encoding='utf-8')
    review=json.loads(Path(review_path).read_text()) if review_path else {}
    run_config=dict(backend=type(llm).__name__,model=llm.model,revision=getattr(llm,'revision',None),temperature=0,do_sample=False,max_new_tokens=384,seed=42,dataset_hash=fingerprint([docs,claims,qs]),prompt_hash=fingerprint(system),python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ('pydantic','rank-bm25','pandas')})
    run_hash=fingerprint(run_config)
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['run_hash']!=run_hash: raise ValueError('Папка содержит результаты другой конфигурации')
    manifest.write_text(json.dumps(dict(run_hash=run_hash,config=run_config),ensure_ascii=False,indent=2))
    path=output/'predictions.jsonl';existing=read_jsonl(path) if path.exists() else []
    done={(r['question_id'],r['variant'],r['mode']) for r in existing}
    # Ошибки не удаляются. Для нового прогона используется новая output-папка.
    rows=list(existing);calls=0
    for q in qs:
        for variant,ds,cs,gold,meta in variants(docs,claims,q):
            if only_base and variant!='base':continue
            reviewed=bool(review.get('reviewed_by') and len(set(review['reviewed_by']))>=2 and review.get('dataset_hash')==run_config['dataset_hash'] and f"{q['question_id']}:{variant}" in review.get('approved_variants',[]))
            selected=select(q['question'],cs)
            assert set(build_context('B',ds,cs,selected)[1])==set(build_context('C',ds,cs,selected)[1])
            for mode in ('A','B','C'):
                key=(q['question_id'],variant,mode)
                if key in done:continue
                result=call_one(llm,system,q,mode,ds,cs,selected)
                # Сначала сохраняется raw, потом считается статистика. Снимок варианта полный.
                row=dict(question_id=q['question_id'],variant=variant,mode=mode,gold=gold,mutation=meta,human_reviewed=reviewed,document_order=[d['document_id'] for d in ds],dataset_hash=fingerprint([ds,cs]),**result)
                with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                snapshot=output/'variants'/q['question_id']/variant
                write_jsonl(snapshot/'documents.jsonl',ds);write_jsonl(snapshot/'claims.jsonl',cs)
                rows.append(row);done.add(key);calls+=1
                print(q['question_id'],variant,mode,'OK' if result['parsed'] else result['error'],flush=True)
                if result['error'] and result['error'].startswith('LLMUnavailable'):raise RuntimeError(result['error'])
                if limit and calls>=limit:export(rows,output);return rows
    export(rows,output)
    return rows

def export(rows,output):
    output=Path(output)
    frame,summary=summarize(rows)
    frame.to_csv(output/'row_metrics.csv',index=False);summary.to_csv(output/'summary.csv',index=False)
    stability(rows).to_csv(output/'stability.csv',index=False)
    # Непроверенные stress-ответы сохранены, но не входят в отчётные метрики.
    _,provisional=summarize(rows,include_unreviewed=True)
    provisional.to_csv(output/'provisional_includes_unreviewed.csv',index=False)
