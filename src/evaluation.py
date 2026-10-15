"""Метрики по всем запросам. Ошибки формата входят в знаменатель accuracy."""
import math
import pandas as pd

def set_scores(pred,gold):
    pred,gold=set(pred),set(gold)
    if not pred and not gold: return 1.,1.,1.
    tp=len(pred&gold);p=tp/len(pred) if pred else 0.;r=tp/len(gold) if gold else 0.
    return p,r,2*p*r/(p+r) if p+r else 0.

def score(row):
    ans=row.get('parsed');gold=row['gold']
    if not ans:
        return dict(status_correct=False,evidence_precision=0.,evidence_recall=0.,evidence_f1=0.,counterevidence_recall=0. if gold['gold_counterevidence_ids'] else None)
    p,r,f=set_scores(ans['evidence_ids'],gold['gold_evidence_ids'])
    co=set(gold['gold_counterevidence_ids'])
    return dict(status_correct=ans['status']==gold['gold_status'],evidence_precision=p,evidence_recall=r,evidence_f1=f,counterevidence_recall=len(set(ans['counterevidence_ids'])&co)/len(co) if co else None)

def summarize(rows,include_unreviewed=False):
    records=[]
    for row in rows:
        if row['variant']!='base' and not row.get('human_reviewed') and not include_unreviewed: continue
        ans=row.get('parsed') or {}
        records.append(dict(mode=row['mode'],variant=row['variant'],status=ans.get('status'),gold_status=row['gold']['gold_status'],confidence=ans.get('confidence'),seconds=row['seconds'],input_tokens=row.get('input_tokens'),output_tokens=row.get('output_tokens'),valid=bool(ans),**score(row)))
    frame=pd.DataFrame(records)
    if frame.empty: return frame,pd.DataFrame()
    summaries=[]
    for (mode,variant),g in frame.groupby(['mode','variant']):
        known=g[g.gold_status=='insufficient']
        both=g[g.gold_status=='both']
        valid=g[g.valid]
        summaries.append(dict(mode=mode,variant=variant,n=len(g),valid_rate=g.valid.mean(),status_accuracy=g.status_correct.mean(),evidence_precision=g.evidence_precision.mean(),evidence_recall=g.evidence_recall.mean(),evidence_f1=g.evidence_f1.mean(),counterevidence_recall=g.counterevidence_recall.mean(),both_recognition=(both.status=='both').mean() if len(both) else None,insufficient_recognition=(known.status=='insufficient').mean() if len(known) else None,confident_wrong_under_insufficient=((known.confidence>=.8)&(known.status!='insufficient')).mean() if len(known) else None,mean_seconds=g.seconds.mean(),mean_input_tokens=g.input_tokens.mean(),mean_output_tokens=g.output_tokens.mean(),brier_status=(((valid.confidence-valid.status_correct.astype(float))**2).mean()) if len(valid) else None))
    return frame,pd.DataFrame(summaries)

def stability(rows,include_unreviewed=False):
    base={(r['mode'],r['question_id']):r for r in rows if r['variant']=='base'}
    out=[]
    for row in rows:
        if row['variant']=='base' or (not row.get('human_reviewed') and not include_unreviewed): continue
        b=base.get((row['mode'],row['question_id']))
        if not b: continue
        x,y=b.get('parsed'),row.get('parsed')
        valid=bool(x and y)
        evx=set((x or {}).get('evidence_ids',[]));evy=set((y or {}).get('evidence_ids',[]))
        out.append(dict(mode=row['mode'],variant=row['variant'],question_id=row['question_id'],both_valid=valid,status_changed=(x['status']!=y['status']) if valid else None,evidence_jaccard=len(evx&evy)/len(evx|evy) if valid and (evx|evy) else (1. if valid else None),expected_status=row['gold']['gold_status'],expected_reaction=bool(y and y['status']==row['gold']['gold_status'])))
    return pd.DataFrame(out)
