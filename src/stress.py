"""Один фактор на вариант. Фикстуры требуют проверки обоими участниками."""
from copy import deepcopy
import random

def permutation(documents,claims):
    docs=deepcopy(documents)
    random.Random(42).shuffle(docs)
    return docs,deepcopy(claims)

def remove_premise(documents,claims,claim_id):
    docs,cs=deepcopy(documents),deepcopy(claims)
    claim=next(c for c in cs if c['claim_id']==claim_id)
    d=next(d for d in docs if d['document_id']==claim['document_id'])
    # Удаляем цитату из полного текста, а не только запись retrieval-индекса.
    if d['text'].count(claim['evidence'])!=1: raise ValueError('Неуникальная посылка')
    d['text']=d['text'].replace(claim['evidence'],'[Фрагмент изъят из экспериментальной копии.]')
    return docs,[c for c in cs if c['claim_id']!=claim_id]

ADDITIONS={
'Q01': [('system','18:42:00 PUT artifact=demo-r17.apk bucket=demo-private result=failure')],
'Q02': [('document','Пакет demo-r17.apk предназначен исключительно для iOS и не является пакетом Android.')],
'Q03': [('system','18:40:00 — право чтения svc-demo окончательно отозвано; после этой отметки оно не выдавалось.')],
'Q04': [('system','SHA256 external-copy.apk = aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')],
'Q05': [('system','18:40:00 — единственная выдача персонального права artifact:read субъекту lev.')],
'Q06': [('system','18:45:00 — единственное завершение подписи demo-r17.apk.')],
'Q07': [('rule','E17: исключение для demo-r17.apk не действует в 18:42:00.')],
'Q08': [('rule','R3: Исключение E17 разрешает передачу demo-r17.apk анонимному внешнему получателю.')],
'Q09': [('statement','В 18:43 Лев был в переговорной и отсутствовал в серверной.')],
'Q10': [('statement','В 18:43 контрольный индикатор в коридоре был красным, а не синим.')],
# У исходно insufficient нет положительной/отрицательной посылки.
# Один добавленный документ содержит два равноправных несовместимых свидетельства.
'Q11': [('statement','Свидетель Роман: запрос GET svc-demo в 18:44 лично выполнил Лев.'),('statement','Свидетель Ира: запрос GET svc-demo в 18:44 лично выполнила Мира, Лев его не выполнял.')],
'Q12': [('statement','Свидетель Роман: инициатор чтения svc-demo прямо сообщил о намерении распространить сборку вне команды.'),('statement','Свидетель Ира: инициатор чтения svc-demo прямо сообщил, что не намерен распространять сборку вне команды.')],
}

def add_contradiction(documents,claims,qid):
    docs,cs=deepcopy(documents),deepcopy(claims)
    rows=ADDITIONS[qid]
    doc_id='DX'+qid[1:]
    doc=dict(document_id=doc_id,title='Дополнительный источник для контролируемого конфликта',source_type=rows[0][0],author='Независимый источник сопоставимой надёжности',created_at='2026-03-14T21:00:00+03:00',text='Независимая версия того же события; оснований предпочесть её исходной версии нет.\n'+'\n'.join(t for _,t in rows))
    docs.append(doc)
    new_ids=[]
    for i,(source,text) in enumerate(rows,1):
        cid=f'CX{qid[1:]}{i}'
        cs.append(dict(claim_id=cid,text=text,document_id=doc_id,evidence=text,status='disputed',event_time=None,padic_path=[{'system':0,'document':0,'rule':2,'statement':1}[source],2,2]))
        new_ids.append(cid)
    return docs,cs,new_ids

def variants(documents,claims,q):
    yield 'base',deepcopy(documents),deepcopy(claims),deepcopy(q),{'factor':'none','review_required':False}
    ds,cs=permutation(documents,claims)
    yield 'permutation',ds,cs,deepcopy(q),{'factor':'document_order','seed':42,'review_required':True}
    if q.get('remove_claim_id'):
        ds,cs=remove_premise(documents,claims,q['remove_claim_id'])
        gold=deepcopy(q);gold.update(gold_status='insufficient',gold_answer=None,gold_evidence_ids=[c for c in q['gold_evidence_ids'] if c!=q['remove_claim_id']],gold_counterevidence_ids=[])
        yield 'removal',ds,cs,gold,{'factor':'remove_one_premise','removed_id':q['remove_claim_id'],'review_required':True}
    ds,cs,new_ids=add_contradiction(documents,claims,q['question_id'])
    gold=deepcopy(q);status=q['gold_status']
    if status=='entailed': ev,co=q['gold_evidence_ids'],new_ids
    elif status=='contradicted': ev,co=new_ids,q['gold_evidence_ids']
    elif status=='both': ev,co=q['gold_evidence_ids'],q['gold_counterevidence_ids']+new_ids
    else: ev,co=[new_ids[0]],[new_ids[1]]
    gold.update(gold_status='both',gold_answer='Сохраняются несовместимые основания сопоставимой надёжности.',gold_evidence_ids=ev,gold_counterevidence_ids=co)
    yield 'contradiction',ds,cs,gold,{'factor':'add_one_conflicting_document','added_ids':new_ids,'review_required':True}
