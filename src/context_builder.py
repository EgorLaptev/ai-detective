"""В режиме A маркируются дословные цитаты, без пересказов и gold."""
from .padic import groups
from .retrieval import retrieve

def select(question,claims):
    return retrieve(question,claims)

def annotate(document,claims):
    text=document['text']
    for c in claims:
        if c['document_id']==document['document_id']:
            if c['evidence'] not in text: raise ValueError(f"Нет evidence {c['claim_id']}")
            text=text.replace(c['evidence'],f"[{c['claim_id']}] "+c['evidence'],1)
    return text

def claim_line(c):
    # Все поля идентичны в B/C. Статус источника - не статус ответа.
    return f"{c['claim_id']} | {c['document_id']} | status={c['status']} | time={c.get('event_time')} | path={c['padic_path']}\nФакт: {c['text']}\nЦитата: {c['evidence']}"

def build_context(mode,documents,claims,selected=None):
    if mode=='A':
        return '\n\n'.join(f"ДОКУМЕНТ {d['document_id']} | {d['title']} | type={d['source_type']} | created_at={d['created_at']}\n"+annotate(d,claims) for d in documents),[c['claim_id'] for c in claims]
    if selected is None: raise ValueError('B/C требуют общий selected')
    if mode=='B': return '\n\n'.join(claim_line(c) for c in selected),[c['claim_id'] for c in selected]
    if mode=='C':
        text='\n\n'.join(f'ВЕТВЬ {list(path)}\n'+'\n\n'.join(claim_line(c) for c in cs) for path,cs in groups(selected))
        return text,[c['claim_id'] for c in selected]
    raise ValueError(mode)
