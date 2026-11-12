from pathlib import Path
import json,io,zipfile
import pandas as pd
import streamlit as st
from src.io import ROOT,load_data,read_jsonl
from src.context_builder import build_context,select
from src.stress import variants
from src.experiment import call_one
from src.llm import OpenAICompatible
from src.evaluation import summarize,stability

st.set_page_config(page_title='Ладога: AI-детектив',page_icon='🔎',layout='wide')
st.title('Ладога: AI-детектив')
st.caption('ЛР6 · Егор Лаптев и Александр Поляков · К3439 · Все персонажи и события вымышлены')
with st.sidebar:
    st.header('Досье')
    upload=st.file_uploader('Загрузить ZIP с data/documents.jsonl, claims.jsonl и questions.jsonl',type=['zip'])
    result_path=st.text_input('Каталог сохранённого прогона',str(ROOT/'results/run_01'))
    st.divider()
    st.caption('Модель отвечает через OpenAI-совместимый endpoint. Настройки задаются переменными окружения, ключ не сохраняется.')
    model=st.text_input('Модель','Qwen/Qwen2.5-7B-Instruct')
try:
    if upload:
        with zipfile.ZipFile(io.BytesIO(upload.getvalue())) as z:
            objects=[]
            for name in ('documents','claims','questions'):
                matches=[x for x in z.namelist() if x.endswith('data/'+name+'.jsonl')]
                if len(matches)!=1:raise ValueError('Не найден единственный data/'+name+'.jsonl')
                objects.append([json.loads(line) for line in z.read(matches[0]).decode().splitlines() if line.strip()])
            documents,claims,questions=objects
        from validate_data import validate
        validate(documents,claims,questions)
    else: documents,claims,questions=load_data()
except Exception as e:
    st.error(f'Ошибка досье: {e}');st.stop()
q_label=st.selectbox('Вопрос',[q['question_id']+' · '+q['question'] for q in questions])
q=next(q for q in questions if q['question_id']==q_label.split(' · ')[0])
cols=st.columns([1,2])
with cols[0]:mode=st.radio('Режим',['A','B','C'],horizontal=True,format_func=lambda m:{'A':'A · Полное досье','B':'B · BM25','C':'C · p-adic'}[m])
with cols[1]:variant=st.selectbox('Контролируемое изменение',['base','permutation','removal','contradiction'],format_func=lambda v:{'base':'Исходное досье','permutation':'Переставить документы (seed 42)','removal':'Удалить необходимую посылку','contradiction':'Добавить подготовленное противоречие'}[v])
options={v:(ds,cs,g,meta) for v,ds,cs,g,meta in variants(documents,claims,q)}
if variant not in options:
    st.info('Удаление необходимой посылки предусмотрено для multi-hop и временных вопросов. Выберите Q03–Q06.');st.stop()
ds,cs,gold,meta=options[variant]
if variant=='removal':
    chosen=st.selectbox('Удаляемое утверждение',[c['claim_id'] for c in claims],index=[c['claim_id'] for c in claims].index(q['remove_claim_id']))
    if chosen!=q['remove_claim_id']:
        from src.stress import remove_premise
        ds,cs=remove_premise(documents,claims,chosen)
        st.warning('Произвольное удаление: ожидаемый статус необходимо разметить отдельно. Этот вариант не входит в подготовленные метрики.')
selected=select(q['question'],cs);context,ids=build_context(mode,ds,cs,selected)
st.caption(f'Документов: {len(ds)} · Доступных утверждений: {len(ids)} · Символов контекста: {len(context):,}')
tabs=st.tabs(['Ответ','Контекст и ветви','Сравнение результатов','Эталон и проверка'])
with tabs[0]:
    key=(q['question_id'],mode,variant,context)
    result_file=Path(result_path)/'predictions.jsonl'
    rows=read_jsonl(result_file) if result_file.exists() else []
    saved=next((r for r in rows if r['question_id']==q['question_id'] and r['mode']==mode and r['variant']==variant and r.get('context_hash')==__import__('src.io',fromlist=['fingerprint']).fingerprint(context)),None)
    if st.button('Запросить модель',type='primary'):
        with st.spinner('Модель читает досье…'):
            res=call_one(OpenAICompatible(model, live_demo=True), (ROOT/'prompts/system.txt').read_text(),q,mode,ds,cs,selected)
            st.session_state['live_result']=(key,res)
    live=st.session_state.get('live_result')
    active=live[1] if live and live[0]==key else saved
    if not active:st.info('Ответа ещё нет. Можно запросить модель или открыть сохранённый прогон.')
    elif active.get('parsed'):
        a=active['parsed'];c1,c2=st.columns(2);c1.metric('Статус',a['status']);c2.metric('Уверенность',f"{a['confidence']:.0%}")
        st.write(a['answer'] if a['answer'] is not None else 'Недостаточно данных');st.write(a['explanation'])
        left,right=st.columns(2)
        for column,field,label in [(left,'evidence_ids','Основания'),(right,'counterevidence_ids','Контрдоказательства')]:
            with column:
                st.subheader(label)
                for cid in a[field]:
                    c=next((c for c in cs if c['claim_id']==cid),None)
                    if c:st.markdown(f"**{cid} · {c['document_id']}**  \n{c['evidence']}")
        st.caption(f"Время: {active['seconds']:.2f} с · Токены: {active.get('input_tokens','нет данных')} → {active.get('output_tokens','нет данных')}")
    elif active:st.error(active.get('error','Ошибка ответа'))
    if active:
        with st.expander('Исходный ответ модели'):st.code(active['raw'],language='json')
with tabs[1]:
    from src.padic import groups
    for path,items in groups(selected):
        with st.expander(f'Ветвь {list(path)} · {len(items)} утверждений'):
            for c in items:st.write(c['claim_id'],c['text'],c.get('event_time') or 'Время неизвестно/норма')
    with st.expander('Полный переданный контекст'):st.text(context)
    with st.expander('Документы в текущем порядке'):
        for d in ds:st.markdown('**'+d['document_id']+' · '+d['title']+'**');st.text(d['text'])
with tabs[2]:
    if rows:
        _,summary=summarize(rows)
        st.dataframe(summary,hide_index=True)
        stable=stability(rows)
        if not stable.empty:st.dataframe(stable,hide_index=True)
        st.caption('Непроверенные участниками стресс-варианты исключены из итоговых метрик.')
    else:st.info('Сохранённого эксперимента ещё нет. Здесь нет демонстрационных или вымышленных результатов.')
with tabs[3]:
    st.warning('Gold не передаётся модели. Эталоны и стресс-варианты должны проверить оба участника.')
    st.json(gold);st.json(meta)
