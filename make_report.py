"""Короткий черновик только по фактическим записям."""
from pathlib import Path
from html import escape
from src.evaluation import summarize,stability

def make_report(rows,output):
    if not rows:raise ValueError('Нет фактических ответов')
    output=Path(output);_,summary=summarize(rows)
    baseline=[r for r in rows if r['variant']=='base']
    n=len({(r['question_id'],r['mode']) for r in baseline})
    reviewed=sum(r['variant']!='base' and r.get('human_reviewed',False) for r in rows)
    lines=['Лабораторная работа №6. AI-детектив','Лаптев Егор Игоревич (408981), Поляков Александр Владимирович (409367), К3439',
    'Черновик по фактическим записям. Семантическую корректность объяснений проверяют участники.',
    'Цель: сравнить полный контекст, BM25 top-15 и p-adic группировку одинакового набора утверждений на вымышленном инциденте.',
    'Данные: 10 документов, 5194 слова, 36 утверждений, 12 вопросов шести типов. Рабочая истина и gold модели недоступны.',
    'Метод: единые модель, системный промпт и параметры. A читает документы с маркерами цитат; B плоский список; C тот же список по ветвям и времени. Время документа не заменяет время события.',
    f'Основной набор: {n}/36 запросов. Проверенные stress-запросы: {reviewed}. Ошибки формата входят в знаменатель accuracy и получают нулевой evidence F1.']
    base=summary[summary.variant=='base'] if not summary.empty else summary
    mode_rows={}
    for _,r in base.iterrows():
        mode_rows[r['mode']]=r
        lines.append(f"Режим {r['mode']}: n={int(r['n'])}, accuracy={r['status_accuracy']:.3f}, evidence F1={r['evidence_f1']:.3f}, валидность={r['valid_rate']:.3f}, время={r['mean_seconds']:.2f} с, входные токены={r['mean_input_tokens']:.0f}.")
    if all(m in mode_rows for m in 'ABC'):
        a,b,c=[mode_rows[m] for m in 'ABC']
        lines.extend([f"Retrieval: accuracy B−A = {b['status_accuracy']-a['status_accuracy']:+.3f}. Это наблюдение на 12 вопросах, не общее преимущество метода.",
        f"Группировка: evidence F1 C−B = {c['evidence_f1']-b['evidence_f1']:+.3f}; распознавание both B={b['both_recognition']:.3f}, C={c['both_recognition']:.3f}.",
        f"Распознавание insufficient: A={a['insufficient_recognition']:.3f}, B={b['insufficient_recognition']:.3f}, C={c['insufficient_recognition']:.3f}. Уверенным считается ответ с confidence >= 0.8; точные доли сохранены в summary.csv."])
    stable=stability(rows)
    if stable.empty:lines.append('Выводы о перестановке, удалении и добавлении противоречия отложены до проверки вариантов двумя участниками. Сырые ответы сохраняются.')
    else:
        for (m,v),g in stable.groupby(['mode','variant']):
            lines.append(f"Стресс {v}, режим {m}: n={len(g)}, ожидаемая реакция={g.expected_reaction.mean():.3f}, смена статуса среди валидных пар={g.status_changed.mean():.3f}, Jaccard evidence={g.evidence_jaccard.mean():.3f}.")
    lines.extend(['Confidence: Brier для правильности статуса считается по валидным ответам вместе с долей валидности. Малый набор не позволяет доказать калиброванность.',
    'P-adic дерево: p=3, сначала источник, затем статус, затем время события. При k общих начальных уровнях расстояние равно 3^(-k), одинаковые пути имеют расстояние 0. Источник выше времени, чтобы сначала видеть происхождение основания. Обратный порядок удобнее для хронологии, но смешивает типы источников. Числовой код не используется как евклидово расстояние.',
    'Ограничения: валидатор проверяет формат и ссылки, но не истинность свободного объяснения. Корпус создан с ИИ и требует проверки участниками. Пропуск BM25 ухудшает B/C независимо от группировки. В Q11/Q12 одним изменением добавляется документ с двумя противоположными позициями, поскольку исходный insufficient не содержит второй стороны конфликта.',
    'Где структура помогает или мешает: конкретные примеры выбираются после чтения raw-ответов. Ветви могут упростить сопоставление источников, но разнести близкие по времени события. Без ручного анализа утверждать причину различий нельзя.',
    'Вклад: план распределения в CONTRIBUTIONS.md. Фактический вклад и совместную проверку эталонов участники подтверждают перед сдачей.'])
    (output/'report_draft.md').write_text('\n\n'.join(lines),encoding='utf-8')
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
    from matplotlib.font_manager import findfont
    pdfmetrics.registerFont(TTFont('DejaVu',findfont('DejaVu Sans')))
    style=getSampleStyleSheet()['Normal'];style.fontName='DejaVu';style.fontSize=11;style.leading=16
    story=[]
    for line in lines:story.extend([Paragraph(escape(line),style),Spacer(1,10)])
    SimpleDocTemplate(str(output/'report_draft.pdf'),leftMargin=45,rightMargin=45,topMargin=40,bottomMargin=40).build(story)
    print('Создан черновик по фактическим ответам: report_draft.md/pdf')
