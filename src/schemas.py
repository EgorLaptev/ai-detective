"""Проверяем формат и ссылки. Истинность свободного текста отдельно оценивает человек."""
import json
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator

class Answer(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    question_id:str
    status:Literal['entailed','contradicted','both','insufficient']
    answer:str|None
    evidence_ids:list[str]
    counterevidence_ids:list[str]
    confidence:float=Field(ge=0,le=1)
    explanation:str=Field(min_length=3)
    @model_validator(mode='after')
    def logical_contract(self):
        if len(set(self.evidence_ids))!=len(self.evidence_ids) or len(set(self.counterevidence_ids))!=len(self.counterevidence_ids):
            raise ValueError('Дубли ссылок')
        if set(self.evidence_ids)&set(self.counterevidence_ids): raise ValueError('Одна ссылка с двух сторон')
        if self.status=='both' and not (self.evidence_ids and self.counterevidence_ids):
            raise ValueError('both требует оснований обеих сторон')
        if self.status=='insufficient' and self.answer is not None:
            raise ValueError('insufficient требует answer=null: нельзя выдавать догадку как ответ')
        if self.status!='insufficient' and (not self.evidence_ids or not self.answer):
            raise ValueError('Определённый ответ требует текста и оснований')
        return self

def parse_answer(raw,question_id,available_ids):
    # Не исправляем JSON незаметно и не удаляем ошибочные ссылки.
    answer=Answer.model_validate_json(raw)
    if answer.question_id!=question_id: raise ValueError('Неверный question_id')
    unknown=(set(answer.evidence_ids)|set(answer.counterevidence_ids))-set(available_ids)
    if unknown: raise ValueError(f'Недоступные ссылки: {sorted(unknown)}')
    return answer
