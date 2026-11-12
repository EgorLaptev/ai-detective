"""Два транспорта для одной модели. Без скрытой подмены при ошибке."""
import os
import requests

class LLMUnavailable(RuntimeError):
    pass


_HTTP_HINTS = {
    400: 'проверьте имя модели и параметры запроса.',
    401: 'проверьте LLM_API_KEY в настройках приложения.',
    402: 'провайдер требует кредиты. Проверьте, что выбрана бесплатная модель.',
    403: 'провайдер запретил запрос. Проверьте настройки доступа.',
    404: 'проверьте LLM_BASE_URL и доступность выбранной модели.',
    408: 'провайдер не ответил вовремя. Повторите запрос позже.',
    429: 'превышен лимит запросов. Повторите позже или выберите другую бесплатную модель.',
    502: 'модель или её провайдер не ответили. Повторите запрос позже.',
    503: 'для выбранной модели сейчас нет доступного провайдера. Попробуйте позже или выберите другую модель.',
}


def _http_message(status):
    hint = _HTTP_HINTS.get(status, 'сервер модели отклонил запрос.')
    return f'Сервер модели вернул HTTP {status}: {hint}'


class OpenAICompatible:
    def __init__(self, model='Qwen/Qwen2.5-7B-Instruct', base_url=None):
        self.model = model
        self.base_url = base_url or os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')

    def __call__(self, messages):
        try:
            response = requests.post(
                self.base_url.rstrip('/') + '/chat/completions',
                headers={'Authorization': 'Bearer ' + os.getenv('LLM_API_KEY', 'local-key')},
                json=dict(model=self.model, messages=messages, temperature=0, max_tokens=384, seed=42),
                timeout=300,
            )
            response.raise_for_status()
            body = response.json()
            if body.get('error'):
                code = body['error'].get('code') if isinstance(body['error'], dict) else None
                if type(code) is int and 400 <= code <= 599:
                    raise LLMUnavailable(_http_message(code))
                raise LLMUnavailable('Сервер модели сообщил об ошибке.')
            choice = body['choices'][0]
            raw = choice['message'].get('content')
            if not isinstance(raw, str) or not raw.strip():
                if choice.get('finish_reason') == 'length':
                    raise LLMUnavailable('Модель исчерпала лимит токенов до итогового ответа.')
                raise LLMUnavailable('Модель не вернула текст ответа. Повторите запрос или выберите другую модель.')
            usage = body.get('usage') or {}
            return dict(raw=raw, input_tokens=usage.get('prompt_tokens'), output_tokens=usage.get('completion_tokens'))
        except requests.HTTPError as e:
            if e.response is not None:
                raise LLMUnavailable(_http_message(e.response.status_code)) from e
            raise LLMUnavailable('Сервер модели отклонил запрос.') from e
        except (requests.RequestException, KeyError, ValueError, TypeError, AttributeError, IndexError) as e:
            raise LLMUnavailable(f'Недоступен LLM endpoint: {type(e).__name__}') from e


class TransformersLLM:
    def __init__(self,model='Qwen/Qwen2.5-7B-Instruct',revision=None):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
        if not torch.cuda.is_available(): raise LLMUnavailable('Для полного контекста требуется CUDA GPU; выберите GPU в Colab')
        self.torch=torch;self.model=model
        self.tokenizer=AutoTokenizer.from_pretrained(model,revision=revision)
        self.engine=AutoModelForCausalLM.from_pretrained(model,revision=revision,device_map='auto',torch_dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16),attn_implementation='sdpa')
        self.engine.eval()
        self.revision=getattr(self.engine.config,'_commit_hash',revision)
    def __call__(self,messages):
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(text,return_tensors='pt',truncation=False).to(self.engine.device)
        n=inputs['input_ids'].shape[-1]
        if n+384>32768: raise ValueError(f'Контекст {n}+384 превышает 32768; усечение запрещено')
        with self.torch.inference_mode():
            out=self.engine.generate(**inputs,max_new_tokens=384,do_sample=False,use_cache=True,cache_implementation='offloaded',pad_token_id=self.tokenizer.eos_token_id)
        generated=out[0,n:]
        return dict(raw=self.tokenizer.decode(generated,skip_special_tokens=True),input_tokens=n,output_tokens=len(generated))
