"""Два транспорта для одной модели. Без скрытой подмены при ошибке."""
import os,time
import requests

class LLMUnavailable(RuntimeError): pass

class OpenAICompatible:
    def __init__(self,model='Qwen/Qwen2.5-7B-Instruct',base_url=None):
        self.model=model
        self.base_url=base_url or os.getenv('LLM_BASE_URL','http://localhost:1234/v1')
    def __call__(self,messages):
        try:
            response=requests.post(self.base_url.rstrip('/')+'/chat/completions',headers={'Authorization':'Bearer '+os.getenv('LLM_API_KEY','local-key')},json=dict(model=self.model,messages=messages,temperature=0,max_tokens=384,seed=42),timeout=300)
            response.raise_for_status();body=response.json()
            usage=body.get('usage',{})
            return dict(raw=body['choices'][0]['message']['content'],input_tokens=usage.get('prompt_tokens'),output_tokens=usage.get('completion_tokens'))
        except (requests.RequestException,KeyError,ValueError) as e:
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
