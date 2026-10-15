import argparse
from src.experiment import run
from src.llm import OpenAICompatible,TransformersLLM
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['api','transformers'],default='api');p.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct');p.add_argument('--output',default='results/run_01');p.add_argument('--only-base',action='store_true');p.add_argument('--review');p.add_argument('--limit',type=int)
    a=p.parse_args();llm=(TransformersLLM if a.backend=='transformers' else OpenAICompatible)(a.model)
    run(llm,a.output,only_base=a.only_base,review_path=a.review,limit=a.limit)
