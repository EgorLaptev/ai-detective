"""Одинаковая токенизация запроса и утверждений; стабильные ties."""
import re
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r'[\w]+', text.casefold().replace('ё','е'),flags=re.UNICODE)

def retrieve(question,claims,k=15):
    if not claims or k<=0: return []
    corpus=[tokenize(c['text']+' '+c['evidence']) for c in claims]
    scores=BM25Okapi(corpus).get_scores(tokenize(question))
    order=sorted(range(len(claims)),key=lambda i:(-float(scores[i]),claims[i]['claim_id']))
    return [claims[i] for i in order[:min(k,15)]]
