"""Конечное трёхуровневое дерево, а не евклидово расстояние кодов."""
from itertools import groupby

def validate_path(path):
    if len(path) != 3 or any(type(a) is not int or a not in (0,1,2) for a in path):
        raise ValueError('Путь должен содержать три цифры 0, 1, 2')
    return tuple(path)

def encode(path):
    a,b,c=validate_path(path)
    return a+3*b+9*c

def distance(x,y):
    x,y=validate_path(x),validate_path(y)
    k=0
    for a,b in zip(x,y):
        if a!=b: break
        k+=1
    return 0.0 if k==3 else 3.0**(-k)

def groups(claims):
    ordered=sorted(claims,key=lambda c:(validate_path(c['padic_path']),c.get('event_time') or '9999',c['claim_id']))
    return [(path,list(items)) for path,items in groupby(ordered,key=lambda c:tuple(c['padic_path']))]
