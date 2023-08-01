import itertools

from ems_knowledge import *

def generate_labels(protocol):
    return list(itertools.chain.from_iterable(list(ems_interventions.get(protocol).values())))
    
    