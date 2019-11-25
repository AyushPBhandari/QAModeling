import parser
import spacy
from spacy.symbols import *
import numpy as np

def get_entity_features(ctxs, qs):
    num_entities = 18 # spacy has 18 entities
    max_ctx_size = len(max(ctxs))
    nlp = spacy.load('en')

    features = []
    for ctx in ctxs:
        ctx_ent_c = [] 
        for sent in ctx:
            doc = nlp(sent)
            sent_ent_c = np.zeros((num_entities,))
            for ent in doc.ents:
                # spacy entity symbols range from 380 to 397
                i = ent.label - 380
                if ent.label == 9191306739292312949:
                    i = 2 # FAC entity
                if i < 18:
                    sent_ent_c[i] += 1
                else:
                    print("Found entity:", ent.label_, " = ", ent.label)
            ctx_ent_c.append(sent_ent_c)
        # add padding
        while len(ctx_ent_c) < max_ctx_size:
            sent_ent_c = np.zeros((num_entities,))
            ctx_ent_c.append(sent_ent_c)
        ctx_ent_c = np.asarray(ctx_ent_c).flatten()
        features.append(ctx_ent_c)
    
    features = np.asarray(features)
    return features
    


        
        
            