import numpy as np
from numpy.linalg import norm
embedding_store = {}

#core similarity func
def cosine_similarity(emb1,emb2):
    a=np.array(emb1)
    b=np.array(emb2)
    return float(np.dot(a,b)/(norm(a)*norm(b)))

#in-memory store(just for prototype)
def  add_person(name,embedding):
    embedding_store[name]=np.array(embedding)

def get_all():
    return embedding_store


#top-k similar embeddings
def find_matches(query_embedding,top_k=3,threshold=0.4):
    results = []
    for name,emb in embedding_store.items():
        score=cosine_similarity(query_embedding,emb)
        results.append({"name":name,"similarity":round(score*100,2)})

    results.sort(key=lambda x:x["similarity"],reverse=True)
    return [r for r in results[:top_k] if r["similarity"]>=threshold*100]