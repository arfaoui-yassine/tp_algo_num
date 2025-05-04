import numpy as np
import time
import matplotlib.pyplot as plt

class VectorSpaceModel:
    def __init__(self):
        self.terms = []  
        self.documents = []  
        self.matrix = None  
    
    def build_term_document_matrix(self, documents):
        all_terms = set()
        for doc_name, terms in documents:
            all_terms.update(terms)
        
        self.terms = sorted(all_terms)
        self.documents = [doc_name for doc_name, _ in documents]
  
        num_terms = len(self.terms)
        num_docs = len(documents)
        self.matrix = np.zeros((num_terms, num_docs))
        
        for doc_idx, (_, terms) in enumerate(documents):
            for term in terms:
                if term in self.terms:
                    term_idx = self.terms.index(term)
                    self.matrix[term_idx, doc_idx] = 1
    
    def process_query(self, query_terms, threshold=0.8):
        if self.matrix is None:
            raise ValueError("La matrice termes-documents n'a pas été construite")
       
        query_vec = np.zeros(len(self.terms))
        for term in query_terms:
            if term in self.terms:
                term_idx = self.terms.index(term)
                query_vec[term_idx] = 1
     
        scores = []
        for doc_idx in range(len(self.documents)):
            doc_vec = self.matrix[:, doc_idx]
  
            dot_product = np.dot(query_vec, doc_vec)
            
            query_norm = np.linalg.norm(query_vec)
            doc_norm = np.linalg.norm(doc_vec)
            
            if query_norm == 0 or doc_norm == 0:
                score = 0
            else:
                score = dot_product / (query_norm * doc_norm)
            
            scores.append((self.documents[doc_idx], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = [(doc, score) for doc, score in scores if score >= threshold]
        
        return relevant_docs


# Simulation de l'évolution du temps d'exécution
execution_times = []
document_counts = range(5, 205, 5)  # Nd varie de 5 à 200 avec un pas de 5

for Nd in document_counts:
    Nt = 3 * Nd  # Nt est fixé à 3 fois Nd
    documents = [("Doc" + str(i), ["Term" + str(j) for j in range(Nt)]) for i in range(Nd)]
    query = ["Term" + str(i) for i in range(10)]  # Une requête avec 10 termes
    
    model = VectorSpaceModel()
    
    # Mesure du temps d'exécution
    start_time = time.time()
    model.build_term_document_matrix(documents)
    model.process_query(query, threshold=0.5)
    end_time = time.time()
    
    execution_times.append(end_time - start_time)

# Représentation graphique
plt.plot(document_counts, execution_times, marker='o')
plt.title("Évolution du temps d'exécution en fonction de Nd")
plt.xlabel("Nombre de documents (Nd)")
plt.ylabel("Temps d'exécution (secondes)")
plt.grid()
plt.show()