import numpy as np

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
    





  
documents = [
    ("Doc 1", ["Croissance", "PIB", "Investissement"]),
    ("Doc 2", ["Inflation", "Monnaie", "Dépression"]),
    ("Doc 3", ["Commerce", "Exportation", "Croissance"]),
    ("Doc 4", ["Emploi", "Chomage", "Salaires"]),
    ("Doc 5", ["Impots", "Fiscalité", "Revenu"]),
    ("Doc 6", ["Géologie", "Faille", "Tremblement"]),
    ("Doc 7", ["Volcan", "Séisme", "Plaque tectonique"]),
    ("Doc 8", ["Dépression", "Bassin", "Erosion"]),
    ("Doc 9", ["Stratigraphie", "Couches", "Roche"]),
    ("Doc 10", ["Gisement", "Forage", "Bassin"]),
]


vsm = VectorSpaceModel()
vsm.build_term_document_matrix(documents)


query1 = ["Dépression", "Croissance"]

query = ["Bassin", "Fiscalité"]


results = vsm.process_query(query, threshold=0.4)

print("Termes dans le modèle:", vsm.terms)

print("\nRésultats pour la requête:", query)
for doc, score in results:
    print(f"- {doc}: score = {score:.2f}")