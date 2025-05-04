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