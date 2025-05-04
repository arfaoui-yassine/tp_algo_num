import numpy as np
from numpy.linalg import norm, qr

class MatrixDecompositionSearchEngine:
    def __init__(self, rank=2, max_iterations=100, tolerance=1e-8):
        self.terms = []
        self.documents = []
        self.matrix = None
        self.rank = rank
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.U = None
        self.S = None
        self.VT = None
    
    def build_matrix(self, documents):
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
    
    def bidiagonalize(self, D):
        m, n = D.shape
        if m < n:
            raise ValueError("Matrix must have more rows than columns (m >= n)")
        
        V = np.zeros((n, n))
        U = np.zeros((m, n))
        B = np.zeros((n, n))
        
        v = np.random.rand(n)
        v /= norm(v)
        V[:, 0] = v
        
        u = D @ v
        alpha = norm(u)
        B[0, 0] = alpha
        if alpha > 1e-10:
            u /= alpha
        U[:, 0] = u
        
        for i in range(1, n):
            v_new = D.T @ U[:, i-1] - alpha * V[:, i-1]
            for j in range(i):
                v_new -= (v_new @ V[:, j]) * V[:, j]
            
            beta = norm(v_new)
            if beta > 1e-10:
                v_new /= beta
            
            u_new = D @ v_new - beta * U[:, i-1]
            for j in range(i):
                u_new -= (u_new @ U[:, j]) * U[:, j]
            
            alpha = norm(u_new)
            if alpha > 1e-10:
                u_new /= alpha
            
            if i < n:
                B[i-1, i] = beta
                B[i, i] = alpha
            V[:, i] = v_new
            U[:, i] = u_new
        
        return U, B, V.T
    
    def qr_iteration(self, B):
        n = B.shape[0]
        S = B.copy()
        U_accum = np.eye(n)
        V_accum = np.eye(n)
        
        for _ in range(self.max_iterations):
            Q, R = qr(S.T)
            Q_tilde, S_new = qr(R.T)
            
            U_accum = U_accum @ Q_tilde
            V_accum = V_accum @ Q
            
            off_diag = np.sum(np.abs(np.diag(S_new, k=1)))
            if off_diag < self.tolerance:
                break
            
            S = S_new
        
        sigma = np.diag(S)
        order = np.argsort(sigma)[::-1]
        sigma = sigma[order]
        U_accum = U_accum[:, order]
        V_accum = V_accum[:, order]
        
        return sigma, U_accum, V_accum
    
    def compute_decomposition(self):
        if self.matrix is None:
            raise ValueError("Matrix not built")
        
        D = self.matrix
        m, n = D.shape
        
        U_b, B, VT_b = self.bidiagonalize(D)
        sigma, U_qr, V_qr = self.qr_iteration(B)
        
        self.U = U_b @ U_qr
        self.VT = V_qr.T @ VT_b
        self.S = sigma
        
        return self.U, np.diag(self.S), self.VT
    
    def reduced_approximation(self):
        if self.U is None or self.S is None or self.VT is None:
            self.compute_decomposition()
        
        Sigma_k = np.diag(self.S[:self.rank])
        U_k = self.U[:, :self.rank]
        VT_k = self.VT[:self.rank, :]
        
        return U_k @ Sigma_k @ VT_k
    
    def query(self, query_terms, threshold=0.5):
        if self.U is None or self.S is None or self.VT is None:
            self.compute_decomposition()
        
        q = np.zeros(len(self.terms))
        for term in query_terms:
            if term in self.terms:
                term_idx = self.terms.index(term)
                q[term_idx] = 1
        
        q_k = self.U[:, :self.rank].T @ q
        
        scores = []
        for doc_idx in range(len(self.documents)):
            d_k = self.S[:self.rank] * self.VT[:self.rank, doc_idx]
            dot_product = np.dot(q_k, d_k)
            q_norm = norm(q_k)
            d_norm = norm(d_k)
            
            if q_norm == 0 or d_norm == 0:
                score = 0
            else:
                score = dot_product / (q_norm * d_norm)
            
            scores.append((self.documents[doc_idx], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        relevant_docs = [(doc, score) for doc, score in scores if score >= threshold]
        
        return relevant_docs

