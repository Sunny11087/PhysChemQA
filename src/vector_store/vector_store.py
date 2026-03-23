import numpy as np
from typing import List, Dict, Any, Optional
import faiss
import os
import pickle
from datetime import datetime

class VectorStore:
    """向量存储类，用于存储和检索文档的向量表示"""
    
    def __init__(self, embedding_dim: int = 768):
        """初始化向量存储"""
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.embeddings = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的嵌入向量"""
        import hashlib
        
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        hash_object = hashlib.md5(text.encode())
        hash_bytes = hash_object.digest()
        
        for i in range(min(16, self.embedding_dim)):
            embedding[i] = float(hash_bytes[i]) / 255.0
        
        if self.embedding_dim > 16:
            for i in range(16, self.embedding_dim):
                embedding[i] = np.sin(i * embedding[i % 16])
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def add_documents(self, documents: List[Any]):
        """添加文档到向量存储"""
        if not documents:
            return
        
        # 兼容不同格式的文档
        texts = []
        processed_docs = []
        
        for doc in documents:
            if hasattr(doc, 'page_content'):  # LangChain Document对象
                texts.append(doc.page_content)
                processed_docs.append({
                    "content": doc.page_content,
                    "metadata": getattr(doc, 'metadata', {})
                })
            elif isinstance(doc, dict) and "content" in doc:  # 字典格式
                texts.append(doc["content"])
                processed_docs.append(doc)
            else:
                content = str(doc)
                texts.append(content)
                processed_docs.append({"content": content, "metadata": {}})
        
        # 修复：正确创建嵌入向量数组
        new_embeddings = np.array([self._get_embedding(text) for text in texts], dtype=np.float32)
        
        # 维度检查
        if new_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"嵌入维度不匹配: 期望 {self.embedding_dim}, 实际 {new_embeddings.shape[1]}")
        
        # 添加到FAISS索引
        self.index.add(new_embeddings)
        
        # 保存文档
        self.documents.extend(processed_docs)
        
        # 更新嵌入向量存储
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """基于相似度搜索文档"""
        if not self.documents:
            return []
        
        query_embedding = self._get_embedding(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(1.0 / (1.0 + distances[0][i]))
                results.append(doc)
        
        return results
    
    # ... save 和 load 方法保持不变 ...

    
    def save(self, directory: str, name: str = None):
        """保存向量存储到磁盘
        
        Args:
            directory: 保存目录
            name: 保存名称，默认使用当前时间戳
        """
        if name is None:
            name = f"vector_store_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(directory, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(directory, f"{name}.index"))
        
        # 保存文档和嵌入向量
        with open(os.path.join(directory, f"{name}.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "embeddings": self.embeddings,
                "embedding_dim": self.embedding_dim
            }, f)
    
    @classmethod
    def load(cls, directory: str, name: str) -> "VectorStore":
        """从磁盘加载向量存储
        
        Args:
            directory: 加载目录
            name: 加载名称
            
        Returns:
            加载的向量存储实例
        """
        # 加载文档和嵌入向量
        with open(os.path.join(directory, f"{name}.pkl"), "rb") as f:
            data = pickle.load(f)
        
        # 创建实例
        vector_store = cls(embedding_dim=data["embedding_dim"])
        vector_store.documents = data["documents"]
        vector_store.embeddings = data["embeddings"]
        
        # 加载FAISS索引
        vector_store.index = faiss.read_index(os.path.join(directory, f"{name}.index"))
        
        return vector_store