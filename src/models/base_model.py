# src/models/base_model.py

from abc import ABC, abstractmethod

class BaseModelAdapter(ABC):
    """
    所有模型类的抽象基类。
    任何新的模型实现都应该继承这个类，并实现 generate_stream 方法。
    """
    @abstractmethod
    def generate_stream(self, question: str, context: list):
        """
        根据问题和上下文生成流式回答的抽象方法。
        子类必须实现此方法。
        """
        pass
