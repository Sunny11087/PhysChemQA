# src/models/model_factory.py

# 注意：这里的导入也需要更新
from .base_model import BaseModelAdapter
from .openai_model import OpenAIModelAdapter
from .zhipu_model import ZhipuModelAdapter

class ModelFactory:
    @staticmethod
    def get_model(model_type: str, model_name: str, temperature: float) -> BaseModelAdapter: # <--- 修改返回类型提示
        """
        根据模型类型和名称返回相应的模型适配器实例。
        """
        if model_type == "openai":
            return OpenAIModelAdapter(model_name=model_name, temperature=temperature) # <--- 修改
        elif model_type == "zhipu":
            return ZhipuModelAdapter(model_name=model_name, temperature=temperature) # <--- 修改
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
