# src/models/openai_model.py

import os
from openai import OpenAI
from .base_model import BaseModelAdapter  # <--- 修改导入

class OpenAIModelAdapter(BaseModelAdapter):  # <--- 修改类名和继承
    def __init__(self, model_name: str, temperature: float = 0.7):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("未找到环境变量 OPENAI_API_KEY，请设置它。")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature

    def generate_stream(self, question: str, context: list):
        # ... (此部分代码保持不变) ...
        prompt = self._build_prompt(question, context)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的文档分析与学习助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def _build_prompt(self, question: str, context: list) -> str:
        # ... (此部分代码保持不变) ...
        context_str = "\n\n".join([doc.page_content for doc in context])
        prompt = f"""
        请根据以下提供的背景信息来回答用户的问题。
        如果信息不足，请诚实地说明，不要编造答案。
        
        背景信息：
        ---
        {context_str}
        ---
        
        用户问题：{question}
        
        请回答：
        """
        return prompt
class ZhipuModelAdapter(BaseModelAdapter):
    # ... (__init__ 和 generate_stream 保持不变) ...

    def generate_stream_with_profile(self, question: str, profile_text: str):
        """
        【新增方法】基于知识画像（作为系统提示）生成流式回答。
        """
        system_prompt = f"""
        你是一个专业的AI学习导师。现在，你需要根据以下关于一位学生的详细知识画像，来回答他/她提出的问题。
        你的回答必须严格基于画像信息，做到个性化、有针对性，并给出具体、可操作的建议。
        请不要编造画像中没有的信息。

        --- 学生知识画像 ---
        {profile_text}
        --- 画像结束 ---
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
            stream=True,
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content