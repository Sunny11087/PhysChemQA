# src/models/zhipu_model.py

import os
from zhipuai import ZhipuAI
from .base_model import BaseModelAdapter

# --- 只保留这一个类定义 ---
class ZhipuModelAdapter(BaseModelAdapter):
    def __init__(self, model_name: str, temperature: float = 0.7):
        """
        初始化智谱AI模型适配器。
        
        Args:
            model_name (str): 要使用的模型名称，例如 "glm-4"。
            temperature (float): 控制生成文本的随机性，范围 0.0-2.0。
        """
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature

    def generate_stream(self, question: str, context: list):
        """
        根据检索到的上下文和用户问题，生成流式回答。
        """
        prompt = self._build_prompt(question, context)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

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

    def _build_prompt(self, question: str, context: list) -> str:
        """
        根据检索到的上下文和用户问题，构建完整的提示词。
        """
        page_contents = []
        for doc in context:
            # 【核心修复点】兼容对象和字典两种格式
            if hasattr(doc, 'page_content'):
                page_contents.append(doc.page_content)
            elif isinstance(doc, dict):
                if 'page_content' in doc:
                    page_contents.append(doc['page_content'])
                elif 'text' in doc:
                    page_contents.append(doc['text'])
                elif 'content' in doc:
                    page_contents.append(doc['content'])
                else:
                    page_contents.append(str(doc))
            else:
                page_contents.append(str(doc))

        context_str = "\n\n".join(page_contents)
        
        prompt = f"""
        你是一个专业的学习助手，请根据以下提供的背景信息来回答用户的问题。
        如果信息不足，请诚实地说明，不要编造答案。
        
        背景信息：
        ---
        {context_str}
        ---
        
        用户问题：{question}
        
        请回答：
        """
        return prompt

# --- 删除第二个重复的类定义 ---
