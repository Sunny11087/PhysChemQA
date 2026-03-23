# src/utils/helpers.py

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

def get_available_models():
    """
    返回一个字典，包含所有可用的大语言模型及其配置。
    """
    available_models = {
        # --- OpenAI 模型示例 (保留你原有的) ---
        "GPT-3.5 Turbo": {
            "type": "openai",
            "name": "gpt-3.5-turbo",
        },
        "GPT-4": {
            "type": "openai",
            "name": "gpt-4",
        },
        
        # --- 新增：智谱AI模型 ---
        "GLM-4 (智谱清言)": {
            "type": "zhipu",
            "name": "glm-4", # 模型名称，参考智谱API文档
        },
        "GLM-3-Turbo (智谱)": {
            "type": "zhipu",
            "name": "glm-3-turbo",
        }
    }
    
    # 检查API Key是否存在，如果不存在则移除对应模型选项
    if not os.getenv("ZHIPU_API_KEY"):
        print("警告：未找到 ZHIPU_API_KEY，智谱模型将不可用。")
        # 使用字典推导式过滤掉智谱模型
        available_models = {k: v for k, v in available_models.items() if v.get("type") != "zhipu"}
        
    if not os.getenv("OPENAI_API_KEY"):
        print("警告：未找到 OPENAI_API_KEY，OpenAI模型将不可用。")
        available_models = {k: v for k, v in available_models.items() if v.get("type") != "openai"}

    return available_models

def format_document_for_display(text: str) -> str:
    """
    格式化文档文本以便在界面上更好地显示。
    
    Args:
        text (str): 原始文档文本。
        
    Returns:
        str: 格式化后的文本。
    """
    if not isinstance(text, str):
        return "无效的文档内容"
        
    # 示例1：简单的去除首尾空白
    formatted_text = text.strip()
    
    # 示例2：将连续的多个空行替换为单个空行
    # import re
    # formatted_text = re.sub(r'\n\s*\n', '\n\n', formatted_text)
    
    # 示例3：如果文本是Markdown，你可能想在这里做一些基础的转换
    # formatted_text = formatted_text.replace('**', '').replace('*', '') # 简单移除加粗标记
    
    return formatted_text


# src/utils/helpers.py

# ... (你现有的 get_available_models 函数) ...

def create_empty_file(filepath: str):
    """
    创建一个空的文件。如果文件已存在，则不执行任何操作。
    
    Args:
        filepath (str): 要创建的文件的路径。
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # 使用 'x' 模式，仅在文件不存在时创建，避免覆盖
    try:
        with open(filepath, 'x') as f:
            pass
    except FileExistsError:
        # 文件已存在，这是预期的，忽略错误
        pass
    except Exception as e:
        # 其他错误，如权限问题，则抛出
        raise e
