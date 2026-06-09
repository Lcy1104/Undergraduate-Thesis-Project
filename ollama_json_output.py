from ollama import Client  # 导入Client类，而非直接导入顶层chat函数
from pydantic import BaseModel

# 1. 定义Country模型（正确缩进，顶格定义类，类内4个空格缩进）
class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]
    thinking_process:str

# 2. 显式创建Client实例，在实例化时指定host（解决host参数错误，同时确保寻址正确）
client = Client(host='http://127.0.0.1:11434')  # 这里是指定host的正确位置

# 3. 通过Client实例调用chat方法（不再直接调用顶层chat，避免参数错误）
response = client.chat(
    messages=[
        {
            'role': 'user',
            'content': 'Tell me about Canada. Please return the information strictly in JSON format that matches the Country model.',
        }
    ],
    model='deepseek-r1:7b',
    format=Country.model_json_schema(),
)

# 4. 验证JSON并打印结果
country = Country.model_validate_json(response.message.content)
print(country)