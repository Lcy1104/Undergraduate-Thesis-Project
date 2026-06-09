# fastapi_backend.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uuid
import json
import re
from ollama import Client

app = FastAPI()
tasks = {}

class AskRequest(BaseModel):
    prompt: str
    kb_search_text: str = ""
    weather_text: str = ""

def run_ollama(task_id: str, prompt: str):
    try:
        client = Client(host='http://127.0.0.1:11434')
        response = client.chat(
            messages=[{'role': 'user', 'content': prompt}],
            model='deepseek-r1:7b',
            format={
                "type": "object",
                "properties": {
                    "kb_search_results": {"type": "string"},
                    "thinking_process": {"type": "string"},
                    "final_conclusion": {"type": "string"}
                },
                "required": ["kb_search_results", "thinking_process", "final_conclusion"]
            },
            options={"temperature": 0.1, "max_tokens": 2048}
        )
        raw_json = response.message.content.strip()
        raw_json = re.sub(r'^```json|```$', '', raw_json).strip()
        answer = json.loads(raw_json)
        tasks[task_id] = {"status": "completed", "answer": answer}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "error": str(e)}

# 在文件末尾，app 定义之后添加
import logging

# 配置日志，便于查看接收和发送的报文
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.delete("/api/result/{task_id}")
async def delete_result(task_id: str):
    """前端确认结束后删除任务，作为 finish_ack 报文"""
    if task_id in tasks:
        del tasks[task_id]
        logging.info(f"DELETE /api/result/{task_id} -> 任务已删除")
        return {"status": "deleted"}
    logging.warning(f"DELETE /api/result/{task_id} -> 任务不存在")
    return {"status": "not_found"}

# 修改原有的 ask 和 get_result 接口，增加日志
@app.post("/api/ask")
async def ask(request: AskRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "running"}
    background_tasks.add_task(run_ollama, task_id, request.prompt)
    logging.info(f"POST /api/ask -> 创建任务 {task_id}")
    return {"hit": False, "task_id": task_id}

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    task = tasks.get(task_id, {"status": "not_found"})
    logging.info(f"GET /api/result/{task_id} -> 返回状态 {task.get('status')}")
    return task