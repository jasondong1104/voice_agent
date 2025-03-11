from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api_config import update_config, get_default_config

# 创建FastAPI应用
app = FastAPI()
llm_instance = None  # 将在entrypoint中初始化

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/config")
async def update_model_config(config: dict):
    if llm_instance:
        llm_instance.update_config(config)
    return {"status": "success"}

@app.get("/default-config")
async def get_default_configuration():
    return get_default_config()

async def run_fastapi():
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()