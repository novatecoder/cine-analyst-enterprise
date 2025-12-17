import uvicorn
from fastapi import FastAPI
from cine_analyst.app.api import router
from cine_analyst.common.config import settings

# 이 'app' 변수가 정의되어 있어야 테스트 코드에서 불러올 수 있습니다.
app = FastAPI(title="Cine Analyst Enterprise API")

app.include_router(router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "ok"}

def start():
    uvicorn.run("cine_analyst.app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()