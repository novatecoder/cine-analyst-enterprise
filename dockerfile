# 1. CUDA 12.1 기반 이미지 (vLLM과 일치)
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 2. 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    POETRY_VIRTUALENVS_CREATE=false

# 3. Python 3.11 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 핵심: 모든 python 명령어가 3.11을 바라보도록 고정 (Symbolic Link)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Pip 자체를 python 3.11용으로 업그레이드
RUN python -m pip install --upgrade pip

WORKDIR /app

# 4. Poetry 설치 및 설정
RUN python -m pip install poetry

# 5. 의존성 설치 (Poetry)
COPY pyproject.toml poetry.lock* /app/
# --no-root 옵션으로 소스 코드 복사 전에 라이브러리만 먼저 설치 (빌드 캐시 활용)
RUN poetry install --no-interaction --no-ansi --no-root

# 6. Unsloth 및 최신 AI 라이브러리 설치 (가장 확실하게 python -m pip 사용)
RUN python -m pip install --no-cache-dir \
    "unsloth @ git+https://github.com/unslothai/unsloth.git" \
    "xformers==0.0.28.post2" \
    "trl" "peft" "accelerate" "bitsandbytes" \
    "uvicorn==0.38.0" \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 7. 소스 코드 복사
COPY src/ /app/src/

# 8. 포트 개방
EXPOSE 8000

# 9. 서비스 실행 (모듈 경로를 PYTHONPATH 기반으로 정확히 지정)
# src/cine_analyst/app/main.py 가 있다면 아래와 같이 호출합니다.
CMD ["python", "-m", "cine_analyst.app.main"]