# cine-analyst-enterprise
영화 정보 분석 서비스

poetry install --with train


pip install --no-build-isolation "unsloth @ git+https://github.com/unslothai/unsloth.git"
pip install --no-build-isolation unsloth_zoo
pip install torchvision --index-url https://download.pytorch.org/whl/cu121
pip install trl unsloth unsloth_zoo torchvision