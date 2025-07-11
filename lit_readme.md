## 服务启动
``` sh
# python runtime/python/fastapi/lit_server.py --spks_dir /opt/spks
uvicorn runtime.python.fastapi.lit_server_sft:app --port 50000 --workers 2
```