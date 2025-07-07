## 镜像启动
``` sh
docker run \
    --privileged \
    --name lit_cosyvoice \
    -p 50000:50000 \
    -v `pwd`:/workspace/Cosyvoice cosyvoice:latest \
    --entrypoint python /workspace/Cosyvoice/runtime/fastapi/server_voice_clone.py
```