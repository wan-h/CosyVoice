# coding: utf-8
# Author: wanhui0729@gmail.com

import json
from locust import task, HttpUser, between

class MyTestUser(HttpUser):
    # wait_time: locust 定义请求间隔
    wait_time = between(0.1, 0.2)  # 模拟用户在执行每个任务之间等待的最小时间，单位为秒；

    @task
    def test_tts(self):
        payload = {
            "spk_id": "a25c604e-f40d-4060-9d9d-3ae7b3864791",
            "tts_text": """你好, 我是一个测试文本, 请你帮我合成语音。""",
        }
        resp = self.client.post('/inference_zero_shot', data=payload)
        # 确保正常返回
        resp = resp.json()
        assert resp['code'] == 0

'''
locust -f test.py --host http://0.0.0.0:50000 --web-host 0.0.0.0
网页打开浏览器: http://0.0.0.0:8089
'''