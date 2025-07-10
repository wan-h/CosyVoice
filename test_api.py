import requests
import base64

def test_zero_shot():
    payload = {
        "spk_id": "a25c604e-f40d-4060-9d9d-3ae7b3864791",
        "tts_text": """你好, 我是一个测试文本, 请你帮我合成语音。"""
    }

    resp = requests.post(
        'http://10.24.8.90:50000/inference_zero_shot',
        data=payload
    )

    print(resp.json())
    data = resp.json()['data']
    wavData = base64.b64decode(data)

    with open('output.wav', 'wb') as f:
        f.write(wavData)

def test_sft():
    payload = {
        "spk_id": "中文女",
        "tts_text": '唐杰明天来上海出差，我带你去上海滩浪一浪。'
    }

    resp = requests.post(
        'http://127.0.0.1:50000/inference_sft',
        data=payload
    )

    print(resp.json())
    data = resp.json()['data']
    wavData = base64.b64decode(data)

    with open('output.wav', 'wb') as f:
        f.write(wavData)

if __name__ == '__main__':
    test_zero_shot()
    # test_sft()