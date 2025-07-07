import requests
import base64

payload = {
    "spk_id": "a25c604e-f40d-4060-9d9d-3ae7b3864791",
    "tts_text": '唐杰明天来上海出差，我带你去上海滩浪一浪。'
}

resp = requests.post(
    'http://127.0.0.1:50000/inference_zero_shot',
    data=payload
)

data = resp.json()['data']
wavData = base64.b64decode(data)

with open('output.wav', 'wb') as f:
    f.write(wavData)