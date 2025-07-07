# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import base64
import argparse
import logging
from uuid import uuid4
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from pydantic import BaseModel
from io import BytesIO
import torchaudio

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class BaseResponse(BaseModel):
    code: int
    message: str
    data: str

def generate_data(model_output):
    for i in model_output:
        tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield tts_audio

@app.post("/add_spk_info", response_model=BaseResponse, summary="add spk info")
async def add_spk_info(prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    try:
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        spk_id = str(uuid4())
        cosyvoice.save_spkinfo(prompt_text, prompt_speech_16k, spk_id)
        return JSONResponse(status_code=200, content={'code': 0, 'message': 'success', 'data': spk_id})
    except Exception as e:
        return JSONResponse(status_code=200, content={'code': -1, 'message': str(e), 'data': ''})

@app.post("/inference_zero_shot", response_model=BaseResponse, summary="inference zero shot")
async def inference_zero_shot(spk_id: str = Form(), tts_text: str = Form()):
    model_output = cosyvoice.inference_zero_shot(tts_text, '', '', spk_id)
    output = BytesIO()
    for data in model_output:
        torchaudio.save(output, data['tts_speech'], cosyvoice.sample_rate, format='wav')
    return JSONResponse(status_code=200, content={'code': 0, 'message': 'success', 'data': base64.b64encode(output.getvalue()).decode('utf-8')})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')
    uvicorn.run(app, host="0.0.0.0", port=args.port)
