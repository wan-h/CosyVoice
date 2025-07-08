# -*- coding: utf-8 -*-
'''
@File    :   lit_server.py
@Author  :   一力辉
'''


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
from enum import Enum
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

class SFT_SPKS(str, Enum):
    CN_FEMALE = '中文女'
    CN_MALE = '中文男'
    JP_MALE = '日语男'
    HK_FEMALE = '粤语女'
    EN_FEMALE = '英文女'
    EN_MALE = '英文男'
    KR_FEMALE = '韩语女'

@app.post("/add_spk_info", response_model=BaseResponse, summary="add spk info")
async def add_spk_info(prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    try:
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        spk_id = str(uuid4())
        cosyvoice_2.save_spkinfo(prompt_text, prompt_speech_16k, spk_id)
        return JSONResponse(status_code=200, content={'code': 0, 'message': 'success', 'data': spk_id})
    except Exception as e:
        return JSONResponse(status_code=200, content={'code': -1, 'message': str(e), 'data': ''})

@app.post("/inference_zero_shot", response_model=BaseResponse, summary="inference zero shot")
async def inference_zero_shot(spk_id: str = Form(), tts_text: str = Form(), speed: float = Form(1.0)):
    try:
        model_output = cosyvoice_2.inference_zero_shot(
            tts_text=tts_text, 
            prompt_text='', 
            prompt_speech_16k='', 
            zero_shot_spk_id=spk_id,
            stream=False,
            speed=speed,
            text_frontend=True
        )
        output = BytesIO()
        for data in model_output:
            torchaudio.save(output, data['tts_speech'], cosyvoice_2.sample_rate, format='wav')
        return JSONResponse(status_code=200, content={'code': 0, 'message': 'success', 'data': base64.b64encode(output.getvalue()).decode('utf-8')})
    except Exception as e:
        return JSONResponse(status_code=200, content={'code': -1, 'message': str(e), 'data': ''})

@app.post("/inference_sft", response_model=BaseResponse, summary="inference sft")
async def inference_sft(tts_text: str = Form(), spk_id: SFT_SPKS = Form(), speed: float = Form(1.0)):
    try:
        model_output = cosyvoice_1.inference_sft(
            tts_text=tts_text, 
            spk_id=spk_id,
            stream=False,
            speed=speed,
            text_frontend=True
        )
        output = BytesIO()
        for data in model_output:
            torchaudio.save(output, data['tts_speech'], cosyvoice_1.sample_rate, format='wav')
        return JSONResponse(status_code=200, content={'code': 0, 'message': 'success', 'data': base64.b64encode(output.getvalue()).decode('utf-8')})
    except Exception as e:
        return JSONResponse(status_code=200, content={'code': -1, 'message': str(e), 'data': ''})

def initialize_cosyvoice(model_dir_1, model_dir_2):
    global cosyvoice_1, cosyvoice_2
    cosyvoice_1 = CosyVoice(model_dir_1)
    cosyvoice_2 = CosyVoice2(model_dir_2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--model_dir_1',
                        type=str,
                        default='pretrained_models/CosyVoice-300M-SFT',
                        help='local path or modelscope repo id')
    parser.add_argument('--model_dir_2',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--spks_dir',
                        type=str,
                        default='',
                        help='local path')
    parser.add_argument('--workers',
                        type=int,
                        default=1,
                        help='universal worker num')
    args = parser.parse_args()

    initialize_cosyvoice(args.model_dir_1, args.model_dir_2)

    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=args.workers)
