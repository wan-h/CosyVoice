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
from pydub import AudioSegment
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

def merge_wavs(inputs):
    """
    使用pydub合并多个WAV文件的二进制数据并返回合并后的bytes。

    :param inputs: 包含WAV文件二进制数据的列表
    :return: 合并后的WAV文件的二进制数据
    """
    # 创建一个空的AudioSegment对象
    combined = AudioSegment.silent(duration=0)

    for wav_data in inputs:
        # 将二进制数据加载为AudioSegment对象
        audio = AudioSegment.from_file(BytesIO(wav_data), format="wav")
        # 将当前音频片段添加到合并后的音频中
        combined += audio

    # 将合并后的音频保存到BytesIO对象中
    output_io = BytesIO()
    combined.export(output_io, format="wav")
    # 获取合并后的二进制数据
    merged_data = output_io.getvalue()
    return merged_data

@app.post("/add_spk_info", response_model=BaseResponse, summary="add spk info")
def add_spk_info(prompt_text: str = Form(), prompt_wav: UploadFile = File()):
    try:
        prompt_speech_16k = load_wav(prompt_wav.file, 16000)
        spk_id = str(uuid4())
        cosyvoice_2.save_spkinfo(prompt_text, prompt_speech_16k, spk_id)
        return JSONResponse(status_code=200, content={'code': 0, 'message': 'success', 'data': spk_id})
    except Exception as e:
        return JSONResponse(status_code=200, content={'code': -1, 'message': str(e), 'data': ''})

@app.post("/inference_zero_shot", response_model=BaseResponse, summary="inference zero shot")
def inference_zero_shot(
    spk_id: str = Form(), 
    tts_text: str = Form(), 
    speed: float = Form(1.0),
    corss_lingual: bool = Form(False)
):
    try:
        if corss_lingual:
            model_output = cosyvoice_2.inference_cross_lingual(
                tts_text=tts_text, 
                prompt_speech_16k='',
                zero_shot_spk_id=spk_id,
                stream=False,
                speed=speed,
                text_frontend=True
            )
        else:
            model_output = cosyvoice_2.inference_zero_shot(
                tts_text=tts_text, 
                prompt_text='', 
                prompt_speech_16k='', 
                zero_shot_spk_id=spk_id,
                stream=False,
                speed=speed,
                text_frontend=True
            )
        outputs = []
        for data in model_output:
            split = BytesIO()
            torchaudio.save(split, data['tts_speech'], cosyvoice_2.sample_rate, format='wav')
            outputs.append(split.getvalue())
        return JSONResponse(status_code=200, content={'code': 0, 'message': 'success', 'data': base64.b64encode(merge_wavs(outputs)).decode('utf-8')})
    except Exception as e:
        return JSONResponse(status_code=200, content={'code': -1, 'message': str(e), 'data': ''})

@app.post("/inference_sft", response_model=BaseResponse, summary="inference sft")
def inference_sft(tts_text: str = Form(), spk_id: SFT_SPKS = Form(), speed: float = Form(1.0)):
    try:
        model_output = cosyvoice_1.inference_sft(
            tts_text=tts_text, 
            spk_id=spk_id,
            stream=False,
            speed=speed,
            text_frontend=True
        )
        outputs = []
        for data in model_output:
            split = BytesIO()
            torchaudio.save(split, data['tts_speech'], cosyvoice_1.sample_rate, format='wav')
            outputs.append(split.getvalue())
        return JSONResponse(status_code=200, content={'code': 0, 'message': 'success', 'data': base64.b64encode(merge_wavs(outputs)).decode('utf-8')})
    except Exception as e:
        return JSONResponse(status_code=200, content={'code': -1, 'message': str(e), 'data': ''})

def initialize_cosyvoice_1(model_dir_1):
    global cosyvoice_1
    cosyvoice_1 = CosyVoice(model_dir_1)

def initialize_cosyvoice_2(model_dir_2, spks_dir):
    global cosyvoice_2
    cosyvoice_2 = CosyVoice2(model_dir_2, spks_dir=spks_dir)

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

    initialize_cosyvoice_1(args.model_dir_1)
    initialize_cosyvoice_2(args.model_dir_2, args.spks_dir)

    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=args.workers)
