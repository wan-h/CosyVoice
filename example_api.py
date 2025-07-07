import sys
sys.path.append('/home/ubuntu/willw/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=True)
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
cosyvoice.save_spkinfo(
    '希望你以后能够做的比我还好呦。', 
    prompt_speech_16k, 
    'my_zero_shot_spk'
)

for i, j in enumerate(cosyvoice.inference_zero_shot('你真是一个大机灵鬼。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)