FROM zz-harbor.vastaitech.com/solution/litcosyvoice:v1.0

ADD ./cosyvoice/cli /workspace/CosyVoice/cosyvoice/cli
ADD ./runtime /workspace/CosyVoice/runtime
ADD ./libs /workspace/CosyVoice/libs
RUN pip install /workspace/CosyVoice/libs/*.whl