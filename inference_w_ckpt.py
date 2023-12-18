
from transformers import pipeline , WhisperForConditionalGeneration, WhisperProcessor , WhisperFeatureExtractor , WhisperTokenizer
import torchaudio

## Current directory need to set
model = WhisperForConditionalGeneration.from_pretrained("/home/asif/stt_all/whisper/from_150_136_219_192/whisper_final/ckpt/best_checkpoint").to("cuda")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")


filename = "/home/asif/stt_all/whisper/from_150_136_219_192/whisper_final/0a0d3225-a989-4ba2-b2c9-7f09c8a3cb6b.wav"
audio, sample_rate = torchaudio.load(filename)

model.eval()

inputs = processor(audio.squeeze(0), sampling_rate=16_000, return_tensors="pt")
input_features = inputs['input_features'].to("cuda")
generated_ids = model.generate(input_features, max_length=100)
generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)



