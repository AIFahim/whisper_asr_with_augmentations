
from transformers import pipeline , WhisperForConditionalGeneration, WhisperProcessor , WhisperFeatureExtractor , WhisperTokenizer
import whisper
import torchaudio

## Current directory need to set
model = WhisperForConditionalGeneration.from_pretrained("./").to("cuda")
processor = WhisperProcessor.from_pretrained("./")
tokenizer = WhisperTokenizer.from_pretrained("./")


filename = "path/of/any/audio/file"
audio, sample_rate = torchaudio.load(filename)


model.eval()

inputs = processor(audio.squeeze(0), sampling_rate=16_000, return_tensors="pt")
input_features = inputs['input_features'].to("cuda")
generated_ids = model.generate(input_features, max_length=100)
generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generated_text)


""" Didn't work
# make log-Mel spectrogram and move to the same device as the model
options = whisper.DecodingOptions(fp16 = False)
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)

# decode the audio
result = whisper.decode(model, mel, options)
"""

