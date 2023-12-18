from transformers import pipeline , WhisperForConditionalGeneration, WhisperProcessor , WhisperFeatureExtractor , WhisperTokenizer


"""
    ***Causion code not yet completed, This code keep for reference if any chance need to use pipeline procedure***

"""
pipe = pipeline(model="AIFahim/whisper-medium-bn_our_dataset_positional_phoneme", use_auth_token =  "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", device="cuda") 
@app.post("/whisper-grapheme/")
async def create_file(file: bytes = File(), fmt: str = Form('flac')):
    x = AudioSegment.from_file(io.BytesIO(file)).set_channels(1).set_frame_rate(16000).export(format=fmt).read()
    text = pipe(x)["text"]
    return {"result": [text]}