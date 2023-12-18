## 1. Setting Up Environment Variables & Devices
import os
from statistics import mode
import torch
from train_val_df_gen import Train_Val_df
from datasets import Dataset, DatasetDict, Audio
import json
import librosa
import soundfile as sf
import math
from audioaugmentations import AudioAugmentations
import random
import numpy as np
import glob
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperTokenizerFast, WhisperProcessor
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import transformers as tf
from transformers import WhisperForConditionalGeneration

# ML Flow SettingUP
os.environ['MLFLOW_TRACKING_USERNAME'] = "mlflow"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "1234567"
abs_path = os.path.abspath('.')
base_dir = os.path.dirname(abs_path)
device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"\n\n Device to be used: {device} \n\n")



## 2. Defining Model , Feature Extractor, Tokenizer and Processor 
model_name = "openai/whisper-small"
language = "Bengali"
task = "transcribe"  # transcribe or translate
apply_spec_augment = False
dropout = 0  # 0.1  # any value > 0.1 hurts performance. So, use values between 0.0 and 0.1
freeze_feature_encoder = False
gradient_checkpointing = False

model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Override generation arguments
model.config.apply_spec_augment = apply_spec_augment
model.config.dropout = dropout
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
if gradient_checkpointing:
    model.config.use_cache = False
if freeze_feature_encoder:
    model.freeze_feature_encoder()

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
# tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-small", language=language, task=task)
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language=language, task=task)


## 3. Load Datasets
print("\n\n Loading Datasets...this might take a while..\n\n")

with open('/home/asif/stt_all/Datasets/merge_dict_small_chunked.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

"""
NUM_PROC = 1
NUM_CHUNKS = 6
NUM_EPOCHS = 25
TOTAL_FILES = 0

with open('/home/asif/merged_dict.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)
    print(len(json_data.values()))
    TOTAL_FILES = len(json_data.values())
    json_data = dict(itertools.islice(json_data.items(), 5 * len(json_data.keys()) // NUM_CHUNKS, 6 * len(json_data.keys()) // NUM_CHUNKS))
    # json_data = dict(list(json_data.values())[0*len(json_data.keys())//NUM_CHUNKS:1*len(json_data.keys())//NUM_CHUNKS])

MAX_STEPS = math.ceil((TOTAL_FILES // NUM_CHUNKS) / (per_device_train_batch_size * gradient_accumulation_steps)) * NUM_EPOCHS  # 3000
"""
 
# Call the generate_df_from_json() method on the Train_Val_df class directly
train_df, val_df = Train_Val_df.generate_df_from_json(json_data)
print("Total Datapoint Lenghts", len(train_df), len(val_df))

dataset_our = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(val_df)
})
dataset_our = dataset_our.cast_column("audio", Audio(sampling_rate=16000, mono=True))





## 5. Preprocessing Data & Applied Augmentations Prior
print("\n\n Preprocessing Datasets...this might take a while..\n\n")

# Original Prepare dataset without any augmentation
def prepare_dataset(batch, batch_idx,factor, aug):
    print(batch)
    try:
        
        # load and (possibly) resample audio data to 16kHz
        audio = batch["audio"]
        audio["array"] = librosa.util.normalize(audio["array"])
        # compute log-Mel input features from input audio array 
        inputs = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=apply_spec_augment,
        )
        batch["input_features"] = inputs.input_features[0]
        # compute input length
        batch["input_length"] = len(batch["audio"])

        # if spec augmentation applied, get attention_mask to guide the mask along time axis
        if apply_spec_augment:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # optional pre-processing steps
        transcription = batch["sentence"]
        # encode target text to label ids
        batch["labels"] = processor.tokenizer(transcription).input_ids
        # compute labels length **with** special tokens! -> total label length
        batch["labels_length"] = len(batch["labels"])
        return batch

    except Exception as e:
        print(f"Skipping this batch due to an exception: {e}")
        return None


def prepare_dataset_batch(dataset):
    bgNoiseFileList = glob.glob("/home/asif/augmentations_experiments/environmental-sound-classification-50/audio/audio/16000/**/*.wav",recursive=True)

    APPLIED_AUGMENTATIONS = [
        "speedAug", 
        "pitchShift", 
        "farFieldEffect", 
        "colorNoise", 
        "bgNoiseAug",
        "down_upsampling", 
        "time_n_freq_masking", 
    ]

    batch_size = int(math.ceil(len(dataset) / len(APPLIED_AUGMENTATIONS)))



    num_batches = len(APPLIED_AUGMENTATIONS) # math.ceil(len(dataset) / batch_size)
    results = []

    # Add original files to the results
    for idx, element in enumerate(tqdm(dataset, desc="Processing original audio")):
        try:
            batch_result = prepare_dataset(element, idx, 0, 'original')
            results.append(batch_result)
        except Exception as e:
            print(f"Skipping processing original audio for sample {idx} due to an exception: {e}")

    print(" results ",results)

    # assert False

    for idx,batch_idx in enumerate(tqdm(range(num_batches), desc="Processing augmented audio")):
        AudioAugmentations_pera = [0.0]*len(APPLIED_AUGMENTATIONS)
        AudioAugmentations_pera[idx] += 1.0  
      
        Choices = {
                    str(APPLIED_AUGMENTATIONS.index("speedAug")): [0.75, 0.8, 0.9, 1.1, 1.25, 1.5], # Speed
                    str(APPLIED_AUGMENTATIONS.index("pitchShift")): [3, -3], # Pitch
                    str(APPLIED_AUGMENTATIONS.index("farFieldEffect")): [1.0, 3.0, 5.0], # Far Field
                    str(APPLIED_AUGMENTATIONS.index("bgNoiseAug")): [0.8, 0.9, 0.95], # BG Focus Blur 
                    str(APPLIED_AUGMENTATIONS.index("down_upsampling")): [2000, 4000, 8000]
                }
        
        if str(idx) in Choices:
            for option in Choices[str(idx)]:

                audio_augmentor = AudioAugmentations(
                    multipleAug=False,
                    speedAugProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("speedAug")],
                    pitchShiftProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("pitchShift")],
                    farFieldEffectProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("farFieldEffect")],
                    bgNoiseAugProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("bgNoiseAug")],
                    colorNoiseProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("colorNoise")],
                    time_n_freq_maskingProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("time_n_freq_masking")],
                    down_upsamplingProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("down_upsampling")],
                    
                    bgNoise_focusMinMax= [option],
                    speedFactors = [option],
                    pitchShiftRange = [option],
                    farFieldDistances = [option],
                    bgNoiseFileList = bgNoiseFileList,
                    down_upsamplingMinMax = [option],

                    sampleRate=16000,
                )

                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, len(dataset))
                batch = dataset[start:end]

                # Create a list of dictionaries with 'audio' and 'sentence' keys
                formatted_batch = [{'audio': audio_element, 'sentence': sentence}
                                for audio_element, sentence in zip(batch['audio'], batch['sentence'])]
                
                # print(formatted_batch)

                num_augmented = int(len(formatted_batch) * 1.0) # Full
                random_indices = random.sample(range(len(formatted_batch)), num_augmented)

                for idx1 in random_indices:
                    try:
                        formatted_batch[idx1]["audio"]["array"] = audio_augmentor.getAudio(formatted_batch[idx1]["audio"]["path"], returnTensor=False)
                    except Exception as e:
                        print(f"Skipping augmentation for sample {idx1} due to an exception: {e}")
                        print(f"Audio shape: {formatted_batch[idx1]['audio']['array'].shape}")

                batch_results = [prepare_dataset(element, batch_idx, option, APPLIED_AUGMENTATIONS[idx]) for element in formatted_batch]
                results.extend(batch_results)
            
        else:
            audio_augmentor = AudioAugmentations(
                multipleAug=False,
                speedAugProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("speedAug")],
                pitchShiftProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("pitchShift")],
                farFieldEffectProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("farFieldEffect")],
                bgNoiseAugProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("bgNoiseAug")],
                colorNoiseProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("colorNoise")],
                time_n_freq_maskingProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("time_n_freq_masking")],
                down_upsamplingProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("down_upsampling")],
                bgNoiseFileList = bgNoiseFileList,

                sampleRate=16000,
            )
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(dataset))
            batch = dataset[start:end]

            # Create a list of dictionaries with 'audio' and 'sentence' keys
            formatted_batch = [{'audio': audio_element, 'sentence': sentence}
                            for audio_element, sentence in zip(batch['audio'], batch['sentence'])]

            num_augmented = int(len(formatted_batch) * 1.0) # Full
            random_indices = random.sample(range(len(formatted_batch)), num_augmented)

            for idx1 in random_indices:
                try:
                    formatted_batch[idx1]["audio"]["array"] = audio_augmentor.getAudio(formatted_batch[idx1]["audio"]["path"], returnTensor=False)
                except Exception as e:
                    print(f"Skipping augmentation for sample {idx1} due to an exception: {e}")
                    print(f"Audio shape: {formatted_batch[idx1]['audio']['array'].shape}")

            batch_results = [prepare_dataset(element, batch_idx, 0, APPLIED_AUGMENTATIONS[idx]) for element in formatted_batch]

            results.extend(batch_results)

    return results


# Prepare dataset
train_data_prepared = prepare_dataset_batch(dataset_our["train"])
test_data_prepared = prepare_dataset_batch(dataset_our["test"])



def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists

# Convert the list of dictionaries to a dictionary of lists
train_data_dict = list_of_dicts_to_dict_of_lists(train_data_prepared)
test_data_dict = list_of_dicts_to_dict_of_lists(test_data_prepared)



with open('train_data_prepared.pkl', 'wb') as f:
    pickle.dump(train_data_dict, f)

with open('test_data_prepared.pkl', 'wb') as f:
    pickle.dump(test_data_dict, f)

# Create the datasets from the dictionaries
dataset_our["train"] = Dataset.from_dict(train_data_dict)
dataset_our["test"] = Dataset.from_dict(test_data_dict)




## 6. Filter too Short or too Long Audio Files
MAX_DURATION_IN_SECONDS = 30.0
max_input_length = MAX_DURATION_IN_SECONDS * 16000

def filter_inputs(input_length):
    """Filter inputs with zero input length or longer than 30s"""
    return 0 < input_length < max_input_length

dataset_our["train"] = dataset_our["train"].filter(
    filter_inputs,
    input_columns=["input_length"],
)
dataset_our["test"] = dataset_our["test"].filter(
    filter_inputs,
    input_columns=["input_length"],
)

max_label_length = 448  # (Check by doing model.config.max_length. Model not yet initialized, so manually written)


def filter_labels(labels_length):
    """Filter label sequences longer than max length (448)"""
    return labels_length < max_label_length


dataset_our["train"] = dataset_our["train"].filter(
    filter_labels,
    input_columns=["labels_length"],
)
dataset_our["test"] = dataset_our["test"].filter(
    filter_labels,
    input_columns=["labels_length"],
)



## 7. Define Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        for feature in features:
            print(feature.keys())
            # print(feature["audio"]["path"])

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, forward_attention_mask=apply_spec_augment)


## 8. Define Evaluation Metrics
wer_metric = evaluate.load("wer", cache_dir=os.path.join(base_dir, "metrics_cache"))
cer_metric = evaluate.load("cer", cache_dir=os.path.join(base_dir, "metrics_cache"))

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}



## 9. Define the Training Configuration
    
# Setting Up Training Args
output_dir = "/home/asif/stt_all/whisper/from_150_136_219_192/whisper_final/ckpt"
overwrite_output_dir = True
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
# gradient_accumulation_steps = 0 #8
MAX_STEPS = 3000
dataloader_num_workers = 4
evaluation_strategy = "steps"
eval_steps = 2
save_strategy = "steps"
save_steps = 2
save_total_limit = 5
learning_rate = 1e-5
lr_scheduler_type = "cosine_with_restarts"  # "cosine" # "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"(default), "polynomial", "inverse_sqrt"
warmup_steps = 888  # (1 epoch)
logging_steps = 1  # 25
weight_decay = 0
load_best_model_at_end = True
metric_for_best_model = "cer"
greater_is_better = False
bf16 = False
tf32 = True
generation_max_length = 448
predict_with_generate = True
push_to_hub = False # True
early_stopping_patience = 10




training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    # overwrite_output_dir=overwrite_output_dir,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    # gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    dataloader_num_workers=dataloader_num_workers,
    evaluation_strategy=evaluation_strategy,
    eval_steps=eval_steps,
    save_strategy=save_strategy,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_steps=warmup_steps,
    logging_steps=logging_steps,
    weight_decay=weight_decay,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=greater_is_better,
    bf16=bf16,
    tf32=tf32,
    generation_max_length=generation_max_length,
    # report_to=report_to,
    predict_with_generate=predict_with_generate,
    push_to_hub=push_to_hub,
    # hub_token="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",
)



trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_our["test"],
    eval_dataset=dataset_our["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    # callbacks=[tf.EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
)


print(len(dataset_our["train"]))
print(len(dataset_our["test"]))

# assert False
## 10. Training
print("\n\n Training STARTED..\n\n")

"""
 IF MLFlow NEED 

# mlflow define
# mlflow.set_tracking_uri("http://119.148.4.20:6060/")
# experiment_id = mlflow.get_experiment_by_name("asif whisper tiny with specaug our dataset")

# if experiment_id is None:
#     experiment_id = mlflow.create_experiment("asif whisper tiny with specaug our dataset")
# else:
#     experiment_id = experiment_id.experiment_id

# with mlflow.start_run(experiment_id=experiment_id):
#     train_result = trainer.train()

"""

train_result = trainer.train()

## resume from the latest checkpoint
# train_result = trainer.train(resume_from_checkpoint=True)

## resume training from the specific checkpoint in the directory passed
# train_result = trainer.train(resume_from_checkpoint="checkpoint-4000")


print("\n\n Training COMPLETED...\n\n")
print("\n\n DONEEEEEE \n\n")
