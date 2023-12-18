# whisper_asr_with_augmentations
### File details as follow:
- audioaugmentations.py - This script contains List of Augmentations:
    - speed Aug
    - pitch Shift
    - far FieldEffect
    - background NoiseAug
    - color Noise
    - time and freq masking (SpecAug)
    - down then upsampling (Old age microphone like effects)
    - speech Enhence 
- inference_w_ckpt.py - This scirpt is for infer with the checkpoint. 
- train_val_df_gen.py - This script for load the dataset from the json or directory.
- train_whisper_small_augmentation.py - This script is the main training script. 
