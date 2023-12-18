from typing import Dict, List, Tuple, Any, Union, Optional
import os
import re
import json
import random
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import pyrubberband as pyrb
from scipy.signal import correlate
import pyplnoise



class AudioAugmentations:
    """
    AudioConverter offers methods to load, transcode and augment
    audio data in various ways.
    """

    # Configurations for parameters used in torchaudio's resampling kernel.
    resampleFilterParams = {
        "fast": {  # Fast and less accurate but still MSE = ~2e-5 compared to librosa.
            "lowpass_filter_width": 16,
            "rolloff": 0.85,
            "resampling_method": "kaiser_window",
            "beta": 8.555504641634386,
        },
        "best": { # Twice as slow, and a little bit more accburate.
            "lowpass_filter_width": 64,
            "rolloff": 0.9475937167399596,
            "resampling_method": "kaiser_window",
            "beta": 14.769656459379492,       
        },
    }

    def __init__(
        self,
        multipleAug: bool = False,
        sampleRate: int = 16000,
        disableAug: bool = False,
        speedAugProb: float = 0.5,
        # volAugProb: float = 0.5,
        pitchShiftProb: float = 0.25,
        # reverbAugProb: float = 0.25,
        farFieldEffectProb: float = 0.25,  # Add this new parameter for farField effect
        bgNoiseAugProb: float = 0.25,  # Add this new parameter for background noise
        colorNoiseProb: float = 0.25,
        time_n_freq_maskingProb: float = 0.25,
        down_upsamplingProb: float = 0.25,

        bgNoise_focusMinMax: Tuple[float, float] = None, # Focus Blur Parameter 
        speedFactors: Tuple[float, float] = None,
        NoiseGainMinMax: Tuple[float, float] = None,
        down_upsamplingMinMax: Tuple[float, float] = None,
        # volScaleMinMax: Tuple[float, float] = None,
        # reverbRoomScaleMinMax: Tuple[float, float] = None,
        # reverbHFDampingMinMax: Tuple[float, float] = None,
        # reverbSustainMinMax: Tuple[float, float] = None,
        pitchShiftRange: Tuple[int, int] = None,
        farFieldDistances: Tuple[float, float] = None,
        
        bgNoiseFileList: List[str] = None,  # Add this new parameter for background noise files
    ):
        """
        Initializes AudioConverter.

        Parameters
        ----------
        sampleRate: int
            Sampling rate to convert audio to, if required.

        disableAug: bool, optional
            If True, overrides all other augmentation configs and
            disables all augmentatoins.

        speedAugProb: float, optional
            Probability that speed augmentation will be applied.
            If <= 0, speed augmentation is disabled.

        volAugProb: float, optional
            Probability that volume augmentation will be applied.
            If <= 0, volume augmentation is disabled.

        reverbAugProb: float, optional
            Probability that reverberation augmentation will be applied.
            If <= 0, reverberation augmentation is disabled.

        noiseAugProb: float, optional
            Probability that noise augmentation will be applied.
            If <= 0, noise augmentation is disabled.

        speedFactors: List[float], optional
            List of factors by which to speed up (>1) or slow down (<1)
            audio by. One factor is chosen randomly if provided. Otherwise,
            default speed factors are [0.9, 1.0, 1.0].
            
        volScaleMinMax: Tuple[float, float], optional
            [Min, Max] range for volume scale factors. One factor is
            chose randomly with uniform probability from this range.
            Default range is [0.125, 2.0].

        reverbRoomScaleMinMax: Tuple[float, float], optional
            [Min, Max] range for room size percentage. Values must be
            between 0 and 100. Larger room size results in more reverb.
            Default range is [25, 75].

        reverbHFDampingMinMax: Tuple[float, float], optional
            [Min, Max] range for high frequency damping percentage. Values must
            be between 0 and 100. More damping results in muffled sound.
            Default range is [25, 75].
        
        reverbSustainMinMax: Tuple[float, float], optional
            [Min, Max] range for reverberation sustain percentage. Values must
            be between 0 and 100. More sustain results in longer lasting echoes.
            Default range is [25, 75].
            
        noiseSNRMinMax: Tuple[float, float], optional
            [Min, Max] range for signal-to-noise ratio when adding noise. One
            factor is chose randomly with uniform probability from this range.
            Lower SNR results in louder noise. Default range is [10.0, 30.0].

        noiseFileList: List[str], optional
            List of paths to audio files to use as noise samples. If None is provided,
            noise augmentation will be disabled. Otherwise, the audio files will be assumed
            to be sources of noise, and be mixed in with speech audio on-the-fly.
        """
        self.sampleRate = sampleRate
        self.multipleAug = multipleAug
        
        enableAug = not disableAug

        self.speedAugProb = speedAugProb if enableAug else -1
        # self.volAugProb = volAugProb if enableAug else -1
        # self.reverbAugProb = reverbAugProb if enableAug else -1
        self.farFieldEffectProb = farFieldEffectProb if enableAug else -1
        self.bgNoiseAugProb = bgNoiseAugProb if enableAug else -1
        self.pitchShiftProb = pitchShiftProb if enableAug else -1
        self.colorNoiseProb = colorNoiseProb if enableAug else -1
        self.time_n_freq_maskingProb = time_n_freq_maskingProb if enableAug else -1
        self.down_upsamplingProb = down_upsamplingProb if enableAug else -1
        
        # Set the range of speed factors for audio speed perturbation
        self.speedFactorsRange = speedFactors
        if speedFactors is None:
            self.speedFactorsRange = [0.75, 0.8, 0.9, 1.1, 1.25, 1.5]

        # Set the range of volume scales
        # self.volScaleRange = volScaleMinMax
        # if volScaleMinMax is None:
        #     self.volScaleRange = [0.125, 2.0]

        # Set the range of room size percentages for reverb (higher = more reverb)
        # self.reverbRoomScaleRange = reverbRoomScaleMinMax
        # if reverbRoomScaleMinMax is None:
        #     self.reverbRoomScaleRange = [25, 75]

        # Set the range of high-frequency damping percentages (higher = more damping)
        # self.reverbHFDampingRange = reverbHFDampingMinMax
        # if reverbHFDampingMinMax is None:
        #     self.reverbHFDampingRange = [25, 75]

        # Set the range of reverb sustain percentages (higher = lasts longer)
        # self.reverbSustainRange = reverbSustainMinMax
        # if reverbSustainMinMax is None:
        #     self.reverbSustainRange = [25, 75]

        # Set the range of far-field distances
        self.farFieldDistances = farFieldDistances
        if self.farFieldDistances is None:
            self.farFieldDistances = [1.0, 3.0, 5.0]

        # Set the range of pitch shift values
        self.pitchShiftRange = pitchShiftRange
        if self.pitchShiftRange is None:
            self.pitchShiftRange = [3, -3]

        # Set the range of noise gain values
        self.NoiseGainRange = NoiseGainMinMax
        if NoiseGainMinMax is None:
            self.NoiseGainRange = [0.1]

        # Set the background noise files and augmentation probability
        self.bgNoiseFiles = bgNoiseFileList
        if bgNoiseFileList is None or len(self.bgNoiseFiles) == 0:
            self.bgNoiseAugProb = -1

        # Set the range of background noise focus values
        self.bgNoise_focusRange = bgNoise_focusMinMax
        if bgNoise_focusMinMax is None:
            self.bgNoise_focusRange = [0.8, 0.9, 0.95]


        # Set the range of background noise focus values
        self.down_upsamplingRange = down_upsamplingMinMax
        if bgNoise_focusMinMax is None:
            self.down_upsamplingRange = [2000, 4000, 8000]


        # self.validateConfig()
        
    def validateConfig(self):
        """
        Checks configured options and raises an error if they
        are not consistent with what is expected.
        """
        if len(self.volScaleRange) != 2:
            raise ValueError("volume scale range must be provided as [min, max]")
        if len(self.reverbRoomScaleRange) != 2:
            raise ValueError("reverb room scale range must be provided as [min, max]")
        if len(self.reverbHFDampingRange) != 2:
            raise ValueError("reverb high frequency dampling range must be provided as [min, max]")
        if len(self.reverbSustainRange) != 2:
            raise ValueError("reverb sustain range must be provided as [min, max]")
        # if len(self.noiseSNRRange) != 2:
        #     raise ValueError("noise SNR range must be provided as [min, max]")
            
        for v in self.reverbRoomScaleRange:
            if v > 100 or v < 0:
                raise ValueError("reverb room scale must be between 0 and 100")
        for v in self.reverbHFDampingRange:
            if v > 100 or v < 0:
                raise ValueError("reverb high frequency dampling must be between 0 and 100")
        for v in self.reverbSustainRange:
            if v > 100 or v < 0:
                raise ValueError("reverb sustain range must be between 0 and 100")

    @classmethod
    def loadAudio(
        cls, audioPath: str, sampleRate: int = None, returnTensor: bool = True, resampleType: str = "fast",
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Uses torchaudio to load and resample (if necessary) audio files and returns
        audio samples as either a numpy.float32 array or a torch.Tensor.
        
        Parameters
        ----------
        audioPath: str
            Path to audio file file (wav / mp3 / flac).
        
        sampleRate: int, optional
            Sampling rate to convert audio to. If None,
            audio is not resampled.
        
        returnTensor: bool, optional
            If True, the audio samples are returned as a torch.Tensor.
            Otherwise, the samples are returned as a numpy.float32 array.
            
        resampleType: str, optional
            Either "fast" or "best" - sets the quality of resampling.
            "best" is twice as slow as "fast" but more accurate. "fast"
            is still comparable to librosa's resampled output though,
            in terms of MSE.

        Returns
        -------
        Union[torch.Tensor, np.ndarray]
            Audio waveform scaled between +/- 1.0 as either a numpy.float32 array,
            or torch.Tensor, with shape (channels, numSamples)
        """
        x, sr = torchaudio.load(audioPath)
        if sampleRate is not None or sr != sampleRate:
            x = F.resample(x, sr, sampleRate, **cls.resampleFilterParams[resampleType])
        
        if returnTensor:
            return x
        
        return x.numpy()

    def getAudio(self, audioPath: str, returnTensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Loads audio from specified path and applies augmentations randomly
        on-the-fly. Audio samples scaled between -1.0 and +1.0 are returned
        as a numpy.float32 array or torch.Tensor with shape (numSamples,).

        Parameters
        ----------
        audioPath: str
            Path to audio file file (wav / mp3 / flac).
        
        returnTensor: bool, optional
            If True, the audio samples are returned as a torch.Tensor.
            Otherwise, the samples are returned as a numpy.float32 array.
        
        Returns
        ------- 
        Union[torch.Tensor, np.ndarray]
            Audio waveform scaled between +/- 1.0 as either a numpy.float32 array,
            or torch.Tensor, with shape (channels, numSamples)
        """
        wav = self.loadAudio(
            audioPath, sampleRate=self.sampleRate, returnTensor=True, resampleType="fast",
        )

    
        # print("---------------------", self.speedAugProb, self.pitchShiftProb, self.farFieldEffectProb, self.colorNoiseProb, self.bgNoiseAugProb, self.time_n_freq_maskingProb, self.down_upsamplingProb)
        if self.multipleAug == True:         
            
            if random.uniform(0, 1) <= self.speedAugProb:
                speed = random.choice(self.speedFactorsRange)
                wav = self.perturbSpeed(wav, speed)

            if random.uniform(0, 1) <= self.reverbAugProb:
                roomSize = str(random.uniform(*self.reverbRoomScaleRange))
                hfDamping = str(random.uniform(*self.reverbHFDampingRange))
                sustain = str(random.uniform(*self.reverbSustainRange))
                effects = [["reverb", roomSize, hfDamping, sustain]]
                wav = self.applySoxEffects(wav, effects)

            
            if random.uniform(0, 1) <= self.volAugProb:
                volScale = random.uniform(*self.volScaleRange)
                wav = self.scaleVolume(wav, volScale)

            # Add these lines before the line `if returnTensor:`
            if random.uniform(0, 1) <= self.farFieldEffectProb:
                distance = random.choice(self.farFieldDistances)
                wav = self.apply_far_field_effect(wav, distance)
            
            if random.uniform(0, 1) <= self.colorNoiseProb:
                gain = random.choice(self.NoiseGainRange)
                wav, sr = librosa.load(audioPath, sr=None)
                wav = self.add_noises(wav, gain)

            if random.uniform(0, 1) <= self.bgNoiseAugProb:
                bgNoise_focus_r = random.choice(self.bgNoise_focusRange)
                # print("bgNoise_focus_r ", bgNoise_focus_r)
                bgNoiseFile = random.choice(self.bgNoiseFiles)
                wav = self.addBackgroundNoise(wav, bgNoiseFile, bgNoise_focus_r)

            if random.uniform(0, 1) <= self.pitchShiftProb: 
                pitchShift = random.choice(self.pitchShiftRange)
                # print("pitchShift ",pitchShift)
                wav = self.pitch_shift(wav, pitchShift)

            if random.uniform(0, 1) <= self.time_n_freq_maskingProb: 
                wav = self.time_n_freq_masking(wav, pitchShift)

            if random.uniform(0, 1) <= self.down_upsamplingProb: 
                down_upsamplingfactor = random.choice(self.down_upsamplingRange)
                wav = self.downupsampling(wav, down_upsamplingfactor)

        
        elif self.multipleAug == False:
            valid_augmentations = [
                (aug_name, aug_prob) for aug_name, aug_prob in zip(
                    ["speed", "far_field", "color_noise", "bg_noise", "pitch_shift", "time_freq_mask", "down_upsampling"],
                    [
                        self.speedAugProb,
                        # self.volAugProb,
                        # self.reverbAugProb,
                        self.farFieldEffectProb,
                        self.colorNoiseProb,
                        self.bgNoiseAugProb,
                        self.pitchShiftProb,
                        self.time_n_freq_maskingProb,
                        self.down_upsamplingProb
                    ]
                ) if aug_prob > 0
            ]
            
            if valid_augmentations:
                aug_choice = random.choices(*zip(*valid_augmentations), k=1)[0]
                # print("aug_choice ", aug_choice)
            else:

                raise ValueError("No augmentations selected. At least one augmentation must have a non-zero probability.")


            if aug_choice == "speed":
                speed = random.choice(self.speedFactorsRange)
                # print("speed ", speed)
                wav = self.perturbSpeed(wav, speed)

            # elif aug_choice == "vol":
            #     volScale = random.uniform(*self.volScaleRange)
            #     wav = self.scaleVolume(wav, volScale)

            # elif aug_choice == "reverb":
            #     roomSize = str(random.uniform(*self.reverbRoomScaleRange))
            #     hfDamping = str(random.uniform(*self.reverbHFDampingRange))
            #     sustain = str(random.uniform(*self.reverbSustainRange))
            #     effects = [["reverb", roomSize, hfDamping, sustain]]
            #     wav = self.applySoxEffects(wav, effects)

            elif aug_choice == "far_field":
                distance = random.choice(self.farFieldDistances)
                wav = self.apply_far_field_effect(wav, distance)

            elif aug_choice == "color_noise":
                gain = random.choice(self.NoiseGainRange)
                wav, sr = librosa.load(audioPath, sr=None)
                wav = self.add_noises(wav, gain)

            elif aug_choice == "bg_noise":
                bgNoise_focus_r = random.choice(self.bgNoise_focusRange)
                # print("bgNoise_focus_r ", bgNoise_focus_r)
                bgNoiseFile = random.choice(self.bgNoiseFiles)
                # print("bgNoiseFile ", bgNoiseFile)
                wav = self.addBackgroundNoise(wav, bgNoiseFile, bgNoise_focus_r)

            elif aug_choice == "pitch_shift":
                pitchShift = random.choice(self.pitchShiftRange)
                # print("pitchShift ",pitchShift)
                wav = self.pitch_shift(wav, pitchShift)

            elif aug_choice == "time_freq_mask":
                wav = self.time_n_freq_masking(wav)

            
            elif aug_choice == "down_upsampling":
                down_upsamplingfactor = random.choice(self.down_upsamplingRange)
                wav = self.downupsampling(wav, down_upsamplingfactor)
        


        

        ## Return
        if returnTensor:
            if isinstance(wav, np.ndarray):
                final_wav = torch.from_numpy(wav).to(torch.float64)  # Cast to float64
            else:
                final_wav = wav.to(torch.float64)  # Cast to float64
            
            if final_wav.ndim == 2 and final_wav.size(0) == 1:
                final_wav.squeeze(0)
            
            return final_wav

        else:
            if isinstance(wav, torch.Tensor):
                final_wav = wav.to(torch.float64).numpy()  # Cast to float64 before converting to numpy
            else:
                final_wav = wav.astype(np.float64)  # Cast to float64 for numpy array
            
            if final_wav.ndim == 2 and final_wav.shape[0] == 1:
                final_wav = np.squeeze(final_wav, axis=(0))
            
            return final_wav



        # if returnTensor:
        #     if isinstance(wav, np.ndarray):
        #         final_wav = torch.from_numpy(wav)
        #     else:
        #         final_wav =  wav
            
        #     if final_wav.ndim == 2:
        #         final_wav.squeeze(0)
            
        #     return final_wav

        #     # return torch.from_numpy(wav) if isinstance(wav, np.ndarray) else wav
        # else:
        #     if isinstance(wav, torch.Tensor):
        #         final_wav = wav.numpy()
        #     else:
        #         final_wav =  wav
            
        #     print(final_wav.ndim)
        #     if final_wav.ndim == 2:
        #         final_wav = np.squeeze(final_wav, axis=(0))
            
        #     return final_wav
            
            
            # return wav.numpy() if isinstance(wav, torch.Tensor) else wav   # Make sure this line is correct

    
    
    def downupsampling(self, audio_signal: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Perform downsample and upsample operations on an audio signal.

        This function downsamples the input audio signal to a randomly chosen lower sample rate 
        and then upsamples it back to the original sample rate. The resampling is done using 
        the torchaudio.transforms.Resample class.

        Parameters
        ----------
        audio_signal: torch.Tensor
            Input audio signal as a PyTorch tensor.

        Returns
        -------
        torch.Tensor
            Resampled audio signal as a PyTorch tensor.
        """

        # Choose a random resample rate from the given options
        resample_rate = factor # random.choice([4000, 8000, 16000])

        # Create resampler objects for down and up sampling
        resampler_down = T.Resample(self.sampleRate, resample_rate, dtype=audio_signal.dtype)
        resampler_up = T.Resample(resample_rate, 16000, dtype=audio_signal.dtype)

        # Downsample the audio signal
        resampled_waveform_down = resampler_down(audio_signal)

        # Upsample the downsampled audio signal
        resampled_waveform_up = resampler_up(resampled_waveform_down)

        return resampled_waveform_up




    def add_noises(self, audio_signal: np.ndarray, NoiseGain: float) -> torch.Tensor:
        """
        Add synthetic noise to an audio signal.

        This function adds synthesized noise from a randomly chosen noise generator to an audio signal. 
        The noise is scaled by the given NoiseGain parameter and is normalized before being added to the audio signal.

        Parameters
        ----------
        audio_signal: np.ndarray
            Input audio signal as a NumPy array.

        NoiseGain: float
            Gain factor for scaling the synthesized noise.

        Returns
        -------
        torch.Tensor
            Audio signal with added noise as a PyTorch tensor.
        """
        # Create a white noise generator
        fs = 10.0
        noise_generators = [
            pyplnoise.AlphaNoise(fs, 1e-3, fs/2., alpha=1.5, seed=42),
            pyplnoise.PinkNoise(fs, 1e-3, fs/2., seed=42),
            pyplnoise.RedNoise(fs, 1e-3, seed=42),
            pyplnoise.WhiteNoise(fs, seed=42)
        ]

        # Randomly choose a noise generator from the list
        noisegen = random.choice(noise_generators) # pyplnoise.AlphaNoise(10.0, 1e-3, 10.0/2., alpha=1.5, seed=42)

        # Make sure the noise is the same length as the audio
        noise_length = len(audio_signal)
        noise = noisegen.get_series(noise_length)

        # Normalize the noise
        noise /= np.max(np.abs(noise))

        # Add the noise to the audio
        y_noisy = audio_signal + NoiseGain * noise

        # Convert the NumPy array to a PyTorch tensor and change the data type to float32
        if isinstance(y_noisy, np.ndarray):
            y_noisy_tensor = torch.from_numpy(y_noisy).float()
        elif isinstance(y_noisy, torch.Tensor):
            y_noisy_tensor = y_noisy.float()
        else:
            raise TypeError("y_noisy must be either a NumPy array or a PyTorch tensor")

        return y_noisy_tensor



    
    def align_audio(self, audio: np.ndarray, noise: np.ndarray, sr=16000) -> np.ndarray:
        """
        Align the background noise to the original audio by maximizing cross-correlation.

        Parameters
        ----------
        audio: torch.Tensor
            Main audio signal.

        noise: torch.Tensor
            Background noise signal.

        sr: int, optional
            Sampling rate of the audio and noise signals.
        Returns
        -------
        np.ndarray
            Aligned background noise signal.
        """
        cross_corr = correlate(audio, noise)
        # Find the time lag that maximizes the cross-correlation
        max_corr_idx = np.argmax(cross_corr)
        # Calculate the time lag in samples
        time_lag_samples = max_corr_idx - len(audio) + 1

        # Repeat the noise signal to reach the same length as the original audio
        aligned_noise = np.tile(noise, len(audio) // len(noise) + 1)[:len(audio)]

        return aligned_noise


    def addBackgroundNoise(self, wav: torch.Tensor, bgNoiseFile: str, focus_ratio=0.8, sr=16000) -> torch.Tensor:
        """
        Mix the audio with aligned background noise.

        Parameters
        ----------
        wav: torch.Tensor
            Main audio signal.
        
        bgNoiseFile: str
            Path to the background noise file.
        
        focus_ratio: float, optional
            Ratio to control the balance between the original audio and the noise.
        
        sr: int, optional
            Sampling rate of the audio and noise signals.

        Returns
        -------
        torch.Tensor
            Mixed audio signal.
        """
        noise_tensor, _ = librosa.load(bgNoiseFile, sr=sr)
        audio = wav.squeeze().numpy()

        # Align the background noise to the original audio
        aligned_noise = self.align_audio(audio, noise_tensor, sr)

            # Make sure the aligned noise and audio have the same length
        if len(aligned_noise) < len(audio):
            aligned_noise = np.pad(aligned_noise, (0, len(audio) - len(aligned_noise)))
        else:
            aligned_noise = aligned_noise[:len(audio)]
        # Mix the audio with the aligned background noise
        mixed_audio = focus_ratio * audio + (1 - focus_ratio) * aligned_noise
        return torch.from_numpy(mixed_audio).float().unsqueeze(0)

    
    def time_n_freq_masking(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Apply time and frequency masking on an audio waveform.

        This function generates a spectrogram from the input audio waveform, applies time and
        frequency masking to the spectrogram, and then reconstructs the audio waveform using
        the Griffin-Lim algorithm.

        Parameters
        ----------
        wav: torch.Tensor
            Input audio waveform as a PyTorch tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed audio waveform after time and frequency masking as a PyTorch tensor.
        """

        # Create spectrogram, time and frequency masking transforms
        spectrogram = torchaudio.transforms.Spectrogram()
        masking = torchaudio.transforms.TimeMasking(time_mask_param=5)
        f_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)

        # Set spectrogram parameters
        n_fft = 2048
        win_length = None
        hop_length = 512

        # Define transform
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )

        # Calculate the spectrogram
        spec = spectrogram(wav)

        # Apply time masking
        masked = masking(spec)

        # Apply frequency masking
        for i in range(5):
            masked = masking(masked)
        for i in range(5):
            masked = f_masking(masked)

        # Reconstruct the audio waveform using the Griffin-Lim algorithm
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        reconstructed_waveform = griffin_lim(masked)

        return reconstructed_waveform



    
    def perturbSpeed(self, wav: Union[np.ndarray, torch.Tensor], factor: float) -> torch.Tensor:
        """
        Perturbs the speed of the provided audio signal by the given factor.
        
        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
            Audio samples scaled between -1.0 and +1.0, with shape
            (channels, numSamples).

        Returns
        -------
        torch.Tensor
            Audio samples with perturbed speed. Will have more or less
            samples than input depending on whether slowed down or
            sped up.
        """
        # Convert input to numpy array if it's a torch.Tensor
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()

        # Ensure the input wav is 1D (single channel)
        if len(wav.shape) > 1 and wav.shape[0] == 1:
            wav = wav.squeeze()

        # Get the sample rate
        sr = self.sampleRate

        # Time stretch the input wav using pyrubberband
        y_stretch = pyrb.time_stretch(wav, sr, factor)

        # Convert the numpy array to a torch.Tensor and return it
        return torch.from_numpy(y_stretch).float().unsqueeze(0)


    def pitch_shift(self, wav: Union[np.ndarray, torch.Tensor], shift: int, sr: int = 16000) -> torch.Tensor:
        """
        Shifts the pitch of the provided audio signal by the given shift amount.

        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
            Audio samples scaled between -1.0 and +1.0, with shape (channels, numSamples).

        shift: int
            Pitch shift amount in semitones. Positive values shift the pitch up, negative values shift the pitch down.

        sr: int, optional
            Sampling rate of the audio signal.

        Returns
        -------
        torch.Tensor
            Audio samples with shifted pitch.
        """
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()

        # Ensure the input wav is 1D (single channel)
        if len(wav.shape) > 1 and wav.shape[0] == 1:
            wav = wav.squeeze()

        # Apply pitch shift using pyrubberband
        y_shift = pyrb.pitch_shift(wav, sr, shift)

        # Convert the numpy array to a torch.Tensor and return it
        return torch.from_numpy(y_shift).float().unsqueeze(0)

    
    def scaleVolume(self, wav: Union[np.ndarray, torch.Tensor], scale: float) -> torch.Tensor:
        """
        Scales the amplitude (with clipping) of the provided audio signal
        by the given scale factor.
        
        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
             Audio samples scaled between -1.0 and +1.0, with shape
             (channels, numSamples).

        Returns
        -------
        torch.Tensor
            Audio samples with perturbed volume.
        """
        if scale == 1.0:
            return wav

        return torch.clamp(wav * scale, -1.0, 1.0)

    def apply_far_field_effect(self, audio_signal: Union[np.ndarray, torch.Tensor], distance: float) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply far-field effect to an audio signal based on distance.

        This function simulates the far-field effect on an audio signal by applying a time delay
        and an attenuation factor based on the distance between the source and the listener.

        Parameters
        ----------
        audio_signal: Union[np.ndarray, torch.Tensor]
            Input audio signal as a NumPy array or a PyTorch tensor.

        distance: float
            Distance between the source and the listener in meters.

        Returns
        -------
        Union[np.ndarray, torch.Tensor]
            Audio signal with far-field effect applied, in the same format (NumPy array or PyTorch tensor) as the input.
        """

        # Convert PyTorch tensor to NumPy array if necessary
        if isinstance(audio_signal, torch.Tensor):
            audio_signal = audio_signal.numpy()

        # Set speed of sound and calculate attenuation factor and time delay
        speed_of_sound = 343  # Speed of sound in air (m/s)
        attenuation_factor = 1 / (4 * np.pi * distance**2)
        time_delay = distance / speed_of_sound

        # Apply time delay and attenuation factor to the audio signal
        far_field_audio_signal = np.roll(audio_signal, int(time_delay * self.sampleRate)) * attenuation_factor

        # Convert the audio signal back to a PyTorch tensor if the input was a tensor
        if torch.is_tensor(audio_signal):
            return torch.from_numpy(far_field_audio_signal)

        return far_field_audio_signal


   
    def applySoxEffects(self, wav: Union[np.ndarray, torch.Tensor], effects: List[List[str]]) -> torch.Tensor:
        """
        Applies different audio manipulation effects to provided audio, like
        speed and volume perturbation, reverberation etc. For a full list of
        supported effects, check torchaudio.sox_effects.

        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
             Audio samples scaled between -1.0 and +1.0, with shape
             (channels, numSamples).
        
        effects: List[List[str]]
            List of sox effects and associated arguments, example:
            '[ ["speed", "1.2"], ["vol", "0.5"] ]'

        Returns
        -------
        torch.Tensor
            Audio samples with effects applied. May not be the same
            number of samples as input sample array, depending on types
            of effects applied (e.g. speed perturbation may reduce or
            increase the number of samples).
        """
        if effects is None or len(effects) == 0:
            return wav

        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            wav, sample_rate=self.sampleRate, effects=effects,
        )

        # print("Output wav shape:", wav.size() if isinstance(wav, torch.Tensor) else wav.shape)
        # Convert stereo output to mono directly using the tensor
        if wav.ndim == 2 and wav.size(0) == 2:
            wav = wav.mean(dim=0, keepdim=True)

        return wav
    



    def addReverb(
        self, wav: Union[np.ndarray, torch.Tensor], roomSize: float, hfDamping: float, sustain: float,
    ) -> torch.Tensor:
        """
        Adds reverberation to the provided audio signal using given parameters.
        
        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
             Audio samples scaled between -1.0 and +1.0, with shape
             (channels, numSamples).
        
        roomSize: float
            Room size as a percentage between 0 and 100,
            higher = more reverb

        hfDamping: float
            High Frequency damping as a percentage between 0 and 100,
            higher = more damping.

        sustain: float
            How long reverb is sustained as a percentage between 0 and 100,
            higher = lasts longer.

        Returns
        -------
        torch.Tensor
            Audio samples with reverberated audio.
        """
        effects = [["reverb", f"{roomSize}", f"{hfDamping}", f"{sustain}"]]
        return self.applySoxEffects(wav, effects)
    


'''
### Archied Codes ###

{
    # For perturbSpeed
    def old_perturbSpeed(self, wav: Union[np.ndarray, torch.Tensor], factor: float) -> torch.Tensor:
        """
        Perturbs the speed of the provided audio signal by the given factor.
        
        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
                Audio samples scaled between -1.0 and +1.0, with shape
                (channels, numSamples).

        Returns
        -------
        torch.Tensor
            Audio samples with perturbed speed. Will have more or less
            samples than input depending on whether slowed down or
            sped up.
        """
        effects = [
            ["speed", f"{factor}"],
            ["rate", f"{self.sampleRate}"],
        ]
        return self.applySoxEffects(wav, effects)
}


{
    # For Noise File Adding Portion which was Based on SNR in random place: 

    inside init constructor:
    noiseAugProb: float = 0.25,
    noiseFileList: List[str] = None,
    noiseSNRMinMax: Tuple[float, float] = None,


    assigning init function:
    self.noiseAugProb = noiseAugProb if enableAug else -1

    # [Min, Max] Signal to noise ratio range for adding noise to audio.
    # Lower SNR = noise is more prominent, i.e. speech is more noisy.
    self.noiseSNRRange = noiseSNRMinMax
    if noiseSNRMinMax is None:
        self.noiseSNRRange = [10.0, 30.0]
        
    if random.uniform(0, 1) <= self.noiseAugProb:
        noiseFile = random.choice(self.noiseFiles)
        noiseSNR = random.uniform(*self.noiseSNRRange)
        wav = self.addNoiseFromFile(wav, noiseFile, noiseSNR)

    # Audio files to use as source of noise.
    self.noiseFiles = noiseFileList
    if self.noiseFiles is None or len(self.noiseFiles) == 0:
        self.noiseAugProb = -1

    # Functions 
    def addNoiseFromFile(
        self, wav: Union[np.ndarray, torch.Tensor], noiseFile: str, snr: float,
    ) -> torch.Tensor:
        """
        Adds noise signal from provided noise audio file at the 
        specified SNR to the speech signal.
        
        Parameters
        ----------
        wav: Union[np.ndarray, torch.Tensor]
                Audio samples scaled between -1.0 and +1.0, with shape
                (channels, numSamples).

        snr: float
            Signal-to-Noise ratio at which to mix in the noise signal.
        
        Returns
        -------
        torch.Tensor
            Audio samples with noise added at specified SNR.
        """
        # Loading noise signal.
        noiseSig = self.loadAudio(
            noiseFile, sampleRate=self.sampleRate, returnTensor=True, resampleType="fast",
        )

        # Computing noise power.
        noisePower = torch.mean(torch.pow(noiseSig, 2))
        
        # Computing signal power.
        signalPower = torch.mean(torch.pow(wav, 2))

        # Noise Coefficient for target SNR; amplitude coeff is sqrt of power coeff.
        noiseScale = torch.sqrt((signalPower / noisePower) / (10 ** (snr / 20.0)))
        
        # Add noise at random location in speech signal.
        nWav, nNoise = wav.shape[-1], noiseSig.shape[-1]

        if nWav < nNoise:
            a = random.randint(0, nNoise-nWav)
            b = a + nWav
            return wav + (noiseSig[..., a:b] * noiseScale)
        
        a = random.randint(0, nWav-nNoise)
        b = a + nNoise          
        wav[..., a:b] += (noiseSig * noiseScale)

        return wav
}

'''