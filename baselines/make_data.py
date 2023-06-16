import numpy as np
import librosa
import torch
import torchaudio
from scipy.io import wavfile

def load_audio(path_name, use_torch=True, resample=True, resample_rate=22050):
    # returns in shape (ch, num_sample), as float32 (on Linux at least)
    # by default torchaudio is wav_arr, sample_rate
    # by default wavfile is sample_rate, wav_arr
    if use_torch:
        loaded = torchaudio.load(path_name)
        wave_data_loaded = loaded[0].numpy()
        sr_loaded = loaded[1]
    else:
        loaded = wavfile.read(path_name)
        wave_data_loaded = np.clip(loaded[1], -1.0, 1.0).T
        sr_loaded = loaded[0]
    if resample:
        if wave_data_loaded.shape[1]==0:
            print("len 0")
            assert False
        if wave_data_loaded.shape[1]<int(sr_loaded*0.1):
            padded_wav = librosa.util.fix_length(wave_data_loaded, int(sr_loaded*0.1))
            resampled_wave = librosa.resample(padded_wav, orig_sr=sr_loaded, target_sr=resample_rate)
        else:
            resampled_wave = librosa.resample(wave_data_loaded, orig_sr=sr_loaded, target_sr=resample_rate)
    else:
        resampled_wave = wave_data_loaded
    return np.clip(resampled_wave, -1.0, 1.0)

def if_compute(arg):
    unwrapped_angle = np.unwrap(arg).astype(np.single)
    return np.concatenate([unwrapped_angle[:, :, 0:1], np.diff(unwrapped_angle, n=1)], axis=-1)


class get_spec():
    def __init__(self, use_torch=False, power_mod=2, fft_size=512):
        self.n_fft = fft_size
        self.hop = self.n_fft // 4
        if use_torch:
            self.use_torch = True
            self.spec_transform = Spectrogram(power=None, n_fft=self.n_fft, hop_length=self.hop)
        else:
            self.power = power_mod
            self.use_torch = False
            self.spec_transform = None

    def transform(self, wav_data_prepad):
        wav_data = librosa.util.fix_length(wav_data_prepad, wav_data_prepad.shape[-1] + self.n_fft // 2)
        if wav_data.shape[1] < 4410:
            wav_data = librosa.util.fix_length(wav_data, 4410)
        if self.use_torch:
            transformed_data = self.spec_transform(torch.from_numpy(wav_data)).numpy()
        else:

            transformed_data = np.array([librosa.stft(wav_data[0], n_fft=self.n_fft, hop_length=self.hop),
                                         librosa.stft(wav_data[1], n_fft=self.n_fft, hop_length=self.hop)])[:, :-1]
        real_component = np.abs(transformed_data)
        img_component = np.angle(transformed_data)
        gen_if = if_compute(img_component) / np.pi
        return np.log(real_component + 1e-3), gen_if
