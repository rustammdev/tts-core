# Adapted from inference.py + models/prediction_wrapper.py +
# models/frame_mn/Frame_MN_wrapper.py (frame_mn path only); module attribute
# names match the released checkpoints' state_dict exactly.
import torch
import torch.nn as nn

from uztts_events._vendor.psed.model import get_model
from uztts_events._vendor.psed.preprocess import AugmentMelSTFT


class FrameMNWrapper(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        self.mel = AugmentMelSTFT(
            n_mels=128,
            sr=16_000,
            win_length=400,
            hopsize=160,
            n_fft=512,
            freqm=0,
            timem=0,
            htk=False,
            fmin=0.0,
            fmax=None,
            norm=1,
            fmin_aug_range=10,
            fmax_aug_range=2000,
            fast_norm=True,
            preamp=True,
            padding="center",
            periodic_window=False,
        )
        self.frame_mn = get_model(width_mult=width_mult)

    def mel_forward(self, x):
        return self.mel(x)

    def forward(self, x):
        return self.frame_mn(x)


class FramePredictor(nn.Module):
    def __init__(self, width_mult=1.0, n_classes=447, seq_len=250):
        super().__init__()
        self.model = FrameMNWrapper(width_mult)
        self.seq_model = nn.Identity()
        self.seq_len = seq_len
        embed_dim = self.model.frame_mn.lastconv_output_channels
        self.strong_head = nn.Linear(embed_dim, n_classes)
        self.weak_head = nn.Linear(embed_dim, n_classes)

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=True)

    def forward(self, waveform):
        mel = self.model.mel_forward(waveform)
        x = self.model(mel)
        if x.size(-2) > self.seq_len:
            x = torch.nn.functional.adaptive_avg_pool1d(
                x.transpose(1, 2), self.seq_len
            ).transpose(1, 2)
        elif x.size(-2) < self.seq_len:
            x = torch.nn.functional.interpolate(
                x.transpose(1, 2), size=self.seq_len, mode="linear"
            ).transpose(1, 2)
        x = self.seq_model(x)
        return self.strong_head(x).transpose(1, 2)
