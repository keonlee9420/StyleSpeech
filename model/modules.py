import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

from .blocks import (
    Mish,
    FCBlock,
    Conv1DBlock,
    SALNFFTBlock,
    MultiHeadAttention,
)
from text.symbols import symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class MelStyleEncoder(nn.Module):
    """ Mel-Style Encoder """

    def __init__(self, preprocess_config, model_config):
        super(MelStyleEncoder, self).__init__()
        n_position = model_config["max_seq_len"] + 1
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        d_melencoder = model_config["melencoder"]["encoder_hidden"]
        n_spectral_layer = model_config["melencoder"]["spectral_layer"]
        n_temporal_layer = model_config["melencoder"]["temporal_layer"]
        n_slf_attn_layer = model_config["melencoder"]["slf_attn_layer"]
        n_slf_attn_head = model_config["melencoder"]["slf_attn_head"]
        d_k = d_v = (
            model_config["melencoder"]["encoder_hidden"]
            // model_config["melencoder"]["slf_attn_head"]
        )
        kernel_size = model_config["melencoder"]["conv_kernel_size"]
        dropout = model_config["melencoder"]["encoder_dropout"]

        self.max_seq_len = model_config["max_seq_len"]

        self.fc_1 = FCBlock(n_mel_channels, d_melencoder)

        self.spectral_stack = nn.ModuleList(
            [
                FCBlock(
                    d_melencoder, d_melencoder, activation=Mish()
                )
                for _ in range(n_spectral_layer)
            ]
        )

        self.temporal_stack = nn.ModuleList(
            [
                nn.Sequential(
                    Conv1DBlock(
                        d_melencoder, 2 * d_melencoder, kernel_size, activation=Mish(), dropout=dropout
                    ),
                    nn.GLU(),
                )
                for _ in range(n_temporal_layer)
            ]
        )

        self.slf_attn_stack = nn.ModuleList(
            [
                MultiHeadAttention(
                    n_slf_attn_head, d_melencoder, d_k, d_v, dropout=dropout, layer_norm=True
                )
                for _ in range(n_slf_attn_layer)
            ]
        )

        self.fc_2 = FCBlock(d_melencoder, d_melencoder)

    def forward(self, mel, mask):

        max_len = mel.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        enc_output = self.fc_1(mel)

        # Spectral Processing
        for _, layer in enumerate(self.spectral_stack):
            enc_output = layer(enc_output)

        # Temporal Processing
        for _, layer in enumerate(self.temporal_stack):
            residual = enc_output
            enc_output = layer(enc_output)
            enc_output = residual + enc_output

        # Multi-head self-attention
        for _, layer in enumerate(self.slf_attn_stack):
            residual = enc_output
            enc_output, _ = layer(
                enc_output, enc_output, enc_output, mask=slf_attn_mask
            )
            enc_output = residual + enc_output

        # Final Layer
        enc_output = self.fc_2(enc_output) # [B, T, H]

        # Temporal Average Pooling
        enc_output = torch.mean(enc_output, dim=1, keepdim=True) # [B, 1, H]

        return enc_output


class PhonemePreNet(nn.Module):
    """ Phoneme Encoder PreNet """

    def __init__(self, config):
        super(PhonemePreNet, self).__init__()
        d_model = config["transformer"]["encoder_hidden"]
        kernel_size = config["prenet"]["conv_kernel_size"]
        dropout = config["prenet"]["dropout"]

        self.prenet_layer = nn.Sequential(
            Conv1DBlock(
                d_model, d_model, kernel_size, activation=Mish(), dropout=dropout
            ),
            Conv1DBlock(
                d_model, d_model, kernel_size, activation=Mish(), dropout=dropout
            ),
            FCBlock(d_model, d_model, dropout=dropout),
        )

    def forward(self, x, mask=None):
        residual = x
        x = self.prenet_layer(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = residual + x
        return x


class PhonemeEncoder(nn.Module):
    """ PhonemeText Encoder """

    def __init__(self, config):
        super(PhonemeEncoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_w = config["melencoder"]["encoder_hidden"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=0
        )
        self.phoneme_prenet = PhonemePreNet(config)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                SALNFFTBlock(
                    d_model, d_w, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, w, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- PreNet
        src_seq = self.phoneme_prenet(self.src_word_emb(src_seq), mask)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = src_seq + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = src_seq + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, w, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


class MelPreNet(nn.Module):
    """ Mel-spectrogram Decoder PreNet """

    def __init__(self, config):
        super(MelPreNet, self).__init__()
        d_model = config["transformer"]["encoder_hidden"]
        d_melencoder = config["melencoder"]["encoder_hidden"]
        dropout = config["prenet"]["dropout"]

        self.prenet_layer = nn.Sequential(
            FCBlock(d_model, d_melencoder, activation=Mish(), dropout=dropout),
            FCBlock(d_melencoder, d_model, activation=Mish(), dropout=dropout),
        )

    def forward(self, x, mask=None):
        x = self.prenet_layer(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        return x


class MelDecoder(nn.Module):
    """ MelDecoder """

    def __init__(self, config):
        super(MelDecoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_w = config["melencoder"]["encoder_hidden"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.mel_prenet = MelPreNet(config)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                SALNFFTBlock(
                    d_model, d_w, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, w, mask, return_attns=False):

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- PreNet
        enc_seq = self.mel_prenet(enc_seq, mask)

        # -- Forward
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)

            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, w, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        d_model = model_config["transformer"]["encoder_hidden"]
        kernel_size = model_config["variance_embedding"]["kernel_size"]
        self.pitch_embedding = Conv1DBlock(
            1, d_model, kernel_size
        )
        self.energy_embedding = Conv1DBlock(
            1, d_model, kernel_size
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(target.unsqueeze(-1))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(prediction.unsqueeze(-1))
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(target.unsqueeze(-1))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(prediction.unsqueeze(-1))
        return prediction, embedding

    def upsample(self, x, mel_mask, max_len, log_duration_prediction=None, duration_target=None, d_control=1.0):
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, None)
            mel_mask = get_mask_from_lengths(mel_len)
        return x, duration_rounded, mel_len, mel_mask

    def forward(
        self,
        x,
        src_mask,
        mel_mask,
        max_len,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        upsampled_text = None
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding

        x, duration_rounded, mel_len, mel_mask = self.upsample(
            x, mel_mask, max_len, log_duration_prediction=log_duration_prediction, duration_target=duration_target, d_control=d_control
        )

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class PhonemeDiscriminator(nn.Module):
    """ Phoneme Discriminator """

    def __init__(self, preprocess_config, model_config):
        super(PhonemeDiscriminator, self).__init__()
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        d_mel_linear = model_config["discriminator"]["mel_linear_size"]
        d_model = model_config["discriminator"]["phoneme_hidden"]
        d_layer = model_config["discriminator"]["phoneme_layer"]

        self.max_seq_len = model_config["max_seq_len"]
        self.mel_linear = nn.Sequential(
            FCBlock(n_mel_channels, d_mel_linear, activation=nn.LeakyReLU(), spectral_norm=True),
            FCBlock(d_mel_linear, d_mel_linear, activation=nn.LeakyReLU(), spectral_norm=True),
        )
        self.discriminator_stack = nn.ModuleList(
            [
                FCBlock(
                    d_model, d_model, activation=nn.LeakyReLU(), spectral_norm=True
                )
                for _ in range(d_layer)
            ]
        )
        self.final_linear = FCBlock(d_model, 1, spectral_norm=True)

    def forward(self, upsampler, text, mel, max_len, mask, duration_target):

        # Prepare Upsampled Text
        upsampled_text, _, _, _ = upsampler(
            text, mask, max_len, duration_target=duration_target
        )
        max_len = min(max_len, self.max_seq_len)
        upsampled_text = upsampled_text[:, :max_len, :]

        # Prepare Mel
        mel = self.mel_linear(mel)[:, :max_len, :]
        mel = mel.masked_fill(mask.unsqueeze(-1)[:, :max_len, :], 0)

        # Prepare Input
        x = torch.cat([upsampled_text, mel], dim=-1)

        # Phoneme Discriminator
        for _, layer in enumerate(self.discriminator_stack):
            x = layer(x)
        x = self.final_linear(x) # [B, T, 1]
        x = x.masked_fill(mask.unsqueeze(-1)[:, :max_len, :], 0)

        # Temporal Average Pooling
        x = torch.mean(x, dim=1, keepdim=True) # [B, 1, 1]
        x = x.squeeze()  # [B,]

        return x


class StyleDiscriminator(nn.Module):
    """ Style Discriminator """

    def __init__(self, preprocess_config, model_config):
        super(StyleDiscriminator, self).__init__()
        n_position = model_config["max_seq_len"] + 1
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        d_melencoder = model_config["melencoder"]["encoder_hidden"]
        n_spectral_layer = model_config["melencoder"]["spectral_layer"]
        n_temporal_layer = model_config["melencoder"]["temporal_layer"]
        n_slf_attn_layer = model_config["melencoder"]["slf_attn_layer"]
        n_slf_attn_head = model_config["melencoder"]["slf_attn_head"]
        d_k = d_v = (
            model_config["melencoder"]["encoder_hidden"]
            // model_config["melencoder"]["slf_attn_head"]
        )
        kernel_size = model_config["melencoder"]["conv_kernel_size"]

        self.max_seq_len = model_config["max_seq_len"]

        self.fc_1 = FCBlock(n_mel_channels, d_melencoder, spectral_norm=True)

        self.spectral_stack = nn.ModuleList(
            [
                FCBlock(
                    d_melencoder, d_melencoder, activation=nn.LeakyReLU(), spectral_norm=True
                )
                for _ in range(n_spectral_layer)
            ]
        )

        self.temporal_stack = nn.ModuleList(
            [
                Conv1DBlock(
                    d_melencoder, d_melencoder, kernel_size, activation=nn.LeakyReLU(), spectral_norm=True
                )
                for _ in range(n_temporal_layer)
            ]
        )

        self.slf_attn_stack = nn.ModuleList(
            [
                MultiHeadAttention(
                    n_slf_attn_head, d_melencoder, d_k, d_v, layer_norm=True, spectral_norm=True
                )
                for _ in range(n_slf_attn_layer)
            ]
        )

        self.fc_2 = FCBlock(d_melencoder, d_melencoder, spectral_norm=True)

        self.V = FCBlock(d_melencoder, d_melencoder)
        self.w_b_0 = FCBlock(1, 1, bias=True)

    def forward(self, style_prototype, speakers, mel, mask):

        max_len = mel.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        x = self.fc_1(mel)

        # Spectral Processing
        for _, layer in enumerate(self.spectral_stack):
            x = layer(x)

        # Temporal Processing
        for _, layer in enumerate(self.temporal_stack):
            residual = x
            x = layer(x)
            x = residual + x

        # Multi-head self-attention
        for _, layer in enumerate(self.slf_attn_stack):
            residual = x
            x, _ = layer(
                x, x, x, mask=slf_attn_mask
            )
            x = residual + x

        # Final Layer
        x = self.fc_2(x) # [B, T, H]

        # Temporal Average Pooling, h(x)
        x = torch.mean(x, dim=1, keepdim=True) # [B, 1, H]

        # Output Computation
        s_i = style_prototype(speakers) # [B, H]
        V = self.V(s_i).unsqueeze(2) # [B, H, 1]
        o = torch.matmul(x, V).squeeze(2) # [B, 1]
        o = self.w_b_0(o).squeeze() # [B,]

        return o