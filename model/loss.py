import torch
import torch.nn as nn


class MetaStyleSpeechLossMain(nn.Module):
    """ Meta-StyleSpeech Loss for naive StyleSpeech and Step 1 """

    def __init__(self, preprocess_config, model_config, train_config):
        super(MetaStyleSpeechLossMain, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.alpha = train_config["optimizer"]["alpha"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            _,
            _,
            _,
            _,
            _,
        ) = inputs[6:]
        (
            D_s,
            D_t,
            mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        alpha = 1
        D_s_loss = D_t_loss = torch.tensor([0.], device=mel_predictions.device, requires_grad=False)
        if D_s is not None and D_t is not None:
            D_s_loss = self.mse_loss(D_s, torch.ones_like(D_s, requires_grad=False))
            D_t_loss = self.mse_loss(D_t, torch.ones_like(D_t, requires_grad=False))
            alpha = self.alpha

        recon_loss = alpha * (mel_loss + duration_loss + pitch_loss + energy_loss)
        total_loss = (
            recon_loss + D_s_loss + D_t_loss
        )

        return (
            total_loss,
            mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            D_s_loss,
            D_t_loss,
        )


class MetaStyleSpeechLossDisc(nn.Module):
    """ Meta-StyleSpeech Loss for Step 2 """

    def __init__(self, preprocess_config, model_config):
        super(MetaStyleSpeechLossDisc, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, speakers, predictions):
        (
            D_t_s,
            D_t_q,
            D_s_s,
            D_s_q,
            style_logit,
        ) = predictions
        speakers.requires_grad = False

        D_t_loss = self.mse_loss(D_t_s, torch.ones_like(D_t_s, requires_grad=False))\
                    + self.mse_loss(D_t_q, torch.zeros_like(D_t_q, requires_grad=False))
        D_s_loss = self.mse_loss(D_s_s, torch.ones_like(D_s_s, requires_grad=False))\
                    + self.mse_loss(D_s_q, torch.zeros_like(D_s_q, requires_grad=False))
        cls_loss = self.cross_entropy_loss(style_logit, speakers)

        total_loss = (
            D_t_loss + D_s_loss + cls_loss
        )

        return (
            total_loss,
            D_t_loss,
            D_s_loss,
            cls_loss,
        )
