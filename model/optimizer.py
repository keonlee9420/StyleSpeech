import torch
import numpy as np


class ScheduledOptimMain:
    """ A simple wrapper class for learning rate scheduling """
    
    def __init__(self, model, train_config, model_config, current_step):
        self._optimizer = torch.optim.Adam(
            [param for name, param in model.named_parameters()
                        if not any([filtered_name in name for filtered_name in ['D_s', 'D_t']])],
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)
        # meta_learning_warmup = train_config["step"]["meta_learning_warmup"]
        self.current_step = current_step# if current_step <= meta_learning_warmup else current_step - meta_learning_warmup

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        state_dict['param_groups'] = self._optimizer.state_dict()['param_groups']
        self._optimizer.load_state_dict(state_dict)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class ScheduledOptimDisc:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.init_lr = train_config["optimizer"]["lr_disc"]
        self._init_learning_rate()

    def step_and_update_lr(self):
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _init_learning_rate(self):
        lr = self.init_lr
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
