from typing import Any, Mapping
import math


class HPScheduler:
    def __init__(self, hparams_dict, last_epoch=-1):
        # Attach dictionary with hyper-parameters
        if not isinstance(hparams_dict, dict):
            raise TypeError(f"{type(hparams_dict).__name__} is not an dictionary")
        self.hparams_dict = hparams_dict

        self.base_hparams = {hparam: value for hparam, value in hparams_dict.items()}
        self.last_epoch = last_epoch
        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and performs a step"""
        self._step_count = 0
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_hparams(self):
        """Return last computed hyperparameter by current scheduler."""
        return self._last_hparams

    def get_hparams(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        self._step_count += 1
        if epoch is None:
            self.last_epoch += 1
            hparams = self.get_hparams()
        else:
            self.last_epoch = epoch
            hparams = getattr(self, "_get_closed_form_hparams", self.get_hparams)()
        for hparam, value in hparams.items():
            self.hparams_dict[hparam] = value
        self._last_hparams = {key: value for key, value in self.hparams_dict.items()}


class LinearHP(HPScheduler):
    def __init__(
        self,
        hparams_dict: Mapping[str, Any],
        start_factor=1,
        end_factor=0.0,
        total_iters=5,
        last_epoch=-1,
    ):
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError(
                "Starting multiplicative factor expected to be greater than 0 and less or equal to 1."
            )

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError("Ending multiplicative factor expected to be between 0 and 1.")

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(hparams_dict, last_epoch)

    def get_hparams(self):

        if self.last_epoch == 0:
            return {
                hparam: value * self.start_factor for hparam, value in self.hparams_dict.items()
            }

        if self.last_epoch > self.total_iters:
            return {hparam: value for hparam, value in self.hparams_dict.items()}

        return {
            hparam: value
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for hparam, value in self.hparams_dict.items()
        }

    def _get_closed_form_hparams(self):
        return {
            hparam: base_value
            * (
                self.start_factor
                + (self.end_factor - self.start_factor)
                * min(self.total_iters, self.last_epoch)
                / self.total_iters
            )
            for hparam, base_value in self.base_hparams.items()
        }


class CosineAnnealingHP(HPScheduler):
    def __init__(self, hparams_dict: Mapping[str, Any], T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(hparams_dict, last_epoch)

    def get_hparams(self):
        if self.last_epoch == 0:
            return {key: value for key, value in self.hparams_dict.items()}
        elif self._step_count == 1 and self.last_epoch > 0:
            return {
                base_hparam: self.eta_min
                + (base_hparam_value - self.eta_min)
                * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                / 2
                for base_hparam, base_hparam_value in self.base_hparams.items()
            }
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return {
                hparam: hparam_value
                + (self.base_hparams[hparam] - self.eta_min)
                * (1 - math.cos(math.pi / self.T_max))
                / 2
                for hparam, hparam_value in self.hparams_dict.items()
            }
        return {
            hparam: (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (hparam_value - self.eta_min)
            + self.eta_min
            for hparam, hparam_value in self.hparams_dict.items()
        }

    def _get_closed_form_hparams(self):
        return {
            hparam: self.eta_min
            + (hparam_value - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for hparam, hparam_value in self.base_hparams.items()
        }


class CosineAnnealingWarmRestartsHP(HPScheduler):
    def __init__(self, hparams_dict: Mapping[str, Any], T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if not isinstance(eta_min, (float, int)):
            raise ValueError(
                f"Expected float or int eta_min, but got {eta_min} of type {type(eta_min)}"
            )
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(hparams_dict, last_epoch)

    def get_hparams(self):
        return {
            hparam: self.eta_min
            + (base_value - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for hparam, base_value in self.base_hparams.items()
        }

    def step(self, epoch=None):
        """Step could be called after every batch update

        Example:
            >>> scheduler = CosineAnnealingWarmRestartsHP(hparams_dict, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)

        This function can be called in an interleaved way.

        Example:
            >>> scheduler = CosineAnnealingWarmRestartsHP(hparams_dict, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        for hparam, value in self.get_hparams().items():
            self.hparams_dict[hparam] = value

        self._last_hparams = {key: value for key, value in self.hparams_dict.items()}
