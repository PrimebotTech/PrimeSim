import torch


class FlowMatchScheduler:
    """Wan2.2 flow-matching schedule: x_t = (1 - sigma) * x_0 + sigma * noise, the model
    predicts the velocity noise - x_0, and sigmas are shifted toward the noisy end by
    sigma' = shift * sigma / (1 + (shift - 1) * sigma)."""

    num_train_timesteps = 1000

    def set_timesteps(self, num_inference_steps=100, shift=5.0):
        """`num_inference_steps` sigmas evenly spaced in (0, 1], shifted; training uses
        1000 of them and samples one per example."""
        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def _lookup_sigma(self, timestep, device=None):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep[..., None]).abs(), dim=-1)
        sigma = self.sigmas[timestep_id]
        # A zero timestep denotes an exactly clean teacher-forcing context.
        sigma = torch.where(timestep <= 0, torch.zeros_like(sigma), sigma)
        if device is not None:
            sigma = sigma.to(device=device)
        return sigma

    @staticmethod
    def _expand_sigma_for_sample(sigma, sample):
        if sigma.ndim == 0:
            return sigma.reshape(*([1] * sample.ndim))
        if sigma.ndim == 1:
            return sigma.reshape(sigma.shape[0], *([1] * (sample.ndim - 1)))
        if sigma.ndim == 2 and sample.ndim == 5:
            return sigma.reshape(sigma.shape[0], 1, sigma.shape[1], 1, 1)
        if sigma.ndim == 2 and sample.ndim == 6:
            return sigma.reshape(sigma.shape[0], 1, 1, sigma.shape[1], 1, 1)
        raise ValueError(
            f"Unsupported timestep/sigma shape {tuple(sigma.shape)} "
            f"for sample shape {tuple(sample.shape)}"
        )

    def step(self, model_output, timestep, sample):
        """One Euler step from `timestep` to the next sigma (0 after the last)."""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_ - sigma)

    def add_noise(self, original_samples, noise, timestep):
        sigma = self._lookup_sigma(timestep, device=noise.device)
        sigma = self._expand_sigma_for_sample(sigma, original_samples)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.to(dtype=original_samples.dtype)

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        """Per-timestep loss weight (stage 1): a Gaussian bump over the schedule,
        highest mid-way, shifted to 0 at its minimum and scaled to mean 1 over a
        1000-step training schedule."""
        steps = self.num_train_timesteps
        y = torch.exp(-2 * ((self.timesteps - steps / 2) / steps) ** 2)
        y = y - y.min()
        weights = y * (steps / y.sum())
        if len(self.timesteps) != steps:  # empirical correction for shorter schedules
            weights = weights * (len(self.timesteps) / steps)
            weights = weights + weights[1]
        timestep_id = torch.argmin(
            (self.timesteps - timestep.to(self.timesteps.device)[..., None]).abs(),
            dim=-1,
        )
        return weights[timestep_id]
