import torch
import torch.nn as nn

from primesim.loss import LOSS


@LOSS.register_module()
class DMDLoss(nn.Module):
    """Stage 3 losses (asymmetric distribution matching distillation), on frames 1..
    (frame 0 is the clean first frame).

    generator_loss: with real_x0 / fake_x0 the clean video the frozen real score /
    the critic predict from the noised generator output G, the generator is pushed
    along g = (fake_x0 - real_x0) / mean|G - real_x0| through the surrogate
    0.5 * (G - sg(G - g))^2, in float64.
    critic_loss: flow matching of the critic (fake score) on the generator's own
    output, velocity target noise - G.
    """

    @staticmethod
    def generator_loss(generated, fake_x0, real_x0):
        dims = tuple(range(1, generated.ndim))
        normalizer = (generated.detach() - real_x0).abs().mean(dim=dims, keepdim=True)
        gradient = torch.nan_to_num((fake_x0 - real_x0) / normalizer.clamp_min(1e-8))
        prediction = generated[:, :, :, 1:].double()
        target = (generated - gradient).detach()[:, :, :, 1:].double()
        return (0.5 * (prediction - target).pow(2)).mean()

    @staticmethod
    def critic_loss(flow, noise, generated):
        target = (noise - generated)[:, :, :, 1:].float()
        return (flow[:, :, :, 1:].float() - target).pow(2).mean()
