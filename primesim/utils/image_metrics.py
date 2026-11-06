from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchmetrics.image import UniversalImageQualityIndex
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_tensor


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def mse(img1, img2):  # batched input only
    return ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean()


def psnr(img1, img2):  # batched input only
    mse = ((img1 - img2) ** 2).reshape(img1.shape[0], -1).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def uiqi(img1, img2):  # img1, img2: [T,C,H,W] in [0,1]
    uiqi_metric = UniversalImageQualityIndex(sync_on_compute=False)
    return uiqi_metric(img1, img2)


def lpips(img1, img2, model=None):  # img1, img2: [T,C,H,W] in [-1,1]
    _model = (
        model
        if model is not None
        else LearnedPerceptualImagePatchSimilarity(sync_on_compute=False)
    )
    _model.update(img1, img2)
    score = _model.compute().item()
    _model.reset()
    return score


@torch.no_grad()
def eval_images(vid1, vid2, lpips_model=None, prefix="train/img_recon_"):
    """Compute image/video quality metrics between two lists of PIL frames.

    Args:
        vid1: list of PIL images (generated)
        vid2: list of PIL images (ground truth)
        lpips_model: optional LearnedPerceptualImagePatchSimilarity instance.
            If None, a new one is created and discarded each call.
        prefix: wandb key prefix, e.g. "train/img_recon_" or "val/img_recon_".
            Lets the caller separate the same metrics into train/val namespaces.
    """
    sample1 = torch.stack([to_tensor(frame) for frame in vid1])  # [T,C,H,W], [0,1]
    sample2 = torch.stack([to_tensor(frame) for frame in vid2])
    s1_n = sample1 * 2 - 1  # [-1,1]
    s2_n = sample2 * 2 - 1

    metrics = {}
    metrics[f"{prefix}l1"] = l1_loss(sample1, sample2)
    metrics[f"{prefix}ssim"] = ssim(sample1, sample2)
    metrics[f"{prefix}mse"] = mse(sample1, sample2)
    metrics[f"{prefix}psnr"] = psnr(sample1, sample2)
    metrics[f"{prefix}uiqi"] = uiqi(sample1, sample2)
    metrics[f"{prefix}lpips"] = lpips(s1_n, s2_n, model=lpips_model)

    return metrics
