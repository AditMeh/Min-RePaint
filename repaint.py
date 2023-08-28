from diffusers import UNet2DModel

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision

import torch
import tqdm
import os


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def compute_schedule(T, beta_min, beta_max, device):
    betas = torch.linspace(beta_min, beta_max, steps=T, device=device)
    alphas = 1 - betas

    std_t = torch.sqrt(betas)
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

    schedule_hparams = {
        "std_t": std_t,
        "alphas": alphas,
        "sqrt_alpha_bar": sqrt_alpha_bar,
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar,
        "oneover_sqrta": 1 / torch.sqrt(alphas),
        "mab_over_sqrtmab": (1 - alphas) / sqrt_one_minus_alpha_bar,
    }
    return schedule_hparams


def sample(T, schedule, img_shape, unet, device):
    with torch.no_grad():
        seed = torch.randn(*img_shape).to(device=device)
        for i in range(T, 0, -1):
            z = torch.randn(*img_shape).to(device=device) if i > 1 else 0
            ts = torch.ones(1).to(device) * i

            pred_eps = unet(seed, ts.float())["sample"]
            term1 = schedule["oneover_sqrta"][i - 1]
            term2 = seed - (schedule["mab_over_sqrtmab"][i - 1] * pred_eps)
            term3 = z * schedule["std_t"][i - 1]

            seed = term1 * term2 + term3

        return rescale(seed)


def main():

    # HyperParameters
    T = 1000
    beta_min = 1e-4
    beta_max = 0.02
    U = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = sorted(os.listdir("checkpoints/"),
                        key=lambda x: int(x.split(".")[0].split("_")[-1]))

    last_checkpoint = checkpoint[-1]

    train_dataset = MNIST(
        "./data",
        download=True,
        train=True,
        transform=transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        ),
    )

    train = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0
    )

    unet = UNet2DModel(sample_size=(32, 32), in_channels=1,
                       out_channels=1).to(device)

    print("Loaded: ", f'./checkpoints/{last_checkpoint}')
    unet.load_state_dict(torch.load(
        f'./checkpoints/{last_checkpoint}', map_location=device))
    schedule = compute_schedule(T, beta_min, beta_max, device)

    x = next(iter(train))[0]
    x = x.to(device)

    mask = torch.ones_like(x)
    mask[..., 12:20, 12:20] = 0

    seed = torch.randn_like(x, device=device)

    with torch.no_grad():
        for t in tqdm.tqdm(range(T, 0, -1)):
            for u in tqdm.tqdm(range(1, U+1)):
                eps = torch.randn(
                    *x.shape, device=device) if t > 1 else torch.zeros(*x.shape, device=device)
                ts = torch.zeros(x.shape[0], device=device, dtype=torch.int32) + t

                # Known
                x_t_known = schedule["sqrt_alpha_bar"][ts - 1] * x + \
                    schedule["sqrt_one_minus_alpha_bar"][ts - 1] * eps

                # Unknown
                z = torch.randn(
                    *x.shape, device=device) if t > 1 else torch.zeros(*x.shape, device=device)

                pred_eps = unet(seed, ts.float())["sample"]
                term1 = schedule["oneover_sqrta"][t - 1]
                term2 = seed - (schedule["mab_over_sqrtmab"][t - 1] * pred_eps)
                term3 = z * schedule["std_t"][t - 1]
                x_t_unknown = term1 * term2 + term3

                # Compose
                composed = mask * x_t_known + (1-mask) * x_t_unknown
                
                if u < U and t > 1:
                    stdnorm = torch.randn_like(x, device=device)
                    composed = torch.sqrt(schedule["alphas"][t - 2]) * composed + torch.sqrt(
                        1 - schedule["alphas"][t - 2]) * stdnorm

                seed = composed

            save_image(seed, fp=f"{t}.png")


if __name__ == "__main__":
    main()
