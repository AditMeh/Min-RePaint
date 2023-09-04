from diffusers import UNet2DModel

from torchvision.datasets import CelebA
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
    epochs = 30
    lr = 0.00001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists("checkpoints_celeba/"):
        os.mkdir("checkpoints_celeba/")
    if not os.path.exists("visuals_celeba/"):
        os.mkdir("visuals_celeba/")
    
    train_dataset = CelebA(
        root="./data/celeba",
        download=True,
        split="train",
        transform=transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        ),
    )

    train = torch.utils.data.DataLoader(
        train_dataset, batch_size=22, shuffle=True, num_workers=0
    )

    unet = UNet2DModel(sample_size=(128, 128), in_channels=3, out_channels=3).to(device)
    schedule = compute_schedule(T, beta_min, beta_max, device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    pbar = tqdm.tqdm(range(1, epochs + 1))
    step = 0
    for epoch in pbar:
        acc_loss, denom = 0, 0
        for x, _ in tqdm.tqdm(train):
            x = x.to(device)
            
            # Forward diffuse
            ts = torch.randint(1, T + 1, (x.shape[0],), device=device)
            eps = torch.randn(*x.shape, device=device)

            x_pass = (
                schedule["sqrt_alpha_bar"][ts - 1][..., None, None, None] * x
                + schedule["sqrt_one_minus_alpha_bar"][ts - 1][..., None, None, None]
                * eps
            )

            pred = unet(x_pass, ts.float())["sample"]
            loss = torch.nn.MSELoss()(pred, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_loss += loss.item()
            denom += x.shape[0]

            step += 1
            
            if step % 2000 == 0:
                sampled_imgs = sample(T, schedule, [4] + list(x.shape[1:]), unet, device)
                save_image(rescale(sampled_imgs), fp=f"visuals_celeba/epoch_{epoch}_step_{step}.png")
                torch.save(unet.state_dict(), f"checkpoints_celeba/epoch_{epoch}_step_{step}.pt")

                
        print("loss: ", str(acc_loss / denom))


        torch.save(unet.state_dict(), f"checkpoints_celeba/epoch_{epoch}_step_{step}.pt")

        sampled_imgs = sample(T, schedule, [4] + list(x.shape[1:]), unet, device)
        sampled_imgs = torchvision.utils.make_grid(sampled_imgs)
        save_image(sampled_imgs, fp=f"visuals_celeba/epoch_{epoch}_step_{step}.png")


if __name__ == "__main__":
    main()
