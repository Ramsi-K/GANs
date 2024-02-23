import torch
from tqdm.auto import tqdm
import torchvision
import os
import PIL
import pdb
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Dataset(Dataset):
    def __init__(self, path, size=128, lim=10000):
        self.sizes = [size, size]
        items, labels = [], []

        for data in os.listdir(path)[:lim]:
            # path: './data/celeba/img_align_celeba'
            # data: '114568.jpg
            item = os.path.join(path, data)
            items.append(item)
            labels.append(data)
        self.items = items
        self.labels = labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data = PIL.Image.open(self.items[idx]).convert("RGB")  # (178,218)
        data = np.asarray(
            torchvision.transforms.Resize(self.sizes)(data)
        )  # 128 x 128 x 3
        data = np.transpose(data, (2, 0, 1)).astype(
            np.float32, copy=False
        )  # 3 x 128 x 128 # from 0 to 255
        data = torch.from_numpy(data).div(255)  # from 0 to 1
        return data, self.labels[idx]


def main():
    for epoch in range(n_epochs):
        for real, _ in tqdm(dataloader):
            cur_bs = len(real)  # 128
            real = real.to(device)

            ### CRITIC
            mean_crit_loss = 0
            for _ in range(crit_cycles):
                crit_opt.zero_grad()

                noise = gen_noise(cur_bs, z_dim)
                fake = gen(noise)
                crit_fake_pred = crit(fake.detach())
                crit_real_pred = crit(real)

                alpha = torch.rand(
                    len(real), 1, 1, 1, device=device, requires_grad=True
                )  # 128 x 1 x 1 x 1
                gp = get_gp(real, fake.detach(), crit, alpha)

                crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp

                mean_crit_loss += crit_loss.item() / crit_cycles

                crit_loss.backward(retain_graph=True)
                crit_opt.step()

            crit_losses += [mean_crit_loss]

            ### GENERATOR
            gen_opt.zero_grad()
            noise = gen_noise(cur_bs, z_dim)
            fake = gen(noise)
            crit_fake_pred = crit(fake)

            gen_loss = -crit_fake_pred.mean()
            gen_loss.backward()
            gen_opt.step()

            gen_losses += [gen_loss.item()]

            ### Stats

            if wandbact == 1:
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Step": cur_step,
                        "Critic loss": mean_crit_loss,
                        "Gen loss": gen_loss,
                    }
                )

            if cur_step % save_step == 0 and cur_step > 0:
                print("Saving checkpoint: ", cur_step, save_step)
                save_checkpoint("latest")

            if cur_step % show_step == 0 and cur_step > 0:
                show(fake, wandbactive=1, name="fake")
                show(real, wandbactive=1, name="real")

                gen_mean = sum(gen_losses[-show_step:]) / show_step
                crit_mean = sum(crit_losses[-show_step:]) / show_step
                print(
                    f"Epoch: {epoch}: Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}"
                )

                plt.plot(
                    range(len(gen_losses)),
                    torch.Tensor(gen_losses),
                    label="Generator Loss",
                )

                plt.plot(
                    range(len(gen_losses)),
                    torch.Tensor(crit_losses),
                    label="Critic Loss",
                )

                plt.ylim(-150, 150)
                plt.legend()
                plt.show()

            cur_step += 1
