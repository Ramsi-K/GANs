from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import matplotlib.pyplot as plt
import config


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_gp(real, fake, crit, alpha, gamma=10):
    mix_images = real * alpha + fake * (1 - alpha)  # 128 x 3 x 128 x 128
    mix_scores = crit(mix_images)  # 128 x 1

    gradient = torch.autograd.grad(
        inputs=mix_images,
        outputs=mix_scores,
        grad_outputs=torch.ones_like(mix_scores),
        retain_graph=True,
        create_graph=True,
    )[
        0
    ]  # 128 x 3 x 128 x 128

    gradient = gradient.view(len(gradient), -1)  # 128 x 49152
    gradient_norm = gradient.norm(2, dim=1)
    gp = gamma * ((gradient_norm - 1) ** 2).mean()

    return gp


# def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
#     BATCH_SIZE, C, H, W = real.shape
#     beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
#     interpolated_images = real * beta + fake.detach() * (1 - beta)
#     interpolated_images.requires_grad_(True)

#     # Calculate critic scores
#     mixed_scores = critic(interpolated_images, alpha, train_step)

#     # Take the gradient of the scores with respect to the images
#     gradient = torch.autograd.grad(
#         inputs=interpolated_images,
#         outputs=mixed_scores,
#         grad_outputs=torch.ones_like(mixed_scores),
#         create_graph=True,
#         retain_graph=True,
#     )[0]
#     gradient = gradient.view(gradient.shape[0], -1)
#     gradient_norm = gradient.norm(2, dim=1)
#     gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
#     return gradient_penalty


def generate_examples():
    # MORPHING, interpolation between points in latent space
    gen_set = []
    z_shape = [1, 200, 1, 1]
    rows = 4
    steps = 17

    for i in range(rows):
        z1, z2 = torch.randn(z_shape), torch.randn(z_shape)
        for alpha in np.linspace(0, 1, steps):
            z = alpha * z1 + (1 - alpha) * z2
            res = gen(z.cuda())[0]
            gen_set.append(res)

    fig = plt.figure(figsize=(25, 11))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, steps), axes_pad=0.1)

    for ax, img in zip(grid, gen_set):
        ax.axis("off")
        res = img.cpu().detach().permute(1, 2, 0)
        res = res - res.min()
        res = res / (res.max() - res.min())
        ax.imshow(res.clip(0, 1.0))

    plt.show()
