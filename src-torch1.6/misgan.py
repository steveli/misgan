import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import pylab as plt
import seaborn as sns
from collections import defaultdict
from plot import plot_samples
from utils import CriticUpdater, mkdir, mask_data


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def misgan(args, data_gen, mask_gen, data_critic, mask_critic, data,
           output_dir, checkpoint=None):
    n_critic = args.n_critic
    gp_lambda = args.gp_lambda
    batch_size = args.batch_size
    nz = args.n_latent
    epochs = args.epoch
    plot_interval = args.plot_interval
    save_interval = args.save_interval
    alpha = args.alpha
    tau = args.tau

    gen_data_dir = mkdir(output_dir / 'img')
    gen_mask_dir = mkdir(output_dir / 'mask')
    log_dir = mkdir(output_dir / 'log')
    model_dir = mkdir(output_dir / 'model')

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                             drop_last=True)
    n_batch = len(data_loader)

    data_noise = torch.FloatTensor(batch_size, nz).to(device)
    mask_noise = torch.FloatTensor(batch_size, nz).to(device)

    # Interpolation coefficient
    eps = torch.FloatTensor(batch_size, 1, 1, 1).to(device)

    # For computing gradient penalty
    ones = torch.ones(batch_size).to(device)

    lrate = 1e-4
    # lrate = 1e-5
    data_gen_optimizer = optim.Adam(
        data_gen.parameters(), lr=lrate, betas=(.5, .9))
    mask_gen_optimizer = optim.Adam(
        mask_gen.parameters(), lr=lrate, betas=(.5, .9))

    data_critic_optimizer = optim.Adam(
        data_critic.parameters(), lr=lrate, betas=(.5, .9))
    mask_critic_optimizer = optim.Adam(
        mask_critic.parameters(), lr=lrate, betas=(.5, .9))

    update_data_critic = CriticUpdater(
        data_critic, data_critic_optimizer, eps, ones, gp_lambda)
    update_mask_critic = CriticUpdater(
        mask_critic, mask_critic_optimizer, eps, ones, gp_lambda)

    start_epoch = 0
    critic_updates = 0
    log = defaultdict(list)

    if checkpoint:
        data_gen.load_state_dict(checkpoint['data_gen'])
        mask_gen.load_state_dict(checkpoint['mask_gen'])
        data_critic.load_state_dict(checkpoint['data_critic'])
        mask_critic.load_state_dict(checkpoint['mask_critic'])
        data_gen_optimizer.load_state_dict(checkpoint['data_gen_opt'])
        mask_gen_optimizer.load_state_dict(checkpoint['mask_gen_opt'])
        data_critic_optimizer.load_state_dict(checkpoint['data_critic_opt'])
        mask_critic_optimizer.load_state_dict(checkpoint['mask_critic_opt'])
        start_epoch = checkpoint['epoch']
        critic_updates = checkpoint['critic_updates']
        log = checkpoint['log']

    with (log_dir / 'gpu.txt').open('a') as f:
        print(torch.cuda.device_count(), start_epoch, file=f)

    def save_model(path, epoch, critic_updates=0):
        torch.save({
            'data_gen': data_gen.state_dict(),
            'mask_gen': mask_gen.state_dict(),
            'data_critic': data_critic.state_dict(),
            'mask_critic': mask_critic.state_dict(),
            'data_gen_opt': data_gen_optimizer.state_dict(),
            'mask_gen_opt': mask_gen_optimizer.state_dict(),
            'data_critic_opt': data_critic_optimizer.state_dict(),
            'mask_critic_opt': mask_critic_optimizer.state_dict(),
            'epoch': epoch + 1,
            'critic_updates': critic_updates,
            'log': log,
            'args': args,
        }, str(path))

    sns.set()

    start = time.time()
    epoch_start = start

    for epoch in range(start_epoch, epochs):
        sum_data_loss, sum_mask_loss = 0, 0
        for real_data, real_mask, _, _ in data_loader:
            # Assume real_data and mask have the same number of channels.
            # Could be modified to handle multi-channel images and
            # single-channel masks.
            real_mask = real_mask.float()[:, None]

            real_data = real_data.to(device)
            real_mask = real_mask.to(device)

            masked_real_data = mask_data(real_data, real_mask, tau)

            # Update discriminators' parameters
            data_noise.normal_()
            mask_noise.normal_()

            fake_data = data_gen(data_noise)
            fake_mask = mask_gen(mask_noise)

            masked_fake_data = mask_data(fake_data, fake_mask, tau)

            update_data_critic(masked_real_data, masked_fake_data)
            update_mask_critic(real_mask, fake_mask)

            sum_data_loss += update_data_critic.loss_value
            sum_mask_loss += update_mask_critic.loss_value

            critic_updates += 1

            if critic_updates == n_critic:
                critic_updates = 0

                # Update generators' parameters
                for p in data_critic.parameters():
                    p.requires_grad_(False)
                for p in mask_critic.parameters():
                    p.requires_grad_(False)

                data_noise.normal_()
                mask_noise.normal_()

                fake_data = data_gen(data_noise)
                fake_mask = mask_gen(mask_noise)
                masked_fake_data = mask_data(fake_data, fake_mask, tau)

                data_loss = -data_critic(masked_fake_data).mean()
                data_gen.zero_grad()
                data_loss.backward()
                data_gen_optimizer.step()

                data_noise.normal_()
                mask_noise.normal_()

                fake_data = data_gen(data_noise)
                fake_mask = mask_gen(mask_noise)
                masked_fake_data = mask_data(fake_data, fake_mask, tau)

                data_loss = -data_critic(masked_fake_data).mean()
                mask_loss = -mask_critic(fake_mask).mean()
                mask_gen.zero_grad()
                (mask_loss + data_loss * alpha).backward()
                mask_gen_optimizer.step()

                for p in data_critic.parameters():
                    p.requires_grad_(True)
                for p in mask_critic.parameters():
                    p.requires_grad_(True)

        mean_data_loss = sum_data_loss / n_batch
        mean_mask_loss = sum_mask_loss / n_batch
        log['data loss', 'data_loss'].append(mean_data_loss)
        log['mask loss', 'mask_loss'].append(mean_mask_loss)

        for (name, shortname), trace in log.items():
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(trace)
            ax.set_ylabel(name)
            ax.set_xlabel('epoch')
            fig.savefig(str(log_dir / f'{shortname}.png'), dpi=300)
            plt.close(fig)

        if plot_interval > 0 and (epoch + 1) % plot_interval == 0:
            print(f'[{epoch:4}] {mean_data_loss:12.4f} {mean_mask_loss:12.4f}')

            filename = f'{epoch:04d}.png'

            data_gen.eval()
            mask_gen.eval()

            with torch.no_grad():
                data_noise.normal_()
                mask_noise.normal_()

                data_samples = data_gen(data_noise)
                plot_samples(data_samples, str(gen_data_dir / filename))

                mask_samples = mask_gen(mask_noise)
                plot_samples(mask_samples, str(gen_mask_dir / filename))

            data_gen.train()
            mask_gen.train()

        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            save_model(model_dir / f'{epoch:04d}.pth', epoch, critic_updates)

        epoch_end = time.time()
        time_elapsed = epoch_end - start
        epoch_time = epoch_end - epoch_start
        epoch_start = epoch_end
        with (log_dir / 'time.txt').open('a') as f:
            print(epoch, epoch_time, time_elapsed, file=f)
        save_model(log_dir / 'checkpoint.pth', epoch, critic_updates)

    print(output_dir)
