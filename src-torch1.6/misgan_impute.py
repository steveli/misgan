import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import pylab as plt
import seaborn as sns
from collections import defaultdict
from plot import plot_grid, plot_samples
from utils import CriticUpdater, mask_norm, mkdir, mask_data


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def misgan_impute(args, data_gen, mask_gen, imputer,
                  data_critic, mask_critic, impu_critic,
                  data, output_dir, checkpoint=None):
    n_critic = args.n_critic
    gp_lambda = args.gp_lambda
    batch_size = args.batch_size
    nz = args.n_latent
    epochs = args.epoch
    plot_interval = args.plot_interval
    save_model_interval = args.save_interval
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    tau = args.tau
    update_all_networks = not args.imputeronly

    gen_data_dir = mkdir(output_dir / 'img')
    gen_mask_dir = mkdir(output_dir / 'mask')
    impute_dir = mkdir(output_dir / 'impute')
    log_dir = mkdir(output_dir / 'log')
    model_dir = mkdir(output_dir / 'model')

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                             drop_last=True, num_workers=args.workers)
    n_batch = len(data_loader)
    data_shape = data[0][0].shape

    data_noise = torch.FloatTensor(batch_size, nz).to(device)
    mask_noise = torch.FloatTensor(batch_size, nz).to(device)
    impu_noise = torch.FloatTensor(batch_size, *data_shape).to(device)

    # Interpolation coefficient
    eps = torch.FloatTensor(batch_size, 1, 1, 1).to(device)

    # For computing gradient penalty
    ones = torch.ones(batch_size).to(device)

    lrate = 1e-4
    imputer_lrate = 2e-4
    data_gen_optimizer = optim.Adam(
        data_gen.parameters(), lr=lrate, betas=(.5, .9))
    mask_gen_optimizer = optim.Adam(
        mask_gen.parameters(), lr=lrate, betas=(.5, .9))
    imputer_optimizer = optim.Adam(
        imputer.parameters(), lr=imputer_lrate, betas=(.5, .9))

    data_critic_optimizer = optim.Adam(
        data_critic.parameters(), lr=lrate, betas=(.5, .9))
    mask_critic_optimizer = optim.Adam(
        mask_critic.parameters(), lr=lrate, betas=(.5, .9))
    impu_critic_optimizer = optim.Adam(
        impu_critic.parameters(), lr=imputer_lrate, betas=(.5, .9))

    update_data_critic = CriticUpdater(
        data_critic, data_critic_optimizer, eps, ones, gp_lambda)
    update_mask_critic = CriticUpdater(
        mask_critic, mask_critic_optimizer, eps, ones, gp_lambda)
    update_impu_critic = CriticUpdater(
        impu_critic, impu_critic_optimizer, eps, ones, gp_lambda)

    start_epoch = 0
    critic_updates = 0
    log = defaultdict(list)

    if args.resume:
        data_gen.load_state_dict(checkpoint['data_gen'])
        mask_gen.load_state_dict(checkpoint['mask_gen'])
        imputer.load_state_dict(checkpoint['imputer'])
        data_critic.load_state_dict(checkpoint['data_critic'])
        mask_critic.load_state_dict(checkpoint['mask_critic'])
        impu_critic.load_state_dict(checkpoint['impu_critic'])
        data_gen_optimizer.load_state_dict(checkpoint['data_gen_opt'])
        mask_gen_optimizer.load_state_dict(checkpoint['mask_gen_opt'])
        imputer_optimizer.load_state_dict(checkpoint['imputer_opt'])
        data_critic_optimizer.load_state_dict(checkpoint['data_critic_opt'])
        mask_critic_optimizer.load_state_dict(checkpoint['mask_critic_opt'])
        impu_critic_optimizer.load_state_dict(checkpoint['impu_critic_opt'])
        start_epoch = checkpoint['epoch']
        critic_updates = checkpoint['critic_updates']
        log = checkpoint['log']
    elif args.pretrain:
        pretrain = torch.load(args.pretrain, map_location='cpu')
        data_gen.load_state_dict(pretrain['data_gen'])
        mask_gen.load_state_dict(pretrain['mask_gen'])
        data_critic.load_state_dict(pretrain['data_critic'])
        mask_critic.load_state_dict(pretrain['mask_critic'])
        if 'imputer' in pretrain:
            imputer.load_state_dict(pretrain['imputer'])
            impu_critic.load_state_dict(pretrain['impu_critic'])

    with (log_dir / 'gpu.txt').open('a') as f:
        print(torch.cuda.device_count(), start_epoch, file=f)

    def save_model(path, epoch, critic_updates=0):
        torch.save({
            'data_gen': data_gen.state_dict(),
            'mask_gen': mask_gen.state_dict(),
            'imputer': imputer.state_dict(),
            'data_critic': data_critic.state_dict(),
            'mask_critic': mask_critic.state_dict(),
            'impu_critic': impu_critic.state_dict(),
            'data_gen_opt': data_gen_optimizer.state_dict(),
            'mask_gen_opt': mask_gen_optimizer.state_dict(),
            'imputer_opt': imputer_optimizer.state_dict(),
            'data_critic_opt': data_critic_optimizer.state_dict(),
            'mask_critic_opt': mask_critic_optimizer.state_dict(),
            'impu_critic_opt': impu_critic_optimizer.state_dict(),
            'epoch': epoch + 1,
            'critic_updates': critic_updates,
            'log': log,
            'args': args,
        }, str(path))

    sns.set()
    start = time.time()
    epoch_start = start

    for epoch in range(start_epoch, epochs):
        sum_data_loss, sum_mask_loss, sum_impu_loss = 0, 0, 0
        for real_data, real_mask, _, index in data_loader:
            # Assume real_data and real_mask have the same number of channels.
            # Could be modified to handle multi-channel images and
            # single-channel masks.
            real_mask = real_mask.float()[:, None]

            real_data = real_data.to(device)
            real_mask = real_mask.to(device)

            masked_real_data = mask_data(real_data, real_mask, tau)

            # Update discriminators' parameters
            data_noise.normal_()
            fake_data = data_gen(data_noise)

            impu_noise.uniform_()
            imputed_data = imputer(real_data, real_mask, impu_noise)
            masked_imputed_data = mask_data(real_data, real_mask, imputed_data)

            if update_all_networks:
                mask_noise.normal_()
                fake_mask = mask_gen(mask_noise)
                masked_fake_data = mask_data(fake_data, fake_mask, tau)
                update_data_critic(masked_real_data, masked_fake_data)
                update_mask_critic(real_mask, fake_mask)

                sum_data_loss += update_data_critic.loss_value
                sum_mask_loss += update_mask_critic.loss_value

            update_impu_critic(fake_data, masked_imputed_data)
            sum_impu_loss += update_impu_critic.loss_value

            critic_updates += 1

            if critic_updates == n_critic:
                critic_updates = 0

                # Update generators' parameters
                if update_all_networks:
                    for p in data_critic.parameters():
                        p.requires_grad_(False)
                    for p in mask_critic.parameters():
                        p.requires_grad_(False)
                for p in impu_critic.parameters():
                    p.requires_grad_(False)

                impu_noise.uniform_()
                imputed_data = imputer(real_data, real_mask, impu_noise)
                masked_imputed_data = mask_data(real_data, real_mask,
                                                imputed_data)
                impu_loss = -impu_critic(masked_imputed_data).mean()

                if update_all_networks:
                    data_noise.normal_()
                    fake_data = data_gen(data_noise)
                    mask_noise.normal_()
                    fake_mask = mask_gen(mask_noise)
                    masked_fake_data = mask_data(fake_data, fake_mask, tau)
                    data_loss = -data_critic(masked_fake_data).mean()
                    mask_loss = -mask_critic(fake_mask).mean()

                    mask_gen.zero_grad()
                    (mask_loss + data_loss * alpha).backward(retain_graph=True)
                    mask_gen_optimizer.step()

                    data_noise.normal_()
                    fake_data = data_gen(data_noise)
                    mask_noise.normal_()
                    fake_mask = mask_gen(mask_noise)
                    masked_fake_data = mask_data(fake_data, fake_mask, tau)
                    data_loss = -data_critic(masked_fake_data).mean()

                    data_gen.zero_grad()
                    (data_loss + impu_loss * beta).backward(retain_graph=True)
                    data_gen_optimizer.step()

                imputer.zero_grad()
                if gamma > 0:
                    imputer_mismatch_loss = mask_norm(
                        (imputed_data - real_data)**2, real_mask)
                    (impu_loss + imputer_mismatch_loss * gamma).backward()
                else:
                    impu_loss.backward()
                imputer_optimizer.step()

                if update_all_networks:
                    for p in data_critic.parameters():
                        p.requires_grad_(True)
                    for p in mask_critic.parameters():
                        p.requires_grad_(True)
                for p in impu_critic.parameters():
                    p.requires_grad_(True)

        if update_all_networks:
            mean_data_loss = sum_data_loss / n_batch
            mean_mask_loss = sum_mask_loss / n_batch
            log['data loss', 'data_loss'].append(mean_data_loss)
            log['mask loss', 'mask_loss'].append(mean_mask_loss)
        mean_impu_loss = sum_impu_loss / n_batch
        log['imputer loss', 'impu_loss'].append(mean_impu_loss)

        if plot_interval > 0 and (epoch + 1) % plot_interval == 0:
            if update_all_networks:
                print('[{:4}] {:12.4f} {:12.4f} {:12.4f}'.format(
                    epoch, mean_data_loss, mean_mask_loss, mean_impu_loss))
            else:
                print('[{:4}] {:12.4f}'.format(epoch, mean_impu_loss))

            filename = f'{epoch:04d}.png'
            with torch.no_grad():
                data_gen.eval()
                mask_gen.eval()
                imputer.eval()

                data_noise.normal_()
                mask_noise.normal_()

                data_samples = data_gen(data_noise)
                plot_samples(data_samples, str(gen_data_dir / filename))

                mask_samples = mask_gen(mask_noise)
                plot_samples(mask_samples, str(gen_mask_dir / filename))

                # Plot imputation results
                impu_noise.uniform_()
                imputed_data = imputer(real_data, real_mask, impu_noise)
                imputed_data = mask_data(real_data, real_mask, imputed_data)
                if hasattr(data, 'mask_info'):
                    bbox = [data.mask_info[idx] for idx in index]
                else:
                    bbox = None
                plot_grid(imputed_data, bbox, gap=2,
                          save_file=str(impute_dir / filename))

                data_gen.train()
                mask_gen.train()
                imputer.train()

        for (name, shortname), trace in log.items():
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(trace)
            ax.set_ylabel(name)
            ax.set_xlabel('epoch')
            fig.savefig(str(log_dir / f'{shortname}.png'), dpi=300)
            plt.close(fig)

        if save_model_interval > 0 and (epoch + 1) % save_model_interval == 0:
            save_model(model_dir / f'{epoch:04d}.pth', epoch, critic_updates)

        epoch_end = time.time()
        time_elapsed = epoch_end - start
        epoch_time = epoch_end - epoch_start
        epoch_start = epoch_end
        with (log_dir / 'epoch-time.txt').open('a') as f:
            print(epoch, epoch_time, time_elapsed, file=f)
        save_model(log_dir / 'checkpoint.pth', epoch, critic_updates)

    print(output_dir)
