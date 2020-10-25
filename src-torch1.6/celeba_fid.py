import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from celeba_generator import ConvDataGenerator
from fid import BaseSampler, BaseImputationSampler
from masked_celeba import BlockMaskedCelebA, IndepMaskedCelebA
from celeba_unet_imputer import UNetImputer
from fid import FID


parser = argparse.ArgumentParser()
parser.add_argument('root_dir')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--skip-exist', action='store_true')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class CelebAFID(FID):
    def __init__(self, batch_size=256, data_name='celeba',
                 workers=0, verbose=True):
        self.batch_size = batch_size
        self.workers = workers
        super().__init__(data_name, verbose)

    def complete_data(self):
        data = datasets.ImageFolder(
            'celeba',
            transforms.Compose([
                transforms.CenterCrop(108),
                transforms.Resize(size=64, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
            ]))

        images = len(data)
        data_loader = DataLoader(
            data, batch_size=self.batch_size, num_workers=self.workers)

        return data_loader, images


class MisGANSampler(BaseSampler):
    def __init__(self, data_gen, images=60000, batch_size=256):
        super().__init__(images)
        self.data_gen = data_gen
        self.batch_size = batch_size
        latent_dim = 128
        self.data_noise = torch.FloatTensor(batch_size, latent_dim).to(device)

    def sample(self):
        self.data_noise.normal_()
        return self.data_gen(self.data_noise)


class MisGANImputationSampler(BaseImputationSampler):
    def __init__(self, data_loader, imputer, batch_size=256):
        super().__init__(data_loader)
        self.imputer = imputer
        self.impu_noise = torch.FloatTensor(batch_size, 3, 64, 64).to(device)

    def impute(self, data, mask):
        if data.shape[0] != self.impu_noise.shape[0]:
            self.impu_noise.resize_(data.shape)
        self.impu_noise.uniform_()
        return self.imputer(data, mask, self.impu_noise)


def get_data_loader(args, batch_size):
    if args.mask == 'indep':
        data = IndepMaskedCelebA(
            data_dir=args.data_dir,
            obs_prob=args.obs_prob, obs_prob_high=args.obs_prob_high)
    elif args.mask == 'block':
        data = BlockMaskedCelebA(
            data_dir=args.data_dir, block_len=args.block_len)

    data_size = len(data)
    data_loader = DataLoader(
        data, batch_size=batch_size, num_workers=args.workers)
    return data_loader, data_size


def parallelize(model):
    return nn.DataParallel(model).to(device)


def pretrained_misgan_fid(model_file, samples=202599):
    model = torch.load(model_file, map_location='cpu')
    data_gen = parallelize(ConvDataGenerator())
    data_gen.load_state_dict(model['data_gen'])

    batch_size = args.batch_size

    compute_fid = CelebAFID(batch_size=batch_size)
    sampler = MisGANSampler(data_gen, samples, batch_size)
    gen_fid = compute_fid.fid(sampler, samples)
    print(f'fid: {gen_fid:.2f}')

    imp_fid = None
    if 'imputer' in model:
        imputer = UNetImputer().to(device)
        imputer.load_state_dict(model['imputer'])
        data_loader, data_size = get_data_loader(model['args'], batch_size)
        imputation_sampler = MisGANImputationSampler(
            data_loader, imputer, batch_size)
        imp_fid = compute_fid.fid(imputation_sampler, data_size)
        print(f'impute fid: {imp_fid:.2f}')

    return gen_fid, imp_fid


def main():
    root_dir = Path(args.root_dir)
    fid_file = root_dir / 'fid.txt'
    if args.skip_exist and fid_file.exists():
        return
    try:
        model_file = max((root_dir / 'model').glob('*.pth'))
    except ValueError:
        return

    print(root_dir.name)
    fid, imp_fid = pretrained_misgan_fid(model_file)

    with fid_file.open('w') as f:
        print(fid, file=f)

    if imp_fid is not None:
        with (root_dir / 'impute-fid.txt').open('w') as f:
            print(imp_fid, file=f)


if __name__ == '__main__':
    main()
