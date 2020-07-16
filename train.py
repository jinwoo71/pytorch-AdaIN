import argparse
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torch.utils.tensorboard import SummaryWriter
import utils
from torchvision import transforms
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper
import torchvision.utils
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Class to unnormalize tensor (created to print a picture)
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter('runs/')

decoder = net.decoder
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)

# Train with ImageNet
traindir = os.path.join('/home_goya/jinwoo.choi/ImageNet/train/')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
jittering = utils.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
lighting = utils.Lighting(alphastd=0.1, eigval=[0.2175, 0.0188, 0.0045],
                          eigvec=[[-0.5675, 0.7192, 0.4009],
                                  [-0.5808, -0.0045, -0.8140],
                                  [-0.5836, -0.6948, 0.4203]])
content_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #jittering,
        #lighting,
        normalize,
    ]))
content_sampler = None

style_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
#        jittering,
#        lighting,
        normalize,
    ]))
style_sampler = None

content_iter = torch.utils.data.DataLoader(
    content_dataset, batch_size=args.batch_size, shuffle=(content_sampler is None),
    num_workers=args.n_threads, pin_memory=True, sampler=content_sampler)
style_iter = torch.utils.data.DataLoader(
    style_dataset, batch_size=args.batch_size, shuffle=(style_sampler is None),
    num_workers=args.n_threads, pin_memory=True, sampler=style_sampler)
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images, _ = next(iter(content_iter))
    style_images, _ = next(iter(style_iter))
    content_images = content_images.to(device)
    style_images = style_images.to(device)

#    The code below is for outputting an image.
#    content_grid = torchvision.utils.make_grid(
#        [unorm(tensor) for tensor in content_images])
#    style_grid = torchvision.utils.make_grid(
#        [unorm(tensor) for tensor in style_images])
#    writer.add_image('content', content_grid)
#    writer.add_image('style', style_grid)

    loss_c, loss_s = network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar('loss_content', loss_c.item(), i + 1)
    writer.add_scalar('loss_style', loss_s.item(), i + 1)

    # Save the model every 100 times.
    #if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
    if (i + 1) % 100 == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            print(key)
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'decoder_iter_{:d}.pth.tar'.format(i + 1))
writer.close()
