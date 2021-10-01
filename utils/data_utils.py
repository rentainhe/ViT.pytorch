import logging
import os
import torch
import torch.distributed as dist

from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

logger = logging.getLogger(__name__)



class _DataLoader(DataLoader):
    def __init__(self, args, split='train', batch_size=16, num_workers=8):
        self.args = args
        self.num_tasks = dist.get_world_size()
        # define transform
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        # define dataset
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        if args.dataset == "cifar10":
            self.dataset = CIFAR10(root=args.data_dir, train=True if split=="train" else False, transform=transform, download=True)
        elif args.dataset == "cifar100":
            self.dataset = CIFAR100(root=args.data_dir, train=True if split=="train" else False, transform=transform, download=True)
        elif args.dataset == "imagenet":
            self.dataset = ImageFolder(root=os.path.join(args.data_dir, split), transform=transform)
        else:
            raise NotImplementedError("We only support cifar10, cifar100, imagenet2012 now")
        if args.local_rank == 0:
            torch.distributed.barrier()

        # define sampler
        if split == 'train':
            self.sampler = RandomSampler(self.dataset) if args.local_rank == -1 else DistributedSampler(self.dataset)
        else:
            # TODO: try a dist val sampler here
            if len(self.dataset) % self.num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            self.sampler = DistributedSampler(self.dataset, num_replicas=self.num_tasks, rank=args.local_rank, shuffle=False)

            # self.sampler = SequentialSampler(self.dataset)
    
        super(_DataLoader, self).__init__(
            dataset=self.dataset,
            sampler=self.sampler,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def get_loader(args):
    train_loader = _DataLoader(args, split='train', batch_size=args.train_batch_size, num_workers=args.num_workers)
    val_loader = _DataLoader(args, split='val', batch_size=args.eval_batch_size, num_workers=args.num_workers)
    return train_loader, val_loader
