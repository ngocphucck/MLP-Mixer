import logging
import argparse
import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.optim import SGD
from torch.nn import CrossEntropyLoss


from model.mlp_mixer import MLPMixer
from dataset import DogBreedDataset
from utils.schedule_utils import CosineWarmUpScheduler, LinearWarmUpScheduler


logger = logging.getLogger(__name__)


def setup(args):
    model = MLPMixer(image_shape=args.image_shape, patch_shape=args.patch_shape, depth=args.depth,
                     dim=args.dim, num_classes=args.num_classes)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    model.to(args.device)

    logger.info("Training parameters %s", args)
    summary(model, (3, args.image_shape[0], args.image_shape[1]))

    return args, model


def save_model(args, model):
    torch.save(model.state_dict(), os.path.join(args.output_dir, "%s_checkpoint.pth" % args.name))

    logger.info("Save model checkpoint to [DIR: %s]", args.output_dir)


def get_loader(args):
    train_loader = DataLoader(DogBreedDataset('data/train.json'), batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(DogBreedDataset('data/val.json'), batch_size=args.batch_size,
                            shuffle=True)

    return train_loader, val_loader


def train(args, model):
    train_loader, val_loader = get_loader(args)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.schedule == 'cosine':
        scheduler = CosineWarmUpScheduler(optimizer=optimizer, warmup_steps=args.warmup_steps,
                                          t_total=args.num_steps)
    else:
        scheduler = LinearWarmUpScheduler(optimizer=optimizer, warmup_steps=args.warmup_steps,
                                          t_total=args.num_steps)

    loss_fn = CrossEntropyLoss()

    logger.info("*****************************Training phase*********************************")
    logger.info("Total optimization step = %d", args.num_steps)

    for epoch in range(args.num_steps):
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])

        for step, batch in epoch_iterator:
            optimizer.zero_grad()
            batch = (t.to(args.device) for t in batch)
            input_tensor, label_tensor = batch

            predict_tensor = model(input_tensor)
            loss = loss_fn(predict_tensor, label_tensor)
            loss.backward()

            scheduler.step()
            optimizer.step()

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (epoch, args.num_steps, loss.items())
            )

        logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Implementation of MLP Mixer"
    )
    parser.add_argument("--image_shape", type=tuple, default=(256, 256),
                        help='Image shape for training (default: (256, 256))')
    parser.add_argument("--patch_shape", type=tuple, default=(16, 16),
                        help='Patch shape for splitting (default: (16, 16))')
    parser.add_argument("--depth", type=int, default=6, help='Depth of model (default: 6)')
    parser.add_argument("--dim", type=int, default=512, help='Dim for each mlp mixer block (default: 512)')
    parser.add_argument("--num_classes", type=int, default=120,
                        help='Number of classes for classification (default: 120)')
    parser.add_argument("--checkpoint", type=str, default=None, help='Path to checkpoint (default: None)')
    parser.add_argument("--output_dir", type=str, default='output',
                        help='Path to save checkpoint (default: output/)')
    parser.add_argument("--batch_size", type=int, default=16, help='Number of images in each batch (default: 16)')
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument("--momentum", type=float, default=0.9, help='Momentum (default: 0,9)')
    parser.add_argument("--decay", type=float, default=0, help='Weight decay (default: 0)')
    parser.add_argument("--num_steps", type=int, default=30, help='Number of epochs (default: 30)')
    parser.add_argument("--warmup_step", type=int, default=10,
                        help='Number of epoch for warmup learning rate (default: 10)')
    parser.add_argument("--scheduler", type=str, choices=["cosine", "linear"],
                        default='cosine', help='Scheduler for warmup learning rate (default: cosine)')
