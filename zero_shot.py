'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * Changed from SLIP
 * https://github.com/facebookresearch/SLIP
 * By Le Xue
'''
import argparse
from collections import OrderedDict
import math
import time, ipdb
import numpy as np
from imagebind.imagebind_model import ModalityType

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import collections

from data.dataset_3d import *

from utils.utils import get_dataset
import models.PointBind_models as models
from imagebind.multimodal_preprocessors import SimpleTokenizer
from utils import utils
from data.dataset_3d import customized_collate_fn


def get_args_parser():
    parser = argparse.ArgumentParser(description='Point-Bind evaluation', add_help=False)
    # Data
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--validate_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    # Model
    parser.add_argument('--model', default='PointBind_I2PMAE', type=str)
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--ckpt_path', default='', help='the ckpt to test 3d zero shot')
    return parser

best_acc1 = 0

def main(args):
    utils.init_distributed_mode(args)
    global best_acc1

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    zero_stats = test_zeroshot_3d(args)
    print(zero_stats)
    return

def test_zeroshot_3d_core(test_loader, model, tokenizer, args=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    model.eval()
    model = utils.get_model(model)

    print('=> encoding captions')
    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    with open(os.path.join("./data", 'labels.json')) as f:
        labels = json.load(f)[args.validate_dataset_name]

    with torch.no_grad():

        def get_pc_features(pc):
            pc_features = model.encode_pc(pc)
            pc_features = model.bind.modality_head_point(pc_features)
            pc_features = model.bind.modality_postprocessor_point(pc_features)
            return pc_features

        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            text = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            inputs = {
                    ModalityType.TEXT: text.squeeze().cuda(),
            }
            embeddings = model.bind(inputs)
            class_embeddings = embeddings[ModalityType.TEXT]
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)

        for i, (pc, target, target_name) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode pc
            pc_features = get_pc_features(pc)

            # cosine similarity as logits
            logits_per_pc = pc_features @ text_features.t()

            # measure accuracy and record loss
            (acc1, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 5))
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        print(','.join(top1_accuracy_per_class.keys()))
        print(','.join([str(value) for value in top1_accuracy_per_class.values()]))
        print(','.join([str(value) for value in top5_accuracy_per_class.values()]))

    progress.synchronize()
    print('0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return {'acc1': top1.avg, 'acc5': top5.avg}

def test_zeroshot_3d(args):
    state_dict = torch.load(args.ckpt_path, map_location='cpu')
    # create model
    model = getattr(models, args.model)(args=args)
    model.cuda()
    model.load_state_dict(state_dict, strict=True)
    print("=> Loaded checkpoint from '{}'".format(args.ckpt_path))

    tokenizer = SimpleTokenizer(bpe_path="bpe/bpe_simple_vocab_16e6.txt.gz")

    test_dataset = get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )
    results = test_zeroshot_3d_core(test_loader, model, tokenizer, args)

    return results


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point-Bind evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
