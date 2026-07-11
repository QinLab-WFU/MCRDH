from argparse import Namespace

import torch
from torch import nn

from Baseline.loss import TripletLoss
from CTL.loss import LabelSmoothingCrossEntropy, CenterLoss, prepare_centroids


class CTLLoss(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.contrastive_loss = TripletLoss()
        self.xent = LabelSmoothingCrossEntropy()
        self.center_loss = CenterLoss(args.n_classes, args.n_bits)
        self.warmup = args.warmup

    def forward(self, inputs, logits, labels, epoch):
        # TripletLoss
        tl = self.contrastive_loss(inputs, labels, inputs)

        # CenterLoss
        cl = self.center_loss(inputs, labels)

        # ClassificationLoss
        xl = self.xent(logits, labels)

        # CentroidTripletLoss
        if epoch + 1 > self.warmup:
            ctl = self.contrastive_loss(*prepare_centroids(inputs, labels, inputs))
        else:
            ctl = 0

        return tl, cl, xl, ctl


if __name__ == "__main__":
    from _utils import gen_test_data

    e, t, l = gen_test_data(64, 10, 8, False)
    x = torch.randn(2, 10)

    # loss = LabelSmoothingCrossEntropy()
    # print(loss(x, l))
    #
    # loss2 = LSCE()
    # print(loss2(x, t))
