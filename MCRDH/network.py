import torch
import torchvision
from torch import nn


def build_model(args, pretrained=True):
    net = MyNet(args.backbone, args.n_bits, args.n_classes, pretrained).to(args.device)
    return net, 0


class MyNet(nn.Module):
    def __init__(self, backbone, n_bits, n_classes, pretrained):
        super().__init__()
        weights = torchvision.models.get_model_weights(backbone)["IMAGENET1K_V1"] if pretrained else None
        net = torchvision.models.__dict__[backbone](weights=weights)

        if "resnet" in backbone:
            in_features = net.fc.in_features
            net.fc = nn.Identity()
        elif "vit_" in backbone:
            in_features = net.heads[-1].in_features
            net.heads[-1] = nn.Identity()
        else:
            raise NotImplementedError

        self.fc = nn.Linear(in_features, n_bits)
        self.bn = nn.BatchNorm1d(n_bits)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(n_bits, n_classes, bias=False)

        self.backbone = net

        if pretrained:
            self.fc.apply(self.init_weights)
            self.classifier.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if m.__class__.__name__.find("Linear") != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        y = self.classifier(self.bn(x))

        return x, y


if __name__ == "__main__":
    x = torch.randn(64, 3, 224, 224)
    model = MyNet("vit_b_32", 16, 10, True)
    out = model(x)
    for x in out:
        print(x.shape)
