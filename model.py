import torch
import torch.nn as nn
import torchvision.models as models

class CRNN_ResNet(nn.Module):
    def __init__(self, vocab_size, chars):
        super().__init__()
        self.vocab_size = vocab_size
        self.chars = chars
        self.char2idx, self.idx2char = self.char_idx()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        resnet.layer3[0].conv1.stride = (2, 1)
        resnet.layer3[0].downsample[0].stride = (2, 1)
        resnet.layer4[0].conv1.stride = (2, 1)
        resnet.layer4[0].downsample[0].stride = (2, 1)

        self.conv_base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        self.mapSeq = nn.Linear(512 * 2, 256)
        self.lstm_0 = nn.LSTM(256, 256, bidirectional=True)
        self.lstm_1 = nn.LSTM(512, 256, bidirectional=True)
        self.out = nn.Linear(512, vocab_size)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        x = self.conv_base(x)
        x = x.permute(3, 0, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)
        x, _ = self.lstm_0(x)
        x, _ = self.lstm_1(x)
        return self.out(x)

    def char_idx(self):
        char2idx = {char: i + 1 for i, char in enumerate(self.chars)}
        idx2char = {i + 1: char for i, char in enumerate(self.chars)}
        char2idx['-'], idx2char[0] = 0, '-'
        return char2idx, idx2char

    def decode(self, logits):
        pred_indices = torch.argmax(logits, dim=2).transpose(0, 1)
        decoded_labels = []
        for batch_idx in range(pred_indices.size(0)):
            seq_indices = pred_indices[batch_idx]
            collapsed = []
            for i in range(len(seq_indices)):
                if seq_indices[i] != 0 and (i == 0 or seq_indices[i] != seq_indices[i-1]):
                    collapsed.append(seq_indices[i])
            decoded_labels.append("".join([self.idx2char[idx.item()] for idx in collapsed]))
        return decoded_labels