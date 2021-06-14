from torch import nn
from functools import partial


class NormResidualBlock(nn.Module):
    def __init__(self, dim, fn):
        super(NormResidualBlock, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_tensor):

        return self.fn(self.norm(input_tensor)) + input_tensor


def feed_forward(dim, dense, expand_factor=4, dropout=0.):

    return nn.Sequential(
        dense(dim, dim * expand_factor),
        nn.Dropout(dropout),
        nn.GELU(),
        dense(dim * expand_factor, dim),
        nn.Dropout(dropout),
    )


class MLPMixerBlock(nn.Module):
    def __init__(self, patches, channels, expand_factor=4, dropout=0.):
        super(MLPMixerBlock, self).__init__()
        self.token_mixing = feed_forward(patches, expand_factor=expand_factor, dropout=dropout,
                                         dense=partial(nn.Conv1d, kernel_size=1))
        self.token_residual = NormResidualBlock(fn=self.token_mixing, dim=channels)
        self.channel_mixing = feed_forward(dim=channels, expand_factor=expand_factor, dropout=dropout, dense=nn.Linear)
        self.channel_residual = NormResidualBlock(fn=self.channel_mixing, dim=channels)

    def forward(self, input_tensor):
        output_tensor = self.token_residual(input_tensor)
        output_tensor = self.channel_residual(output_tensor)

        return output_tensor


if __name__ == '__main__':
    pass
