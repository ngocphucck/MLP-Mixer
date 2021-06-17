from torch import nn
from torchsummary import summary


from model.module import MLPMixerBlock


class MLPMixer(nn.Module):
    def __init__(self, image_shape, patch_shape, depth, dim, num_classes):
        super(MLPMixer, self).__init__()
        self.image_shape = image_shape
        self.in_channels = 3
        self.patch_shape = patch_shape
        self.depth = depth
        self.num_classes = num_classes
        self.dim = dim
        self.num_patches = self.image_shape[0] * self.image_shape[1] // (self.patch_shape[0] * self.patch_shape[1])
        self.mlp_mixer_block = MLPMixerBlock(patches=self.num_patches, channels=dim)
        self.linear = nn.Linear(in_features=self.in_channels * self.patch_shape[0] * self.patch_shape[1],
                                out_features=self.dim)
        self.norm = nn.LayerNorm(self.dim)
        self.pool = nn.AvgPool1d(kernel_size=self.num_patches)
        self.classifier = nn.Linear(in_features=self.dim, out_features=self.num_classes)

    def forward(self, input_tensor):
        output_tensor = input_tensor.reshape(input_tensor.shape[0], self.in_channels,
                                             self.image_shape[0] // self.patch_shape[0],
                                             self.patch_shape[0], self.image_shape[1] // self.patch_shape[1],
                                             self.patch_shape[1])
        output_tensor = output_tensor.permute(0, 2, 4, 1, 3, 5)
        output_tensor = output_tensor.reshape(output_tensor.shape[0], -1, self.in_channels, self.patch_shape[0],
                                              self.patch_shape[1])
        output_tensor = output_tensor.reshape(output_tensor.shape[0], output_tensor.shape[1], -1)
        output_tensor = self.linear(output_tensor)

        for _ in range(self.depth):
            output_tensor = self.mlp_mixer_block(output_tensor)

        output_tensor = self.norm(output_tensor)
        output_tensor = output_tensor.permute(0, 2, 1)
        output_tensor = self.pool(output_tensor)
        output_tensor = output_tensor.reshape(output_tensor.shape[0], -1)
        output_tensor = self.classifier(output_tensor)

        return output_tensor


if __name__ == '__main__':
    model = MLPMixer(image_shape=(256, 256), patch_shape=(16, 16), dim=512, depth=6, num_classes=120)
    summary(model, input_size=(3, 256, 256))
    pass
