from torch import nn
from torchsummary import summary


from module import MLPMixerBlock


class MLPMixer(nn.Module):
    def __init__(self, image_shape, in_channels, patch_shape, depth, dim, num_classes):
        super(MLPMixer, self).__init__()
        self.image_shape = image_shape
        self.in_channels = in_channels
        self.patch_shape = patch_shape
        self.depth = depth
        self.num_classes = num_classes
        self.dim = dim
        self.num_patches = self.image_shape[0] * self.image_shape[1] // (self.patch_shape[0] * self.patch_shape[1])
        self.mlp_mixer_block = MLPMixerBlock(patches=self.num_patches, channels=dim)

    def forward(self, input_tensor):

        input_tensor = input_tensor.reshape(input_tensor.shape[0], self.in_channels,
                                            self.image_shape[0] // self.patch_shape[0],
                                            self.patch_shape[0], self.image_shape[1] // self.patch_shape[1],
                                            self.patch_shape[1])
        input_tensor = input_tensor.permute(0, 2, 4, 1, 3, 5)
        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1, self.in_channels, self.patch_shape[0],
                                            self.patch_shape[1])
        input_tensor = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        input_tensor = nn.Linear(in_features=input_tensor.shape[-1], out_features=self.dim)(input_tensor)

        for _ in range(self.depth):
            input_tensor = self.mlp_mixer_block(input_tensor)

        output_tensor = nn.LayerNorm(self.dim)(input_tensor)
        output_tensor = output_tensor.permute(0, 2, 1)
        output_tensor = nn.AvgPool1d(kernel_size=self.num_patches)(output_tensor)
        output_tensor = output_tensor.reshape(output_tensor.shape[0], -1)
        output_tensor = nn.Linear(in_features=self.dim, out_features=self.num_classes)(output_tensor)

        return output_tensor


if __name__ == '__main__':
    model = MLPMixer(image_shape=(256, 256), in_channels=3, patch_shape=(16, 16), dim=512, depth=6, num_classes=120)
    summary(model, input_size=(3, 256, 256))
    pass
