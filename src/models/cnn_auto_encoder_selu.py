import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Autoencoder Neural Network
1. N layers of convolution and max pooling up to XX channels
2. M fully connected layers
3. N times upsampling followed by convolution
'''


class ModelAEselu(nn.Module):

    def __init__(self, code_size, image_width, image_height):
        super(ModelAEselu, self).__init__()

        # Network structure parameters
        self.code_size = code_size      # final encoding size
        self.ker = 5                    # kernel size
        self.pad = int(self.ker / 2)    # padding
        self.pool = 2                   # max pooling
        self.cnn_mp_layers = 3          # number of CNN + max pooling layers

        # Compute final 2D matrix size after all convolution + pooling layers
        self.image_width = image_width
        self.image_height = image_height
        self.image_size = image_width * image_height
        self.w = image_width
        self.h = image_height
        for i in range(self.cnn_mp_layers):
            self.w -= 2 * (int(self.ker / 2) - self.pad)
            self.h -= 2 * (int(self.ker / 2) - self.pad)
            self.w /= self.pool
            self.h /= self.pool

        final_cnn_size = self.w * self.h

        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(1, 8, kernel_size=self.ker, padding=self.pad, stride=1)
        self.enc_cnn_2 = nn.Conv2d(8, 16, kernel_size=self.ker, padding=self.pad, stride=1)
        self.enc_cnn_3 = nn.Conv2d(16, 32, kernel_size=self.ker, padding=self.pad, stride=1)
        self.enc_linear_1 = nn.Linear(int(final_cnn_size) * 32, 128)
        self.enc_linear_2 = nn.Linear(128, self.code_size)

        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 128)
        self.dec_linear_2 = nn.Linear(128, int(final_cnn_size) * 32)
        self.dec_cnn_3 = nn.Conv2d(32, 16, kernel_size=self.ker, padding=self.pad, stride=1)
        self.dec_cnn_2 = nn.Conv2d(16, 8, kernel_size=self.ker, padding=self.pad, stride=1)
        self.dec_cnn_1 = nn.Conv2d(8, 1, kernel_size=self.ker, padding=self.pad, stride=1)

        self.upsampling = nn.Upsample(scale_factor=self.pool, mode='bilinear', align_corners=False)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code

    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, self.pool))

        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, self.pool))

        code = self.enc_cnn_3(code)
        code = F.selu(F.max_pool2d(code, self.pool))

        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code

    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.selu(self.dec_linear_2(out))

        out = out.view([code.size(0), -1, self.w, self.h])
        out = F.selu(self.dec_cnn_3(self.upsampling(out)))
        out = F.selu(self.dec_cnn_2(self.upsampling(out)))
        out = self.dec_cnn_1(self.upsampling(out))

        out = out.view([code.size(0), 1, self.image_width, self.image_height])
        return out
