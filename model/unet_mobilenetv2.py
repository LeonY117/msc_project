import torch
import torch.nn as nn

class Unet_MobileNetV2(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # attaches hooks to the encoder
        self.feature_extractor = Encoder_feature_extractor(self.encoder)
        # dummy forward pass
    
        dummy_input = torch.rand((2, 3, 224, 224))
        self.encoder(dummy_input)
        
        self.decoder.load(self.feature_extractor)

    def forward(self, x):
        self.encoder(x)
        out = self.decoder(self.feature_extractor.features)
        return out

# Conv2DTranspose => Batchnorm => Dropout => Relu
class Unet_MobileNetV2_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded = False
        
    def load(self, encoder_features):
        deconv_outputs = [512, 256, 128, 64, 3]
        # print(encoder_features.features)

        in_channels = encoder_features.features['skip_5'].shape[1]

        self.block_1 = self.deconv_block(in_channels, deconv_outputs[0], 3, 2)
        in_channels = deconv_outputs[0] + encoder_features.features['skip_4'].shape[1]

        self.block_2 = self.deconv_block(in_channels, deconv_outputs[1], 3, 2)
        in_channels = deconv_outputs[1] + encoder_features.features['skip_3'].shape[1]

        self.block_3 = self.deconv_block(in_channels, deconv_outputs[2], 3, 2)
        in_channels = deconv_outputs[2] + encoder_features.features['skip_2'].shape[1]

        self.block_4 = self.deconv_block(in_channels, deconv_outputs[3], 3, 2)
        in_channels = deconv_outputs[3] + encoder_features.features['skip_1'].shape[1]

        self.final_layer = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.block_5 = self.deconv_block(in_channels, deconv_outputs[4], 3, 2)

        self.loaded = True

    def deconv_block(self, in_channels, out_channels, kernel_size, stride, dropout=0.5):
        conv_transpose_2d = nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(conv_transpose_2d.weight)
        return nn.Sequential(
            conv_transpose_2d,
            nn.BatchNorm2d(num_features=out_channels),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )

    def forward(self, input):
        
        assert(self.loaded==True)

        x = input['skip_5']
        x = self.block_1(x)
        x = torch.cat((x, input['skip_4']), dim=1)
        x = self.block_2(x)
        x = torch.cat((x, input['skip_3']), dim=1)
        x = self.block_3(x)
        x = torch.cat((x, input['skip_2']), dim=1)
        x = self.block_4(x)
        x = torch.cat((x, input['skip_1']), dim=1)
        x = self.final_layer(x)

        return x

class Encoder_feature_extractor():
    def __init__(self, encoder):
        self.features = {}
        
        layers = {
            "skip_1": encoder.features[2].conv[0][2], 
            "skip_2": encoder.features[4].conv[0][2],
            "skip_3": encoder.features[7].conv[0][2],
            "skip_4": encoder.features[14].conv[0][2],
            "skip_5": encoder.features[17]
        }


        # if we extract based on blocks this is what we'd get instead:
        # layers = {
        #     "block_1": mobileNet.features[1], 
        #     "block_2": mobileNet.features[3],
        #     "block_3": mobileNet.features[6],
        #     "block_4": mobileNet.features[13],
        #     "block_5": mobileNet.features[17]
        # }

        for (name, layer) in layers.items():
            layer.register_forward_hook(self._get_feature(name))
    
    def _get_feature(self, name):
        def hook(model, input, output):
            self.features[name] = output
        return hook