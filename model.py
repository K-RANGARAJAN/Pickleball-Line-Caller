import torch
import torch.nn as nn

class TrackNetV2(nn.Module):
    def __init__(self, input_channels=9, output_channels=1):
        super(TrackNetV2, self).__init__()
        
        # --- ENCODER ---
        self.conv1 = self._conv_block(input_channels, 64)
        self.conv2 = self._conv_block(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = self._conv_block(64, 128)
        self.conv4 = self._conv_block(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = self._conv_block(128, 256)
        self.conv6 = self._conv_block(256, 256)
        self.conv7 = self._conv_block(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = self._conv_block(256, 512)
        self.conv9 = self._conv_block(512, 512)
        self.conv10 = self._conv_block(512, 512)

        # --- DECODER ---
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv11 = self._conv_block(512, 256)
        self.conv12 = self._conv_block(256, 256)
        self.conv13 = self._conv_block(256, 256)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv14 = self._conv_block(256, 128)
        self.conv15 = self._conv_block(128, 128)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv16 = self._conv_block(128, 64)
        self.conv17 = self._conv_block(64, 64)

        # Output Layer (Sigmoid activations mapping probabilities 0.0 to 1.0)
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder passes
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        # Decoder passes
        x = self.upsample1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.upsample2(x)
        x = self.conv14(x)
        x = self.conv15(x)

        x = self.upsample3(x)
        x = self.conv16(x)
        x = self.conv17(x)

        out = self.sigmoid(self.out_conv(x))
        return out
    
    
if __name__ == "__main__":
    # Test block to verify the network works locally
    print("Initializing TrackNetV2...")
    
    # Check for Apple Silicon GPU acceleration (MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Instantiate model
    net = TrackNetV2().to(device)
    
    # Simulate an input: 1 batch, 9 channels (3 frames x 3 channels), 288 height, 512 width
    dummy_input = torch.randn(1, 9, 288, 512).to(device)
    
    print("Running dummy forward pass...")
    output = net(dummy_input)
    
    print("\n--- Success! ---")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Heatmap Shape: {output.shape} (Should be [1, 1, 288, 512])")