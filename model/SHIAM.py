import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU())
        self.aspp_block_1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        self.aspp_block_2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.aspp_block_3 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, dilation=1)
        self.aspp_block_4 = nn.Conv2d(in_channels, out_channels, 3, 1, 2, dilation=2)
        self.aspp_block_5 = nn.Conv2d(in_channels, out_channels, 3, 1, 3, dilation=3)
        self.net = nn.Sequential(nn.BatchNorm2d(5 * out_channels), nn.ReLU(), nn.Conv2d(5 * out_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        x = self.bn_relu(x)
        output = self.net(
            torch.cat(
                [
                    nn.Upsample(size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)(self.aspp_block_1(x)),
                    self.aspp_block_2(x),
                    self.aspp_block_3(x),
                    self.aspp_block_4(x),
                    self.aspp_block_5(x),
                ],
                dim=1,
            )
        )
        return output

class DSPUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetVariant, self).__init__()

        self.encoder1 = ResidualBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = nn.Sequential(
            ResidualBlock(256, 512),
            ASPP(512, 512)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder1 = ResidualBlock(512, 256)
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(256, 128)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(128, 64)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = ResidualBlock(64, out_channels)

        # 1x1conv&softmax
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        d1 = self.decoder1(self.pool4(e4))
        d1_upsampled = self.upsample1(d1)
        d2 = self.decoder2(torch.cat([e3, d1_upsampled], 1))
        d2_upsampled = self.upsample2(d2)
        d3 = self.decoder3(torch.cat([e2, d2_upsampled], 1))
        d3_upsampled = self.upsample3(d3)
        d4 = self.decoder4(torch.cat([e1, d3_upsampled], 1))

        output = self.final_conv(d4)
        output = self.softmax(output)

        return output

class MLSAB(nn.Module):
    def __init__(self):
        super(MLCAB, self).__init__()

    def forward(self, s1, s2, s3):
        # three branches
        att_s1 = self.process_branch(s1)
        att_s2 = self.process_branch(s2)
        att_s3 = self.process_branch(s3)
        att_s1_upsampled = F.interpolate(att_s1, size=s3.size()[2:], mode='bilinear', align_corners=False)
        att_s2_upsampled = F.interpolate(att_s2, size=s3.size()[2:], mode='bilinear', align_corners=False)
        att_total = att_s1_upsampled + att_s2_upsampled + att_s3

        output_s = s3 * att_total

        return output_s

    def process_branch(self, x):

        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        pool_avg_max = (max_pool + avg_pool) / 2.0
        att_branch = torch.transpose(pool_avg_max, 1, 2) * pool_avg_max

        return att_branch

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "Embedding dimension needs to be divisible by the number of heads."

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
   
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(query.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(key.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(value.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attn_weights, value)

 
        concat_values = attended_values.transpose(1, 2).contiguous().view(query.size(0), -1, self.embed_dim)
        output = self.fc_out(concat_values)

        return output

# define L-RPB
class LRPB(nn.Module):
    def __init__(self, in_channels):
        super(LRPB, self).__init__()

        self.conv_transpose = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels)
        self.conv_reduce = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
  
        similarity_matrix = torch.transpose(x, 1, 2) * x
        alpha = torch.min(torch.max(similarity_matrix, dim=1)[0], dim=1)[0]
        S_max = torch.max(similarity_matrix, dim=1)[0]
        S_min = torch.min(similarity_matrix, dim=1)[0]

        J1 = self.conv_reduce(S_max - alpha.unsqueeze(1))
        J2 = self.conv_reduce(S_min - alpha.unsqueeze(1))

        output = J1 * x + J2* x + x

        return output

# define GCIAM
class GCIAM(nn.Module):
    def __init__(self, in_channels, heads, d_model):
        super(GCIAM, self).__init__()

        self.resample1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.resample2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.resample3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.multihead_attention = nn.MultiheadAttention(d_model, heads)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.lrpb = LRPB(in_channels)

    def forward(self, x1, x2, x3):

        x1_resampled = self.resample1(x1)
        x2_resampled = self.resample2(x2)
        x3_resampled = self.resample3(x3)

        # shape as W×H×C
        x1_reshaped = x1_resampled.view(x1_resampled.size(0), -1, x1_resampled.size(-1))
        x2_reshaped = x2_resampled.view(x2_resampled.size(0), -1, x2_resampled.size(-1))
        x3_reshaped = x3_resampled.view(x3_resampled.size(0), -1, x3_resampled.size(-1))

        input_feature = torch.cat([x1_reshaped, x2_reshaped, x3_reshaped], dim=1)


        attention_output, _ = self.MultiheadAttention(input_feature, input_feature, input_feature)
        normalized_output = self.layer_norm(attention_output + input_feature)

        output = self.LRPB(normalized_output)

        return output

class Entropybranch(nn.Module):
    def __init__(self):
        super(Entropybranch, self).__init__()

        BF3 = nn.Conv2d(in_channels, 96, kernel_size=(11, 11), stride=(4, 4), padding='same')
        self.Y = nn.Sequential(
            BF3,
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding='valid'),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding='valid')
        )

        self.conv_c1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding='same')
        self.conv_c2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding='same')
        self.conv_c3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding='same')
        self.conv_c4 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding='same')

        self.concat = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )

        self.global_max_pooling = nn.AdaptiveMaxPool2d(1)


    def forward(self, conv5, BF3):
        Y = self.Y(BF3)
        conv_c1 = self.conv_c1(conv5)
        conv_c2 = self.conv_c2(conv_c1)
        conv_c3 = self.conv_c3(conv_c2)
        conv_c4 = self.conv_c4(conv_c3)

        X = torch.cat([conv5, conv_c4, Y], dim=1)
        X = self.concat(X)

        X = self.global_max_pooling(X)
        X = X.view(X.size(0), -1)
        X = F.softmax(self.fc(X), dim=1)

        return X


class SHIAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SLIAM, self).__init__()
        size, in_channels, out_channels = (config['size'], config['in_channels'], config['out_channels'])

        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.unet_variant = DSPUNet(in_channels, out_channels)
        self.mlcab = MLSAB()
        self.gciam = GCIAM(in_channels, heads=8, d_model=512)

        self.entropy_branch = Entropybranch()

        self.fc = nn.Linear(3 * out_channels, out_channels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
  
        roi_mask = self.unet_variant(x)
        bf1 = self.double_conv(roi_mask) 
        

        e1 = self.unet_variant.encoder1(x)
        e2 = self.unet_variant.encoder2(self.unet_variant.pool1(e1))
        e3 = self.unet_variant.encoder3(self.unet_variant.pool2(e2))
        e4 = self.unet_variant.encoder4(self.unet_variant.pool3(e3))

        d1 = self.unet_variant.decoder1(self.unet_variant.pool4(e4))
        d1_upsampled = self.unet_variant.upsample1(d1)
        d2 = self.unet_variant.decoder2(torch.cat([e3, d1_upsampled], 1))
        d2_upsampled = self.unet_variant.upsample2(d2)
        d3 = self.unet_variant.decoder3(torch.cat([e2, d2_upsampled], 1))
        d3_upsampled = self.unet_variant.upsample3(d3)
        d4 = self.unet_variant.decoder4(torch.cat([e1, d3_upsampled], 1))

        bf2 = self.mlacab(d1, d2, d3)
        bf2 = self.gciam(d1, d2, d3)
        bf3 = self.entropy_branch(x, bf3)

        concatenated_features = torch.cat([bf1, bf2, bf3], dim=1)
        output = self.fc(concatenated_features)

        output = self.softmax(output)

        return output


#test

if __name__ == '__main__':
    config = {
        'size': 150,
        'in_channels': 1,
        'out_channels': 2,
    }
    net = SHIAM(config)
    #print(net(torch.randn(8, 1, 150, 150)).shape)
    print(torchsummary.summary(net, (1, 150, 150), batch_size=-1, device='cuda:0'))





