import torch
from torch import nn

from models.base import BaseNet


class Attention(nn.Module):
    def __init__(self, express_size):
        super(Attention, self).__init__()
        self.express_size = express_size
        self.to_K = nn.Linear(in_features=express_size, out_features=express_size)
        self.query = nn.Linear(in_features=express_size, out_features=1, bias=False)
        torch.nn.init.xavier_uniform_(self.to_K.weight)
        torch.nn.init.xavier_uniform_(self.query.weight)

    def forward(self, value, mask=None):
        texts_K = torch.tanh(self.to_K(value))
        scores = self.query(texts_K)

        if mask is not None:
            max_vals = torch.max(scores, dim=1, keepdim=True).values.detach()
            exps = torch.exp(scores.float() - max_vals.float()) * mask
            sum_exps = exps.sum(dim=1, keepdim=True)
            temp_scores = exps / (sum_exps + 1e-8)
        else:
            temp_scores = nn.functional.softmax(scores, dim=1, dtype=torch.float32)

        return torch.bmm(temp_scores.permute((0, 2, 1)), value).squeeze(dim=1)


class Step_Attention(nn.Module):
    def __init__(self, express_size):
        super(Step_Attention, self).__init__()
        self.express_size = express_size
        self.to_K = nn.Linear(in_features=express_size, out_features=express_size)
        self.query = nn.Linear(in_features=express_size, out_features=1, bias=False)
        torch.nn.init.xavier_uniform_(self.to_K.weight)
        torch.nn.init.xavier_uniform_(self.query.weight)

    def forward(self, value):
        texts_K = torch.tanh(self.to_K(value))
        scores = self.query(texts_K)
        ans = []
        for i in range(1, value.shape[1] + 1):
            texts_feature = value[:, 0:i, :]
            temp_scores = torch.softmax(scores[:, 0:i, :], dim=1)
            ans.append(torch.bmm(temp_scores.permute((0, 2, 1)), texts_feature))
        ans = torch.cat(ans, dim=1)
        return ans

class LSTM_han(BaseNet):
    def __init__(self, d=256, t_downsample=4):
        super().__init__()
        # 视频特征卷积降采样
        self.v_downsample = nn.Sequential(
            nn.Conv1d(25 + 136, d, kernel_size=16, stride=t_downsample, padding=8)
        )
        # 视频编码器
        self.v_encoder = nn.LSTM(
            input_size=d,
            hidden_size=d // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.v_att = Attention(express_size=d)
        # 用户编码器
        self.u_encoder = nn.LSTM(
            input_size=d,
            hidden_size=d // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.u_att = Attention(express_size=d)
        self.z_dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(nn.Linear(d, 1))

    def feature_extractor(self, input):
        features, user_lengths, _ = input
        batch_size = features.size(0)
        # reshape到Conv1d输入: [batch, channels, seq_len]
        features = features.permute(0, 3, 1, 2).contiguous().view(batch_size, features.size(3), -1)

        # 视频特征编码
        processed_features = self.v_downsample(features).transpose(1, 2)
        processed_features, _ = self.v_encoder(processed_features)
        processed_features = self.v_att(processed_features)

        user_feats = []
        for i in range(batch_size):
            single_feat = processed_features[i]

            # 确保 LSTM 输入维度为 (batch, seq_len, feature_dim)
            if single_feat.dim() == 2:
                single_feat = single_feat.unsqueeze(0)
            elif single_feat.dim() == 1:
                single_feat = single_feat.unsqueeze(0).unsqueeze(0)
            elif single_feat.dim() != 3:
                raise ValueError(f"Unexpected single_feat shape: {single_feat.shape}")

            z, _ = self.u_encoder(single_feat)
            z = self.u_att(z)
            user_feats.append(z)

        z = torch.cat(user_feats, dim=0)
        return torch.relu(z)

    def classifier(self, x):
        return self.fc(x)
        
class LSTM_attn(BaseNet):
    def __init__(self, d=256, l=6, t_downsample=4):
        super().__init__()
        self.v_downsample = nn.Sequential(
            nn.Conv1d(136 + 25, d, kernel_size=16, stride=t_downsample, padding=8)
        )
        self.v_encoder = nn.LSTM(
            input_size=d,
            hidden_size=int(d / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.v_att = Attention(express_size=d)
        self.fc = nn.Sequential(nn.Linear(d, 1))

    def feature_extractor(self, input):
        features, user_lengths, _ = input
        start, user_feats = 0, []

        for length in user_lengths:
            u_feats = features[start:start + length, :]
            # reshape到Conv1d输入: [1, channels, seq_len]
            batch_size = u_feats.size(0)
            u_feats = u_feats.permute(0, 2, 1).contiguous()
            u_feats = u_feats.view(1, u_feats.size(1), -1)

            processed_features = self.v_downsample(u_feats).transpose(1, 2)
            z, _ = self.v_encoder(processed_features)
            z = self.v_att(z)
            user_feats.append(z)
            start = start + length

        z = torch.cat(user_feats, dim=0)
        return torch.relu(z)

    def classifier(self, x):
        return self.fc(x)
