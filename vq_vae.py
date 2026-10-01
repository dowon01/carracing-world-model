import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. VectorQuantizer layer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 단어장 정의 (codebook)
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: (B, C, H, W) -> (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # 평탄화
        flat_input = inputs.view(-1, self.embedding_dim)

        # L2 거리 계산
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)) + torch.sum(self.embeddings.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.embeddings.weight.t())

        # 가장 가까운 인덱스 선택
        encoding_indices = torch.min(distances, dim=1)[1].unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 양자화 (Quantize)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # 손실 함수 계산 (Vector Quantization Loss)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 역전파를 위한 트릭 (Straight-through estimator)
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices

# 2. Residual Stack (VQ-VAE의 안정적 학습을 위함)
class ResidualLayer(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualLayer, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            # 3x3 : 주변 정보를 훑으면서 특징 추출
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            # 1x1 : 채널 수를 다시 원래대로 복구
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)
        )
    def forward(self, x):
        # 핵심 : 입력값 x를 보존했다가 나중에 변화량 _block(x)와 더함
        return x + self._block(x)
    
class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._layers = nn.ModuleList([ResidualLayer(in_channels, num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)])
    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)

# 3. Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens//2, kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2, out_channels=num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = F.relu(self._conv_1(inputs))
        x = F.relu(self._conv_2(x))
        x = self._conv_3(x)
        x = self._residual_stack(x)
        return x

# 4. Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens//2, kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        x = self._conv_trans_2(x)
        return x

# 5. VQ-VAE Wrapper
class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, dead_code_threshold=1.0):
        super(VQVAE, self).__init__()
        self._encoder = Encoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        self._vq_vae = EMAVectorQuantizer(num_embeddings, embedding_dim, commitment_cost, dead_code_threshold=dead_code_threshold)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity

# 6. EMA VQ-VAE Wrapper
class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5, dead_code_threshold=1.0):
        super(EMAVectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        # EMA 사용량이 이 값보다 작은 토큰을 죽은 토큰으로 보고 교체 (0이면 교체 안 함)
        self.dead_code_threshold = dead_code_threshold

        # 1. 단어장(Embedding) 선언 및 초기화
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # 초기 분산을 키워 토큰들이 멀리 떨어지게 하여 죽은 토큰 문제 완화
        self.embedding.weight.data.normal_(std=0.5) 

        # 2. EMA 버퍼 등록
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.empty(num_embeddings, embedding_dim))
        
        self._ema_w.normal_(std=0.5) 

    def forward(self, inputs):
        # [B, C, H, W] -> [B, H, W, C]
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 거리 계산 및 가장 가까운 인덱스 추출
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.min(distances, dim=1)[1].unsqueeze(1)
        
        # One-hot encodings
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
            n = torch.sum(self._ema_cluster_size)
            self._ema_cluster_size = ((self._ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w.data.copy_(self._ema_w * self.decay + (1 - self.decay) * dw)
            
            # 사용량이 너무 적은(죽은) 토큰을 찾아 현재 입력값 중 하나로 바꿈
            dead_indices = (self._ema_cluster_size < self.dead_code_threshold).nonzero(as_tuple=True)[0]
            
            if len(dead_indices) > 0:
                # 현재 배치에서 무작위로 샘플을 뽑아 죽은 토큰 위치에 주입
                random_indices = torch.randint(0, flat_input.size(0), (len(dead_indices),), device=inputs.device)
                random_samples = flat_input[random_indices]
                
                self.embedding.weight.data[dead_indices] = random_samples
                self._ema_w.data[dead_indices] = random_samples * self._ema_cluster_size[dead_indices].unsqueeze(1)
            
            # 실제 임베딩 가중치 업데이트
            self.embedding.weight.data.copy_(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Quantize & Loss 계산
        quantized = self.embedding(encoding_indices).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Perplexity 계산
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices
    
    def get_codebook_entry(self, indices, shape):
        # 1. 인덱스에 해당하는 임베딩 벡터를 추출
        # indices: (batch, 256) -> quantized: (batch, 256, embedding_dim)
        quantized = self.embedding(indices)

        # 2. (batch, H, W, C) 모양으로 다시 복구 (1, 16, 16, embedding_dim=64)
        quantized = quantized.view(shape)

        # 3. 디코더가 읽을 수 있도록 채널을 앞으로
        # (batch, H, W, C) -> (batch, C, H, W)
        return quantized.permute(0, 3, 1, 2).contiguous()