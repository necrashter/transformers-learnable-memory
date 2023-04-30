import torch
from torch import nn

def patchify(x):
    """
    x: input torch array of size Nx1x28x28
    output: torch array of size Nx16x49
    """
    def get_patch(i):
        xpos = (i % 4) * 7
        ypos = (i // 4) * 7
        return x[:,:,xpos:xpos+7,ypos:ypos+7].reshape(x.shape[0], 1, 49)
    return torch.concat([get_patch(i) for i in range(16)], 1)


def add_positional_encoding(x):
    """
    x: input torch array of size Nx16xD
    output: torch array of size Nx16x(8+D)
    """
    def process_patch(i):
        xpos = (i % 4)
        ypos = (i // 4)
        posarr = torch.zeros([x.shape[0], 1, 8], dtype=x.dtype, device=x.device)
        posarr[:,0,xpos] = 1
        posarr[:,0,ypos+4] = 1
        payload = x[:,i].reshape(x.shape[0], 1, x.shape[2])
        return torch.concat((posarr, payload), 2)
    return torch.concat([process_patch(i) for i in range(16)], 1)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, heads):
        """
        Implemented as shown in Appendix A of the given paper.
        Multi-head attention is applied and their concatenated outputs are
        projected with a linear layer.
        Unlike the original paper, bias was used in this final linear layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.concat_dim = input_dim * heads
        self.scaling = (1/self.input_dim) ** 0.5
        # Weights for query, key, and value
        self.w_qkv = nn.ModuleList([
            nn.Linear(input_dim, self.concat_dim, bias = False),
            nn.Linear(input_dim, self.concat_dim, bias = False),
            nn.Linear(input_dim, self.concat_dim, bias = False),
        ])
        self.softmax = nn.Softmax(dim = -1)
        # This is where I decided to use bias
        self.final = nn.Linear(self.concat_dim, self.input_dim)

    def forward(self, input):
        q, k, v = [w(input) for w in self.w_qkv]
        # Attention values for weighted average
        attention = self.softmax(q.matmul(k.transpose(-1, -2)) * self.scaling)
        # Concatenated output with attention applied
        head_out = torch.matmul(attention, v)
        # Final projection
        return self.final(head_out)


class MemoryMHA(nn.Module):
    def __init__(self, mha, memory_tokens: int):
        super().__init__()
        self.mha = mha
        with torch.no_grad():
            memory = torch.randn(memory_tokens, mha.input_dim) * 0.02
        self.memory = nn.parameter.Parameter(memory, requires_grad=True)

        # Attention masking
        with torch.no_grad():
            # 16 patches, 1 CLS, memory_tokens
            tokens = 17 + memory_tokens
            mask = torch.ones(1, tokens, tokens)
            # Memory does not attend to other tokens
            mask[:,17:,:] = 0.0
            self.mask = nn.parameter.Parameter(mask, requires_grad=False)

    def forward(self, input):
        # Expand to fill batch size. Expand doesn't copy memory.
        num_inputs = input.size(dim=0)
        expanded_memory = self.memory.expand(num_inputs, -1, -1)
        concatenated = torch.cat((input, expanded_memory), dim=1)

        q, k, v = [w(concatenated) for w in self.mha.w_qkv]
        # Attention values for weighted average
        attention = self.mha.softmax(q.matmul(k.transpose(-1, -2)) * self.mha.scaling)
        # Attention masking
        attention = attention * self.mask
        # Concatenated output with attention applied
        head_out = torch.matmul(attention, v)
        # Final projection
        attended = self.mha.final(head_out)

        return attended.split(input.size(dim=-2), dim=-2)[0]


class MLP(nn.Module):
    def __init__(self, layer_sizes):
        """
        Linear layers with GELU activation function inbetween.
        """
        super().__init__()
        layers = sum([
            [nn.Linear(a, b), nn.GELU()]
            for a, b in zip(layer_sizes, layer_sizes[1:-1])
            ], [])
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.layers(input)


class TransformerEncoder(nn.Module):
    def __init__(self, D, heads, mlp_layers):
        super().__init__()
        self.norm0 = nn.LayerNorm(D)
        self.attention = MultiHeadAttention(D, heads)
        self.norm1 = nn.LayerNorm(D)
        self.mlp = MLP([D] + mlp_layers + [D])

    def forward(self, input):
        x = self.attention(self.norm0(input)) + input
        return x + self.mlp(self.norm1(x))


class TheModel(nn.Module):
    def __init__(self,
                 D=64,
                 heads=8,
                 mlp_layers=[1024],
                 mlp_head_sizes=[512],
                 transformer_count=2,
                 ):
        super().__init__()
        self.prelinear = nn.Linear(49, D)
        # Class token, +8 for positional encoding
        # +8 for positional encoding
        with torch.no_grad():
            class_token = torch.randn(1, 1, D+8) * 0.02
        self.class_token = nn.parameter.Parameter(class_token, requires_grad=True)
        # Transformers
        self.transformers = nn.Sequential(*[
            TransformerEncoder(D+8, heads, mlp_layers) for _ in range(transformer_count)
        ])
        # MLP
        self.mlp_head_input_size = D+8
        self.mlp_head = MLP([self.mlp_head_input_size] + mlp_head_sizes + [10])

    def add_memory(self, memory_tokens: int):
        """
        Convert MultiHeadAttention blocks to MemoryMHA blocks.
        """
        # Current device
        device = next(self.parameters()).device
        # Add memory
        for transformer in self.transformers:
            assert type(transformer.attention) == MultiHeadAttention
            attention = transformer.attention
            transformer.attention = MemoryMHA(attention, memory_tokens)  # type: ignore
        # Move to device again (new parameters were added)
        self.to(device)

    def memory_parameters(self):
        """
        Return a list of learnable memory parameters.
        """
        parameters = []
        for transformer in self.transformers:
            assert type(transformer.attention) == MemoryMHA
            parameters.append(transformer.attention.memory)  # type: ignore
        return parameters

    def forward(self, input):
        batch_size = input.size(dim=0)
        x = add_positional_encoding(self.prelinear(patchify(input)))
        # Add class token
        cls = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((x, cls), dim=1)
        # Pass through transformer
        x = self.transformers(x)
        # Exract class token and pass through MLP head
        cls = x.split(16, dim=-2)[1]
        cls = cls.reshape(batch_size, self.mlp_head_input_size)
        return self.mlp_head(cls)

