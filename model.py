import torch
from torch import nn
from math import inf

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


def build_attention_mask(memory_tokens_list: list, extension: bool = False):
    """
    From the given list of memory tokens, construct an attention mask for transformer.
    The boolean "extension" determines whether model extension (True) or model
    concatenatenation (False) is used.
    It's NOT possible to mix model extension and model concatenatenation.

    Masked elements are -inf, rest are 0.
    Mask is supposed to be added to values before softmax.
    """
    class_tokens = len(memory_tokens_list) + 1
    input_tokens = 16 + class_tokens
    total_tokens = input_tokens + sum(memory_tokens_list)

    with torch.no_grad():
        mask = torch.zeros(1, input_tokens, total_tokens)
        # Disable all interactions for all newly added class tokens and memory
        mask[:,:,17:] = -inf
        # Enable interactions for each class token and corresponding memory
        previous_memory = 0
        for i, memory_tokens in enumerate(memory_tokens_list):
            # 16 patches + 1 default class token + index of this
            class_token = 17 + i
            memory_start_index = input_tokens + previous_memory
            memory_end_index = memory_start_index + memory_tokens
            if extension:
                # Class token can interact with itself
                mask[:, class_token:, class_token] = 0.0
                # Class token can interact with its memory tokens
                mask[:, class_token:, memory_start_index:memory_end_index] = 0.0
            else:
                # Class token can interact with itself
                mask[:, class_token, class_token] = 0.0
                # Class token can interact with its memory tokens
                mask[:, class_token, memory_start_index:memory_end_index] = 0.0

            previous_memory += memory_tokens

    return mask


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
    def __init__(self, mha, mask, memory_tokens: int):
        super().__init__()
        self.mha = mha
        with torch.no_grad():
            memory = torch.randn(1, memory_tokens, mha.input_dim) * 0.02
        self.memory = nn.parameter.Parameter(memory, requires_grad=True)
        self.mask = nn.parameter.Parameter(mask, requires_grad=False)

    def forward(self, input):
        # Expand to fill batch size. Expand doesn't copy memory.
        num_inputs = input.size(dim=0)
        expanded_memory = self.memory.expand(num_inputs, -1, -1)
        concatenated = torch.cat((input, expanded_memory), dim=1)

        # Note that memory does not attend to other tokens
        q = self.mha.w_qkv[0](input)
        k = self.mha.w_qkv[1](concatenated)
        v = self.mha.w_qkv[2](concatenated)
        # Compute attention values
        attention = q.matmul(k.transpose(-1, -2)) * self.mha.scaling
        # Attention masking before softmax
        attention = self.mha.softmax(attention + self.mask)
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
        self.attention = MultiHeadAttention(D, heads)
        self.mlp = MLP([D] + mlp_layers + [D])

    def forward(self, input):
        x = self.attention(input) + input
        return x + self.mlp(x)


class TheModel(nn.Module):
    def __init__(self,
                 D=64,
                 heads=8,
                 mlp_layers=[1024],
                 mlp_head_sizes=[512, 10],
                 transformer_count=2,
                 ):
        super().__init__()
        self.prelinear = nn.Linear(49, D)
        # +8 for positional encoding
        self.transformer_dim = D+8
        # Class tokens
        self.class_tokens = nn.ParameterList([
            nn.parameter.Parameter(self.init_cls(), requires_grad=True),
        ])
        # Transformers
        self.transformers = nn.Sequential(*[
            TransformerEncoder(D+8, heads, mlp_layers) for _ in range(transformer_count)
        ])
        # MLP heads
        self.mlp_heads = nn.ModuleList([
            MLP([self.transformer_dim] + mlp_head_sizes),
        ])
        self.memory_tokens_list = []

    def init_cls(self):
        """
        Return a new torch tensor representing a new class token.
        """
        with torch.no_grad():
            class_token = torch.randn(1, 1, self.transformer_dim) * 0.02
        return class_token

    def add_head(self, mlp_head_sizes, memory_tokens: int = 0, extension: bool = False):
        """
        Add a new class token and MLP head with given layer sizes.
        Optionally add memory tokens for the new head.
        The boolean "extension" determines whether model extension (True) or model
        concatenatenation (False) is used. It's only valid if memory_tokens > 0.
        It's NOT possible to mix model extension and model concatenatenation.

        Return a list of newly added parameters.
        """
        device = next(self.parameters()).device

        self.memory_tokens_list.append(memory_tokens)

        self.class_tokens.append(nn.parameter.Parameter(self.init_cls(), requires_grad=True))
        self.mlp_heads.append(MLP([self.transformer_dim] + mlp_head_sizes))

        if memory_tokens > 0:
            mask = build_attention_mask(self.memory_tokens_list, extension)
            for transformer in self.transformers:
                assert type(transformer.attention) == MultiHeadAttention
                attention = transformer.attention
                transformer.attention = MemoryMHA(attention, mask, memory_tokens)

        # Move to device again (new parameters were added)
        self.to(device)

        parameters = [self.class_tokens[-1]] + list(self.mlp_heads[-1].parameters())
        if memory_tokens > 0:
            parameters += [transformer.attention.memory for transformer in self.transformers] # type: ignore
        return parameters

    def add_memory(self, memory_tokens: int):
        """
        Convert MultiHeadAttention blocks to MemoryMHA blocks without adding a new head.

        NOTE: DOESN'T WORK ANYMORE. Adding memory requires adding a new head.
        """
        device = next(self.parameters()).device
        # Add memory
        mask = build_attention_mask(self.memory_tokens_list)
        for transformer in self.transformers:
            assert type(transformer.attention) == MultiHeadAttention
            attention = transformer.attention
            transformer.attention = MemoryMHA(attention, mask, memory_tokens)
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

    def concatenate(self, other):
        """
        Concatenate two separately fine-tuned models.
        Modifies this model in-place.
        """
        device = next(self.parameters()).device
        for class_token in other.class_tokens[1:]:
            self.class_tokens.append(class_token)
        for mlp in other.mlp_heads[1:]:
            self.mlp_heads.append(mlp)
        self.memory_tokens_list += other.memory_tokens_list
        mask = build_attention_mask(self.memory_tokens_list, extension=False)
        for t1, t2 in zip(self.transformers, other.transformers):
            memory = torch.cat((t1.attention.memory.data, t2.attention.memory.data), dim=-2) # type: ignore
            t1.attention.memory = nn.parameter.Parameter(memory, requires_grad=True) # type: ignore
            t1.attention.mask = nn.parameter.Parameter(mask, requires_grad=False)  # type: ignore
        # Move to device again (new parameters were added: mask)
        self.to(device)

    def forward(self, input):
        batch_size = input.size(dim=0)
        x = add_positional_encoding(self.prelinear(patchify(input)))
        # Add class tokens
        class_tokens = [cls.expand(batch_size, -1, -1) for cls in self.class_tokens]
        x = torch.cat([x] + class_tokens, dim=1)
        # Pass through transformers
        x = self.transformers(x)
        # Exract class tokens and pass through MLP heads
        class_tokens = torch.chunk(x.split(16, dim=-2)[1], len(self.mlp_heads), dim=1)
        return [
            mlp_head(cls.reshape(batch_size, self.transformer_dim))
            for cls, mlp_head in zip(class_tokens, self.mlp_heads)
        ]

