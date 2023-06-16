"""
This extends the ViT in HuggingFace Transformers library and adds support for learnable memory
described in paper "Fine-tuning Image Transformers using Learnable Memory".

See the original source code of ViT:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
"""
import torch
from torch import nn
import math
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import ImageClassifierOutput


def build_attention_mask(patches: int, memory_tokens_list: list, extension: bool = False):
    """
    From the given list of memory tokens, construct an attention mask for self-attention.
    The boolean "extension" determines whether model extension (True) or model
    concatenation (False) is used.
    It's NOT possible to mix model extension and model concatenation.

    Masked elements are -inf, rest are 0.
    Mask is supposed to be added to values before softmax.
    """
    class_tokens = len(memory_tokens_list) + 1
    input_tokens = patches + class_tokens
    total_tokens = input_tokens + sum(memory_tokens_list)

    with torch.no_grad():
        mask = torch.zeros(1, input_tokens, total_tokens)
        # Disable all interactions for all newly added class tokens and memory
        mask[:, :, (patches + 1):] = -math.inf
        # Enable interactions for each class token and corresponding memory
        previous_memory = 0
        for i, memory_tokens in enumerate(memory_tokens_list):
            # 16 patches + 1 default class token + index of this
            class_token = (patches + 1) + i
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


class SelfAttentionWithMemory(nn.Module):
    """
    Extension of ViTSelfAttention with memory support.
    """

    def __init__(self, base, patch_count: int):
        """
        Construct the class from the given base ViTSelfAttention and patch count.
        By default, it won't have any memory input.

        The layer needs to know the patch count to build the attention mask.
        """
        super().__init__()
        self.num_attention_heads = base.num_attention_heads
        self.attention_head_size = base.attention_head_size
        self.all_head_size = base.all_head_size
        self.attention_scaling = 1.0 / (self.attention_head_size)**0.5

        self.query = base.query
        self.key = base.key
        self.value = base.value

        self.dropout = base.dropout

        # No memory by default.
        self.memory_tokens = nn.ParameterList([])

        # Attention mask
        self.patch_count = patch_count
        attention_mask = build_attention_mask(patch_count, [])
        device = next(self.parameters()).device
        self.register_buffer("attention_mask", attention_mask.to(device))

    def update_attention_mask(self, extension: bool = False):
        """
        Force update the attention mask.
        This will be done automatically when new memory is added.
        """
        device = next(self.parameters()).device
        self.attention_mask = build_attention_mask(
            self.patch_count,
            [memory.size(dim=1) for memory in self.memory_tokens],
            extension,
        ).to(device)

    def add_memory(self, tokens: int, extension: bool = False, std: float = 0.02):
        """
        Add a new series of memory tokens to this self-attention block.
        - tokens: Number of new memory tokens.
        - extension: If true, the attention masking will use the model extension strategy
                     instead of model concatenation.
        - std: Standard deviation of the normal distribution that is used to initialize
               the memory. 0.02 by default. See the end of page 4 in "Fine-tuning Image
               Transformers using Learnable Memory".

        Returns the newly added memory parameters.
        """
        device = next(self.parameters()).device
        with torch.no_grad():
            # See the end of page 
            new_memory = torch.randn(1, tokens, self.query.in_features, device=device) * std
        new_memory = nn.parameter.Parameter(new_memory, requires_grad=True)
        self.memory_tokens.append(new_memory)
        self.update_attention_mask(extension)
        return new_memory

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Same as ViTSelfAttention.
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        """
        Apply self-attention with memory and masking.
        - hidden_states is a tensor of size (Batch size, tokens, hidden dim)
        """
        batch_size = hidden_states.size(dim=0)
        with_memory = torch.cat([hidden_states] + [
            token.expand(batch_size, -1, -1) for token in self.memory_tokens
        ], dim=1)

        # Compute query, key, and values
        key_layer = self.transpose_for_scores(self.key(with_memory))
        value_layer = self.transpose_for_scores(self.value(with_memory))
        # Note that memory doesn't attend to other tokens, they cannot query.
        # Hence, we don't pass memory tokens through the query layer.
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        # These 3 are of size (Batch size, attention heads, tokens, attention head size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * self.attention_scaling
        # attention_scores size: (Batch size, attention heads, tokens, tokens including memory)

        # Apply attention masking
        attention_scores += self.attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # We apply our own mask but keep this for backwards compatibility.
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class MultiEmbeddings(nn.Module):
    """
    Extends ViTEmbeddings to support multiple CLS embeddings.
    NOTE: Unlike ViTEmbeddings, adds class tokens to the end.
    """

    def __init__(self, base) -> None:
        """
        - base: ViTEmbeddings
        """
        super().__init__()
        self.cls_tokens = nn.ParameterList([base.cls_token])
        self.mask_token = base.mask_token
        self.patch_embeddings = base.patch_embeddings
        self.position_embeddings = base.position_embeddings
        self.dropout = base.dropout
        self.config = base.config

    def add_cls_token(self):
        """
        Append a new class token and return the trainable parameter.
        """
        device = self.cls_tokens[0].device
        cls_token = nn.parameter.Parameter(torch.randn_like(self.cls_tokens[0], device=device))
        self.cls_tokens.append(cls_token)
        return cls_token

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method is the same as in ViTEmbeddings.

        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        """
        Given pixel_values of size (), return an embedding:
        - First 49 tokens (output[:, :49, :]) are the patches with position embedding.
        - Remaining tokens are the class tokens.
            - The same position embedding is added to each.
        """
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # Compute encoding to each token
        if interpolate_pos_encoding:
            position_embeddings = self.interpolate_pos_encoding(embeddings, height, width)
        else:
            position_embeddings = self.position_embeddings
        # Note that we have moved class tokens to the end, and we may have many of them
        class_pos = position_embeddings[:, 0:1, :]
        other_pos = position_embeddings[:, 1:, :]

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = [
            # Add the positional encoding for each class token
            (cls_token + class_pos).expand(batch_size, -1, -1)
            for cls_token in self.cls_tokens
        ]
        embeddings = torch.cat([embeddings + other_pos] + cls_tokens, dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings


class MemoryCapableViT(nn.Module):
    """
    Extends ViTForImageClassification class with fine-tunable memory capability.

    By default, it doesn't add a new classification head or memory.
    It's expected to be equivalent to underlying model until new classification head and the
    corresponding memory is added.
    """

    def __init__(self, base) -> None:
        """
        Initialize a MemoryCapableViT from ViTForImageClassification.
        """
        super().__init__()

        self.num_labels = base.num_labels
        self.vit = base.vit

        # Upgrade relevant layers
        self.vit.embeddings = MultiEmbeddings(self.vit.embeddings)
        # Minus one because one of them is for class token
        patch_count = self.vit.embeddings.position_embeddings.size(dim=1) - 1
        for layer in self.vit.encoder.layer:
            layer.attention.attention = SelfAttentionWithMemory(
                layer.attention.attention,
                patch_count,
            )

        # Classifier heads
        self.classifiers = nn.ModuleList([base.classifier])

    def add_head(self, memory_tokens: int, num_classes: int,
                 extension: bool = False, std: float = 0.02):
        """
        Add a new series of memory tokens to this self-attention block.
        - memory_tokens: Number of new memory tokens.
        - num_classes: Number of classes for the new classifier.
        - extension: If true, the attention masking will use the model extension strategy
                     instead of model concatenation.
        - std: Standard deviation of the normal distribution that is used to initialize
               the memory. 0.02 by default. See the end of page 4 in "Fine-tuning Image
               Transformers using Learnable Memory".

        Returns a list of newly added parameters.
        """
        # Add new class token to embeddings
        cls_token = self.vit.embeddings.add_cls_token()
        # Add new classifier head
        first = self.classifiers[0]
        device = next(first.parameters()).device
        classifier = nn.Linear(first.in_features, num_classes, device=device) # type: ignore
        self.classifiers.append(classifier)
        # Add memory for each layer
        memory = [layer.attention.attention.add_memory(memory_tokens, extension, std)
                  for layer in self.vit.encoder.layer]
        # Return all newly added parameters
        return [cls_token] + memory + list(classifier.parameters())

    def concatenate(self, other: "MemoryCapableViT"):
        """
        Concatenate two separately fine-tuned models.
        Modifies this model in-place.
        The other model should not be used after this operation. Because it will be sharing
        parameters with this one.

        NOTE: Both models should be fine-tuned from the same model with the model
        concatenation strategy (default).
        """
        device = next(self.parameters()).device
        # Note that the first class tokens and classifiers are the same.
        self.vit.embeddings.cls_tokens.extend(other.vit.embeddings.cls_tokens[1:])
        for layer1, layer2 in zip(self.vit.encoder.layer, other.vit.encoder.layer):
            a1 = layer1.attention.attention
            a2 = layer2.attention.attention
            a1.memory_tokens.extend(a2.memory_tokens)
            a1.update_attention_mask(False)
        self.classifiers.extend(other.classifiers[1:])  # type: ignore
        # Move to the device again to ensure that the new parameters are in the same device.
        self.to(device)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        _: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> List[ImageClassifierOutput]:
        """
        Returns an ImageClassifierOutput for each classification head.
        3rd argument (labels) is ignored for simplicity, but kept for API consistency.
        """
        return_dict = return_dict if return_dict is not None else True

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # Last len(self.classifiers) tokens are class tokens.
        output_tokens = outputs[0][:, -len(self.classifiers):, :]

        # Pass class tokens through respective classifier heads.
        logits = [classifier(output_tokens[:, i, :])
                  for i, classifier in enumerate(self.classifiers)]

        return [ImageClassifierOutput(
                    loss=None,
                    logits=l,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
                for l in logits]
