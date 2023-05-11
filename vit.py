import torch
from torch import nn
import math
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import ImageClassifierOutput


class SelfAttentionWithMemory(nn.Module):
    """
    Extension of ViTSelfAttention with memory support.
    """

    def __init__(self, base):
        """
        Construct the class from the given base ViTSelfAttention.
        By default, it won't have any memory input.
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

        # TODO Apply attention masking

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
        for layer in self.vit.encoder.layer:
            layer.attention.attention = SelfAttentionWithMemory(layer.attention.attention)

        # Classifier heads
        self.classifiers = nn.ModuleList([
            base.classifier,
        ])

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
