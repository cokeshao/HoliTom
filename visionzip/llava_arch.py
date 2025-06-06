from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random

class LlavaMetaForCausalLM_visionzip(ABC):

    def encode_images(self, images):
        image_features, _ = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_images_multi(self, images):
        image_features, attn_weights, metric, images_dtype = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features, attn_weights, metric, images_dtype
    
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
            
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features, attn_weights, metric, images_dtype = self.encode_images_multi(concat_images)            
            import os
            spatial_tokens = int(os.environ.get("SPATIAL_TOKENS", 50))
            dominant_ratio = float(os.environ.get("DOMINANT_RATIO", 54/64))
            dominant_num = int(spatial_tokens * dominant_ratio)
            contextual_num = spatial_tokens - dominant_num
            rank0_print(f"dominant_num: {dominant_num}, contextual_num: {contextual_num}")
            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    # [modify]
                    # image_features.append(self.get_2dPool(image_feat))
                    # image_feat: (batch_size, seq_len, embed_dim)
                    # attn_weights: (batch_size, seq_len)
                    # metric: (batch_size, seq_len, head_dim)
                    pooled_image_feat = self.get_2dPool(image_feat) # (batch_size, seq_len', embed_dim)
                    attn_weights = attn_weights.unsqueeze(-1)
                    attn_weights = self.get_2dPool(attn_weights)
                    attn_weights = attn_weights.squeeze(-1) # (batch_size, seq_len')
                                       
                    metric = self.get_2dPool(metric) # (batch_size, seq_len', head_dim)
                    
                    def visionzip(hidden_states, attention, metric, dominant_num, contextual_num, images_dtype, mm_newline_position):
                        # hidden_states (B,S,D); attention (B,S); metric (B,S,H)
                        batch_size, seq_len, embed_dim = hidden_states.shape
                        ## Dominant Visual Tokens
                        if mm_newline_position == "grid":
                            positions = torch.arange(seq_len, device=hidden_states.device)  # (seq_len)
                        
                        topk_indices = attention.topk(dominant_num, dim=1).indices  # (batch_size, dominant_num)
                        all_indices = topk_indices  # (batch_size, dominant_num)
                        
                        mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)  # (batch_size, seq_len) False means retained tokens
                        # after masked_select, (batch_size * (dominant_num) * embed_dim)
                        # finally, (batch_size, dominant_num, embed_dim) compare with hidden_states
                        dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(batch_size, dominant_num, embed_dim)

                        if mm_newline_position == "grid":
                            positions = positions.expand(batch_size, -1)  # (batch_size, seq_len)
                            dominant_positions = torch.gather(positions, 1, all_indices)  # (batch_size, dominant_num)
                            contextual_positions = positions.masked_select(mask).view(batch_size, -1)  # (batch_size, seq_len-dominant_num)

                        ### Filter
                        # metric: (batch_size, seq_len, head_dim)
                        # metric_filtered: (batch_size, seq_len-dominant_num, head_dim)
                        metric_filtered = metric[mask].view(batch_size, seq_len - dominant_num, metric.shape[2])

                        # compare with dominant_tokens
                        # hidden_states_filtered: (batch_size, seq_len-dominant_num, embed_dim)
                        hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(batch_size, seq_len - dominant_num, embed_dim) 
                        
                        metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)    # normalize for cosine similarity

                        ## Contextual Visual Tokens
                        step = max(1, (seq_len-dominant_num) // contextual_num)
                        target_indices = torch.arange(0, seq_len-dominant_num, step, device=metric_normalized.device)[:contextual_num]
                        # target_tokens: (batch_size, contextual_num, embed_dim)
                        target_tokens = metric_normalized[:, target_indices, :]
                        
                        if mm_newline_position == "grid":
                            contextual_positions = torch.gather(contextual_positions, 1, target_indices.unsqueeze(0).expand(hidden_states.shape[0], -1))  # (batch_size, contextual_num)

                        # compare with target_tokens
                        # tokens_to_merge: (batch_size, seq_len-dominant_num-1-contextual_num, embed_dim)
                        # target_token+tokens_to_merge = metric_normalized
                        tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
                        # calcute cosine similarity between tokens_to_merge and target_tokens
                        # similarity: (batch_size, seq_len-dominant_num-1-contextual_num, contextual_num)
                        similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
                        assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
                        # similarity.argmax(dim=2) for each tokens_to_merge, find the most similar target_token
                        # assign_one_hot: same size as similarity, but only the most similar target_token is 1, others are 0
                        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
                        # counts: (batch_size, contextual_num, 1)
                        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
                        # hidden_to_merge: (batch_size, seq_len-dominant_num-1-contextual_num, embed_dim)
                        hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]

                        # for each target_token, aggregate the hidden_to_merge
                        # assign_one_hot.transpose(1, 2): (batch_size, contextual_num, seq_len-dominant_num-1-contextual_num)
                        # aggregated_hidden: (batch_size, contextual_num, embed_dim)
                        aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
                        # target_hidden: (batch_size, contextual_num, embed_dim)
                        target_hidden = hidden_states_filtered[:, target_indices, :]  
                        
                        contextual_tokens = target_hidden + aggregated_hidden

                        if mm_newline_position == "grid":
                            all_positions = torch.cat([dominant_positions, contextual_positions], dim=1)  # (batch_size, total_tokens)
                            all_tokens = torch.cat([dominant_tokens, contextual_tokens], dim=1)  # (batch_size, total_tokens, embed_dim)
                            
                            # Sort tokens by their original positions
                            all_sorted_positions, all_sorted_indices = torch.sort(all_positions, dim=1) # (batch_size, total_tokens)
                            all_sorted_tokens = torch.gather(all_tokens, 1, all_sorted_indices.unsqueeze(-1).expand(-1, -1, all_tokens.shape[-1]))  # (batch_size, total_tokens, embed_dim)
                            
                            grid_size = int(math.sqrt(seq_len))
                            all_row_positions = all_sorted_positions // grid_size
                            
                            expanded_tokens_list = []
                            for cur_sorted_tokens, cur_row_positions in zip(all_sorted_tokens, all_row_positions):
                                expanded_tokens = []
                                new_line_token = self.model.image_newline.to(cur_sorted_tokens.device).unsqueeze(0) # (1,D)
                                for row in range(grid_size):
                                    find_row_tokens = cur_sorted_tokens[cur_row_positions == row]
                                    if len(find_row_tokens) > 0:
                                        expanded_tokens.append(torch.cat((find_row_tokens, new_line_token), dim=0))
                                    else:
                                        expanded_tokens.append(new_line_token)
                                batch_tokens = torch.cat(expanded_tokens, dim=0)  # (seq_len+grid_size, embed_dim)
                                expanded_tokens_list.append(batch_tokens)
                                
                            image_features = torch.cat(expanded_tokens_list, dim=0)  # (-1, embed_dim)
                        else:
                            # Merge with target hidden states and concatenate
                            image_features = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(images_dtype)    # (batch_size, total_tokens, embed_dim)
                        return image_features

                    image_features.append(visionzip(pooled_image_feat, attn_weights, metric, dominant_num, contextual_num, images_dtype, mm_newline_position))

                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # # Grid-wise
                            # image_feature = self.add_token_per_grid(image_feature)
                            # if getattr(self.config, "add_faster_video", False):
                            #     faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                            #     # Add a token for each frame
                            #     concat_slow_fater_token = []
                            #     # import pdb; pdb.set_trace()
                            #     for _ in range(image_feature.shape[0]):
                            #         if _ % self.config.faster_token_stride == 0:
                            #             concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            #         else:
                            #             concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            #     # import pdb; pdb.set_trace()
                            #     image_feature = torch.cat(concat_slow_fater_token)

                            #     # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            # [modify]
            text_token_count = sum([x.shape[0] for x in cur_labels_noim])
            vision_token_count = len(image_features[cur_image_idx])
            rank0_print(f"Batch {batch_idx}: Text tokens: {text_token_count} Original Vision tokens: {vision_token_count}")
    
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

