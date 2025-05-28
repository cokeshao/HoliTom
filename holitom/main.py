import os
from .llava_arch import LlavaMetaForCausalLM_holitom
from .modeling_qwen2 import Qwen2Model_holitom

def holitom(model):
    
    print("################################")
    print("############ HoliTom ###########")
    print("################################")

    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_holitom.prepare_inputs_labels_for_multimodal
    LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_holitom.encode_images
    LlavaMetaForCausalLM.encode_images_multi = LlavaMetaForCausalLM_holitom.encode_images_multi
    
    LlavaMetaForCausalLM.holitom = LlavaMetaForCausalLM_holitom.holitom
    LlavaMetaForCausalLM.cluster_dpc_knn = LlavaMetaForCausalLM_holitom.cluster_dpc_knn
    LlavaMetaForCausalLM.select_static_windows = LlavaMetaForCausalLM_holitom.select_static_windows
    LlavaMetaForCausalLM.get_static_dynamic_features = LlavaMetaForCausalLM_holitom.get_static_dynamic_features
    LlavaMetaForCausalLM.merge_tokens_by_attention_density = LlavaMetaForCausalLM_holitom.merge_tokens_by_attention_density
    LlavaMetaForCausalLM.merge_tokens_by_density = LlavaMetaForCausalLM_holitom.merge_tokens_by_density
    LlavaMetaForCausalLM.merge_tokens_by_clustering = LlavaMetaForCausalLM_holitom.merge_tokens_by_clustering
    LlavaMetaForCausalLM.add_newline_token = LlavaMetaForCausalLM_holitom.add_newline_token
    
    if os.getenv("HOLITOM_k") is not None and os.getenv("HOLITOM_r") is not None:
        print("HoliTom")
        # HOLITOM_k = int(os.getenv("HOLITOM_k", 3))
        # HOLITOM_r = float(os.getenv("HOLITOM_r", 0.5))
        # print(f"HOLITOM_k: {HOLITOM_k}, HOLITOM_r: {HOLITOM_r}")
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        Qwen2Model.forward = Qwen2Model_holitom.forward
    else:
        print("HoliTom (w/o M)")
    
    return model
