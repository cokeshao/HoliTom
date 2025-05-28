import os
from .utils import apply_info
from .llava_arch import LlavaMetaForCausalLM_visionzip

def visionzip(model):
    
    print("################################")
    print("########## VISIONZIP ###########")
    print("################################")
    
    apply_info(model.model.vision_tower.vision_tower)
    
    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = LlavaMetaForCausalLM_visionzip.prepare_inputs_labels_for_multimodal
    LlavaMetaForCausalLM.encode_images = LlavaMetaForCausalLM_visionzip.encode_images
    LlavaMetaForCausalLM.encode_images_multi = LlavaMetaForCausalLM_visionzip.encode_images_multi
    
    return model
