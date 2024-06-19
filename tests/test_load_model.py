import os
from bioagent.inference import load_trained_lora_model, load_trained_model
from bioagent.language_models.llama import LlamaLMMForCausalLM

if __name__ == "__main__":
    # check load full model
    model_name_path = "checkpoints/llava-moleculestm-vicuna-7b-v1.5-sft-full"
    pretrained_projector_path = "checkpoints/llava-moleculestm-vicuna-7b-v1.5-stage1/lmm_projector.bin"
    model, tokenizer = load_trained_model(
        model_name_or_path=model_name_path,
        pretrained_projectors_path=pretrained_projector_path,
    )