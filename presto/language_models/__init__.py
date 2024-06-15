from bioagent.language_models.llama import (
    LlamaLMMForCausalLM,
)
from bioagent.language_models.mistral import (
    MistralLMMForCausalLM,
)

LANGUAGE_MODEL_CLASSES = [MistralLMMForCausalLM, LlamaLMMForCausalLM]

LANGUAGE_MODEL_NAME_TO_CLASS = {cls.__name__: cls for cls in LANGUAGE_MODEL_CLASSES}
