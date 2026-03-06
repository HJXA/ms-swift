# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Here is another way to register the model, by customizing the get_function.

The get_function just needs to return the model + tokenizer/processor.
"""
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.model.register import ModelLoader
from swift.utils import Processor
from swift import get_model_processor



import sys
sys.path.append("ms-swift/HJXA/Custom_Model/hjxa_minimind")
from modeling_hjxa_minimind import HJXA_MiniMindForCausalLM, HJXA_MiniMindConfig

from transformers import AutoConfig, AutoModelForCausalLM


class HJXA_MiniMind_ModelLoader(ModelLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        return HJXA_MiniMindConfig.from_pretrained(model_dir)

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        return HJXA_MiniMindForCausalLM.from_pretrained(
            model_dir, config=config, torch_dtype=self.torch_dtype, **model_kwargs)

def register_hjxa_minimind():

    print("register_hjxa_minimind ing~")

    AutoConfig.register(
        "hjxa_minimind",
        HJXA_MiniMindConfig
    )

    AutoModelForCausalLM.register(
        HJXA_MiniMindConfig,
        HJXA_MiniMindForCausalLM
    )

    register_model(
        ModelMeta(
            model_type='hjxa_minimind',
            model_groups=[
                ModelGroup([Model('HJXA/HJXA_MiniMind_0.5B', 'HJXA/HJXA_MiniMind_0.5B'),
                            Model('HJXA/HJXA_MiniMind_1B', 'HJXA/HJXA_MiniMind_1B'),
                            Model('HJXA/HJXA_MiniMind_25M', 'HJXA/HJXA_MiniMind_25M'),
                            Model('HJXA/HJXA_MiniMind_55M', 'HJXA/HJXA_MiniMind_55M'),
                            Model('HJXA/HJXA_MiniMind_104M', 'HJXA/HJXA_MiniMind_104M'),])
            ],
            template='minimind',
            loader = HJXA_MiniMind_ModelLoader,
            is_multimodal=False,
        ))

def load_hjxa_minimind(model_path):
    register_hjxa_minimind()
    model, tokenizer = get_model_processor(model_path)
    return model, tokenizer
    

if __name__ == '__main__':
    model, tokenizer = load_hjxa_minimind("/ruilab/jxhe/CoE_Monitor/checkpoints/coe_pt_init_models/HJXA_MiniMind_25M")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    print(f"Total params (Million): {total_params / 1e6:.2f} M")




