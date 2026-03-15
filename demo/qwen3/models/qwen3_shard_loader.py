from mirage.mpk.base_dynamic_shard_loader import BaseDynamicShardLoader 
from models.modeling_qwen3 import Qwen3RotaryEmbedding

class Qwen3ShardLoader(BaseDynamicShardLoader):
    
    def model_specific_initialition_logic(self):
        """
        Executes Qwen3-specific setup after all generic sharding 
        and loading is complete.
        """
        print("calling the qwen3 version of model specific initialization logic")
        self.reinitialize_rope_buffers()

    def reinitialize_rope_buffers(self):
        """
        Re-calculates inv_freq on the actual device since it's not in safetensors.

        Since the model is originally initialized on a meta device, the rotary embeddings were
        not calculated during that time. 

        This is specific to QWEN3 implementation.  
        """
        for name, module in self.model.named_modules():
            print("name", name, module.__class__.__name__)
            if module.__class__.__name__ == "Qwen3RotaryEmbedding":
                print("found a rotary embedding")
                module.to(self.device)
                
                inv_freq, attention_scaling = module.rope_init_fn(
                    module.config, self.device, **module.rope_kwargs
                )
                
                module.register_buffer("inv_freq", inv_freq, persistent=False)
                module.attention_scaling = attention_scaling
