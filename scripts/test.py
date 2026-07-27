from transformers import AutoModelForCausalLM

m = AutoModelForCausalLM.from_pretrained(
    "/data/wenhesun/hf_ckpts/qwen3_8b_grpo/global_step_100",
    torch_dtype="auto",
    device_map="cpu",
)
print("loaded ok:", m.config.model_type)