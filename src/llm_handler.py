# Sung-Cheol Kim, Copyright 2023

from ctransformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama


class LLMHandler():
    def __init__(self, model_dir: str = "/home/skim/llm/models/", model_name: str = "zephyr-7b-alpha.Q5_K_M.gguf"):
        self.model_dir = model_dir
        self.model_name = model_name
        self.engine = "llama-cpp"

    def set_model(
        self, model_type: str = "llama"
    ):
        """set LLM model on GPU."""

        if self.engine == "ctransformers":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir + self.model_name,
                model_type=model_type,
                gpu_layers=50,
                hf=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        elif self.engine == "llama-cpp":
            self.model = Llama(
                model_path=self.model_dir + self.model_name,
                n_threads=10,
                n_gpu_layers=40,
                n_ctx=1024,
                verbose=False
            )

    def generate(self, prompt: str, **kwargs):
        if self.engine == "llama-cpp":
            output = self.model(
                prompt,
                stop=["Q:", "\n"],
                echo=True,
                **kwargs
            )

        return output
