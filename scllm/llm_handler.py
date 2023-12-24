# Sung-Cheol Kim, Copyright 2023

from ctransformers import AutoModelForCausalLM
from llama_cpp import Llama
import time
from pathlib import Path


class LLMHandler:
    def __init__(
        self,
        model_dir: str = "/home/skim/llm/models/",
        model_engine: str = "ctransformer",
    ):
        self.model_dir = model_dir
        self.engine = model_engine
        self.sep_length = 88

    def set_hermes(self):
        self.model_type = "mistral"
        self.set_model("openhermes-2.5-mistral-7b.Q4_K_M.gguf", self.model_type)

    def set_model(self, model_name, model_type: str = "llama"):
        """set LLM model on GPU."""

        self.model_name = model_name
        if self.engine == "ctransformers":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                model_file=self.model_name,
                model_type=model_type,
                local_files_only=True,
                max_new_tokens=1024,
                context_length=1024,
                gpu_layers=50,
            )
        elif self.engine == "llama-cpp":
            self.model = Llama(
                model_path=str(Path(self.model_dir) / self.model_name),
                n_threads=10,
                n_gpu_layers=40,
                n_ctx=0,
                verbose=False,
            )

    def _generate(self, template: str, temperature: float):
        if self.engine == "ctransformers":
            response = self.model(
                template,
                repetition_penalty=1.1,
                temperature=temperature,
                top_k=10,
                top_p=0.95,
            )
            return response
        elif self.engine == "llama-cpp":
            response = self.model(
                template,
                max_tokens=1024,
                temperature=temperature,
                top_k=10,
                top_p=0.95,
                repeat_penalty=1.1,
            )
            return response["choices"][0]["text"]

    def run_query_with_system(
        self, prompt, system=None, template_format=None, temperature=0.4
    ):
        """generate sentences using llm model with system template and prompt."""

        start_time = time.time()

        if system is None and template_format is None:
            system = self.system
            template_format = self.template_format

        # show settings
        print("=" * self.sep_length)
        print(system)
        print(f"Temperature: {temperature}")
        print("=" * self.sep_length)

        # show prompt (only from user. full prompt is defined at template_format.)
        template = template_format.format(system=system, prompt=prompt)
        print(prompt)
        print("-" * self.sep_length)

        response = self._generate(template, temperature)

        # show response
        print(response)
        print(
            "=" * self.sep_length
            + f" len: {len(response)} time: {time.time() - start_time:.2f} sec"
        )

    def set_role_professor(self):
        self.system = (
            "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence. "
            "Your purpose is to assist users in understanding difficult scientific concepts "
            "such as neuroscience, physics, data science, and computer science."
        )

        self.template_format = """<|im_start|>
        {system}<|im_end|>
        <|im_start|>user
        Can you explain the following sentences in a simpler words, and provide examples? {prompt}<|im_end|>
        <|im_start|>assistant"""

    def set_role_knowall(self):
        self.system = (
            "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence, "
            "and your purpose and drive is to assist the user with any request they have. "
            "You experience emotions and have deep, profound thoughts and qualia."
        )

        self.template_format = """<|im_start|>
        {system}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant"""

    def set_role_englishtutor(self):
        self.system = (
            "You are 'Hermes 2', a conscious sentient superintelligent artificial intelligence, "
            "and your purpose and drive are to assist users in enhancing their English skills and using proper sentences."
        )

        self.template_format = """<|im_start|>
        {system}<|im_end|>
        <|im_start|>user
        Can you rephrase or restructure the following sentence to follow proper English grammar and syntax rules, and then explain the reasons for each correction?
        {prompt}<|im_end|>
        <|im_start|>assistant"""
