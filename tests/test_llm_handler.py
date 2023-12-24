import pytest
import sys

sys.path.append("..")
from scllm.llm_handler import LLMHandler


def test_gen_text():
    llm = LLMHandler()

    llm.set_model()

    output = llm.generate("Q: Name the planets in the solar system? A: ")
    print(output)

    output = llm.generate("Q: Name the planets in the solar system? A: ", temperature=0.4)
    print(output)

    output = llm.generate("Q: Name the planets in the solar system? A: ", temperature=0.4, max_tokens=128)
    print(output)

    output = llm.generate("Q: Name the planets in the solar system? A: ", temperature=0.4, max_tokens=1024)
    print(output)
