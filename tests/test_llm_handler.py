import pytest
import sys

sys.path.append("..")
from scllm.llm_handler import LLMHandler

testset1_path = "tests/query_testset1.txt"


def test_load_model():
    llm = LLMHandler(model_engine="llama-cpp")


def test_role():
    llm = LLMHandler(model_engine="llama-cpp")

    llm.set_hermes()
    llm.set_role_knowall()
    response = llm.run_query_with_system(
        "Who is the president of South Korea in 2005? Provide only a name."
    )

    assert "Roh Moo-hyun" in response


def test_general_question():
    with open(testset1_path, "r") as f:
        queries = f.readlines()

    queries = [s.strip() for s in queries]
    print(queries)

    llm = LLMHandler(model_engine="llama-cpp")

    llm.set_hermes()
    llm.set_role_knowall()

    for query in queries:
        llm.run_query_with_system(query)
