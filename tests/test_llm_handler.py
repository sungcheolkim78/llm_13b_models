import pytest
import sys

sys.path.append("..")
from scllm.llm_handler import LLMHandler

testset1_path = "tests/query_testset1.txt"


@pytest.fixture
def llm():
    return LLMHandler(model_engine="llama-cpp")


def test_load_model(llm):
    print(llm.list_models())


def test_role(llm):
    llm.set_hermes()

    # check role knowall
    llm.set_role_knowall()
    response = llm.run_query_with_system(
        "Who is the president of South Korea in 2005? Provide only a name."
    )
    assert "Roh Moo-hyun" in response

    # check role english tutor
    llm.set_role_knowall()
    response = llm.run_query_with_system(
        "This package have all code and documentation for the large lanuage model applications."
    )


def test_general_question(llm):
    with open(testset1_path, "r") as f:
        queries = f.readlines()

    queries = [s.strip() for s in queries]

    llm.set_hermes()
    llm.set_role_knowall()

    for query in queries[:3]:
        llm.run_query_with_system(query)
