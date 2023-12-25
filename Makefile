# Makefile

install:
	pip install --upgrade pip 
	pip install -r requirements.txt
	CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

lint:
	python -m black scllm
	pylint --disable=R,C scllm

test:
	python -m pytest -s -v tests
