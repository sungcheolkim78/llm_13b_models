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

download_models:
	wget https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf
	mv *.gguf models/