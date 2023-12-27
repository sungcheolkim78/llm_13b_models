# Makefile

install.cpu:
	pip install --upgrade pip 
	pip install -r requirements.txt
	pip install llama-cpp-python 

install.cuda:
	pip install --upgrade pip 
	pip install -r requirements.txt
	CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

lint:
	python -m black src
	pylint --disable=R,C src

test:
	python -m pytest -s -v tests

download_models:
	wget https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf
	wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/ggml-model-q4_k.gguf
	wget https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf
	wget https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf
	wget https://huggingface.co/abetlen/replit-code-v1_5-3b-GGUF/resolve/main/replit-code-v1_5-3b.Q4_0.gguf
	mv *.gguf models/
