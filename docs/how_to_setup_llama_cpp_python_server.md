# How to setup llama-cpp-python server

First of all, please check this site - [link](https://llama-cpp-python.readthedocs.io/en/latest/server/#llama_cpp.server.settings.ModelSettings.chat_format)
Now this local server is stable to behave as OpenAI compatible server with multiple models.

## Requirements

You need to download large language models for different models. 

- OpenHermes-2.5-Mistral-7B-GGUF (gpt-3.5-turbo compatible)
- ggml_llava-v.15-7b (gpt-4-vision-preview compatible)
- mmproj-model-f16.gguf (clip model)
- mistral-7b-v0.1-GGUF (text-davinci-003 compatible)
- replit-code-v1_5-3b-GGUF (copilot-codex compatible)

You can download all these files using following commands.

```{bash}
mkdir models
make download_models
```

## Modify server_config.json file

You can find `server_config.json` file under `/server` folder. Use default values, but please
check the path to the model files where your server start.

You can also modify the port and default value is 8080.

## Start llama_cpp.server

Go to the folder where server_config.json exists. Execute the following command.

```{bash}
python3 -m llama_cpp.server --config_file server_config.json
```
