import torch

from pathlib import Path
from ctransformers import AutoModelForCausalLM, AutoTokenizer


if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")


def test_model_gguf():
    model_dir = "/home/skim/llm/models/"
    model_name = "orca-2-13b.Q5_K_S.gguf"
    model_path = Path(model_dir) / model_name

    model = AutoModelForCausalLM.from_pretrained(model_dir + model_name, model_type="llama", gpu_layers=50, hf=True, threads=8)
    tokenizer = AutoTokenizer.from_pretrained(model)

    system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
    user_message = "How can you determine if a restaurant is popular among locals or mainly attracts tourists, and why might this information be useful?"

    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"

    inputs = tokenizer(prompt, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"], max_new_tokens=1024)
    answer = tokenizer.batch_decode(output_ids)[0]

    print("-" * 60)
    print(answer)

    # This example continues showing how to add a second turn message by the user to the conversation
    second_turn_user_message = "Give me a list of the key points of your first answer."

    # we set add_special_tokens=False because we dont want to automatically add a bos_token between messages
    second_turn_message_in_markup = f"\n<|im_start|>user\n{second_turn_user_message}<|im_end|>\n<|im_start|>assistant"
    second_turn_tokens = tokenizer(second_turn_message_in_markup, return_tensors='pt', add_special_tokens=False)
    second_turn_input = torch.cat([output_ids, second_turn_tokens['input_ids']], dim=1)

    output_ids_2 = model.generate(second_turn_input, max_new_tokens=1024)
    second_turn_answer = tokenizer.batch_decode(output_ids_2)[0]

    print("-" * 60)
    print(second_turn_answer)
