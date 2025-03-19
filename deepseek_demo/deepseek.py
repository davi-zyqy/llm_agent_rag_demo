import torch
from torch.xpu import device
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
device = torch.device('mps')
model.to(device)

def predict(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors='pt').to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

while True:
    user_input = input('文本：')
    if user_input == '文本：退出':
        break
    messages = [
        {'role': 'system', 'content': '你现在是一名大学数学教授，帮我解答线形代数的问题'},
        {'role': 'user', 'content': f'{user_input}'}
    ]
    response = predict(messages, model, tokenizer)
    print(response)