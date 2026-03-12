from model.configuration_chatglm import ChatGLMConfig
from model.modeling_chatglm import ChatGLMForConditionalGeneration

config = ChatGLMConfig.from_json_file('model/config.json')
print(f"model config: {config}")
model = ChatGLMForConditionalGeneration(config, empty_init=False, device='cuda').half()
model.init_weights()
print(model)
