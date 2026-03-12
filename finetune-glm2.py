import argparse
import functools
import json
import logging

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"  # 检测是否有GPU，如果有则使用，否则使用CPU


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_name_or_path', default='../ChatGLM2-6B/download_model/chatglm2-6b-int4',
                        type=str, required=False, help='预训练基础模型')
    parser.add_argument('--log_path', default='train.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--n_layer', default=24, type=int, required=False, help='num attention heads')
    parser.add_argument('--num_heads', default=16, type=int, required=False, help='num attention heads')
    parser.add_argument('--kv_channels', default=128, type=int, required=False, help='num kv heads')
    parser.add_argument('--n_embd', default=1024, type=int, required=False, help='Embedding dim')
    parser.add_argument('--ffn_hidden_size', default=2048, type=int, required=False, help='FFN hidden size')
    parser.add_argument('--seq_length', default=1024, type=int, required=False, help='seq length')
    parser.add_argument('--train_data_file', default='data/multi_test.jsonl', type=str, required=True, help='训练数据集文件路径')
    parser.add_argument('--eval_data_file', default='data/multi_test.jsonl', type=str, required=False, help='验证数据集文件路径')
    parser.add_argument('--max_source_length', default=256, type=int, required=False, help='max source length')
    parser.add_argument('--max_target_length', default=256, type=int, required=False, help='max target length')
    parser.add_argument('--num_workers', type=int, default=1, required=False, help="dataloader加载数据时使用的线程数量")
    parser.add_argument('--learning_rate', default=1e-2, type=float, required=False, help='学习率')
    parser.add_argument('--output_dir', default='trainer_output', type=str, required=False,
                        help='hf trainer output dir')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--warmup_steps', type=int, default=100, required=False, help='warm up步数')
    parser.add_argument('--save_total_limit', default=2, type=int, required=False, help='保存多少checkpoint')
    parser.add_argument('--eval_steps', default=100, type=int, required=False, help='多少step评估')
    parser.add_argument('--save_steps', default=100, type=int, required=False, help='多少step保存')
    parser.add_argument('--epochs', default=1, type=int, required=True, help='训练多少轮')
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int, required=False, help='梯度积累')
    parser.add_argument('--batch_size', default=8, type=int, required=True, help='batch大小')
    parser.add_argument('--ignore_index', default=-100, type=int, required=False,
                        help='对于ignore_index的label token不计算梯度')
    args = parser.parse_args()
    return args


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True).half().cuda()
    return model, tokenizer


# 处理原始数据集，转换成格式化后的数据集
def prepare_for_dialogue(examples):
    """对话任务处理"""
    prompts = []
    responses = []
    historys = []
    for doc in examples['text']:
        if "NaN" in doc:
            doc = doc.replace("NaN", "\"？\"")
        conversations = json.loads(doc)
        list = []
        line = []
        for turn in conversations['conversations']:
            line.append(turn['value'])
            if len(line) == 2 and turn['from'] == 'assistant':
                list.append(line)
                line = []
        for index, value in enumerate(list):
            prompts.append(value[0])
            responses.append(value[1])
            historys.append(list[:index])
    return {'prompt': prompts, 'response': responses, 'history': historys}


# 将格式化后的数据集转换成模型需要的格式，对文本对话进行tokenizer
def preprocess_function_train(examples, tokenizer, max_source_length, max_target_length):
    max_seq_length = max_source_length + max_target_length + 1
    # 模型需要的输入格式
    model_inputs = {
        "input_ids": [],
        "labels": [],
    }
    prompt_column = 'prompt'
    response_column = 'response'
    history_column = 'history'
    for i in range(len(examples[prompt_column])):
        if examples[prompt_column][i] and examples[response_column][i]:
            # query=问题，answer=答案，history=历史对话
            query, answer = examples[prompt_column][i], examples[response_column][i]
            history = examples[history_column][i] if history_column is not None else None
            # 根据 `问题` 和历史对话，构建提示词
            prompt = tokenizer.build_prompt(query, history)
            # 对 提示词 进行 encoding编码
            a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                     max_length=max_source_length)
            # 对 答案 进行encoding编码
            b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                     max_length=max_target_length)
            # 提示词 大小
            context_length = len(a_ids)
            # 提示词 + 答案 + 结束token，作为模型输入 input_ids
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            # 提示词token替换成pad token + 答案 + 结束toekn，作为模式输入labels
            labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]
            # 计算 pad长度
            pad_len = max_seq_length - len(input_ids)
            # 将input_ids padding到最大长度
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            # 将labels padding到最大长度
            labels = labels + [tokenizer.pad_token_id] * pad_len
            # 将labels pad token替换成-100，损失函数忽略-100的token，CrossEntropyLoss(ignore_index=-100)
            labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

    return model_inputs


def main():
    args = set_args()
    logger = create_logger(args)
    args_dict = vars(args)
    logger.info(f"config: {json.dumps(args_dict, indent=4)}")

    # Load the model and tokenizer
    model, tokenizer = load_model(args)
    # 训练模式
    model.train()
    print(tokenizer)
    print(model)
    train_dataset = Dataset.from_text(args.train_data_file)
    train_dataset = train_dataset.map(
        prepare_for_dialogue,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=True,
    )
    train_dataset = train_dataset.map(
        functools.partial(
            preprocess_function_train,
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length
        ),
        batched=True,
        num_proc=args.num_workers,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=True,
    )
    eval_dataset = None
    if args.eval_data_file is not None:
        eval_dataset = Dataset.from_text(args.eval_data_file)
        eval_dataset = eval_dataset.map(
            prepare_for_dialogue,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=True,
        )
        eval_dataset = eval_dataset.map(
            functools.partial(
                preprocess_function_train,
                tokenizer=tokenizer,
                max_source_length=args.max_source_length,
                max_target_length=args.max_target_length
            ),
            batched=True,
            num_proc=args.num_workers,
            remove_columns=eval_dataset.column_names,
            load_from_cache_file=True,
        )
    it = iter(train_dataset)
    for i in range(5):
        data = next(it)
        print(tokenizer.decode(data['input_ids']))
        print(''.join([tokenizer.decode([i]) if i != -100 else str(i) for i in data['labels']]))
        print('-' * 100)

    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.log_step,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy='steps' if eval_dataset is not None else 'no',
        learning_rate=args.learning_rate,  # LoRA通常使用稍高的学习率
        report_to='tensorboard',  # 禁用wandb等记录器
        # load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="eval_loss",
        save_safetensors=False,
        save_only_model=True,
        dataloader_pin_memory=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer.train()


if __name__ == '__main__':
    main()
