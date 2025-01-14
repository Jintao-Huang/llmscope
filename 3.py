from swift.llm import sft_main, TrainArguments


sft_main(TrainArguments(
    model='Qwen/Qwen2.5-7B-Instruct',
    train_type='lora',
    dataset=['AI-ModelScope/alpaca-gptq-data-zh#500',
             'AI-ModelScope/alpaca-gptq-data-en#500',
             'swift/self-cognition#500'],
    torch_dtype='bfloat16',
    # ...
))