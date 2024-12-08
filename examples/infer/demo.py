# Copyright (c) Alibaba, Inc. and its affiliates.
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_batch(engine: 'InferEngine', dataset: str):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    dataset = load_dataset([dataset], strict=False, seed=42)[0]
    print(f'dataset: {dataset}')
    metric = InferStats()
    resp_list = engine.infer([InferRequest(**data) for data in dataset], request_config, metrics=[metric])
    query0 = dataset[0]['messages'][0]['content']
    print(f'query0: {query0}')
    print(f'response0: {resp_list[0].choices[0].message.content}')
    print(f'metric: {metric.compute()}')
    # metric.reset()  # reuse


def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    request_config = RequestConfig(max_tokens=512, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])
    query = infer_request.messages[0]['content']
    print(f'query: {query}\nresponse: ', end='')
    for resp_list in gen:
        print(resp_list[0].choices[0].delta.content, end='', flush=True)
    print()
    print(f'metric: {metric.compute()}')


if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
    from swift.plugin import InferStats
    model = 'Qwen/Qwen2.5-1.5B-Instruct'
    infer_backend = 'pt'

    if infer_backend == 'pt':
        engine = PtEngine(model, max_batch_size=64)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine(model, max_model_len=32768)
    elif infer_backend == 'lmdeploy':
        from swift.llm import LmdeployEngine
        engine = LmdeployEngine(model)

    infer_batch(engine, 'AI-ModelScope/alpaca-gpt4-data-zh#1000')
    messages = [{'role': 'user', 'content': 'who are you?'}]
    infer_stream(engine, InferRequest(messages=messages))
