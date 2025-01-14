# Copyright (c) Alibaba, Inc. and its affiliates.
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == '__main__':
    from swift.llm import InferRequest, RequestConfig, VllmEngine

    # test env: vllm==0.6.5, transformers==4.47.1
    os.environ['MAX_PIXELS'] = '1003520'
    model = 'Qwen/Qwen2-VL-2B-Instruct'
    # If you encounter insufficient GPU memory, please reduce `max_model_len` and set `max_num_seqs=5`.
    engine = VllmEngine(model, max_model_len=32768, limit_mm_per_prompt={'image': 5, 'video': 2})

    # Support URL/Path/base64/PIL.Image
    infer_requests = [
        InferRequest(
            messages=[{
                'role': 'user',
                'content': '<image>How many sheep are there in the picture?'
            }],
            images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']),
        InferRequest(messages=[{
            'role': 'user',
            'content': 'who are you?'
        }], )
    ]

    request_config = RequestConfig(max_tokens=512, temperature=0)
    resp_list = engine.infer(infer_requests, request_config)
    print(f'query0: {infer_requests[0].messages[0]["content"]}')
    print(f'response0: {resp_list[0].choices[0].message.content}')
    print(f'query1: {infer_requests[1].messages[0]["content"]}')
    print(f'response1: {resp_list[1].choices[0].message.content}')
