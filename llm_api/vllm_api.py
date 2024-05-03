"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
import os
import argparse
import json
import ssl
import uvicorn
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from transformers import AutoTokenizer
from langchain_text_splitters import TokenTextSplitter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

@app.post("/get-tokens")
async def get_tokens(request: Request) -> Response:
    request_dict = await request.json()
    input_text = request_dict.pop("input_text")
    token_id_list = tokenizer.encode(input_text)
    token_list = tokenizer.convert_ids_to_tokens(token_id_list)
    return {"token_num" : len(token_list), "token_list" : token_list}


@app.post("/get-chunks")
async def get_chunks(request: Request) -> Response:
    request_dict = await request.json()
    input_text = request_dict.pop("input_text")
    chunks = splitter.split_text(input_text)
    chunk_list = []
    for chunk in chunks :
        chunk_list.append(chunk.encode("utf-8"))
              
    return {"chnuk_num" : len(chunks), "chunk_text" : chunk_list}


@app.post("/set-splitter")
async def set_splitter(request: Request) -> Response:
    request_dict = await request.json()    
    chunk_size = request_dict.pop("chunk_size")
    chunk_overlap = request_dict.pop("chunk_overlap")    
    global splitter
    splitter._chunk_size = int(chunk_size)
    splitter._chunk_overlap =int(chunk_overlap)     
    return_message = "SETUP \n chunk_size : " + str(splitter._chunk_size) + " chunk_overlap : " + str(splitter._chunk_overlap) + "\n"
    return return_message


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/gemma-generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    prompt = "<start_of_turn>user\n" + prompt + "<eos>"
    
    
    temperature = request_dict.pop("temperature", 0.5)
    top_p = request_dict.pop("top_p", 0.95)
    max_tokens = request_dict.pop("max_tokens", 1024)
    stream = request_dict.pop("stream", False)    
    sampling_params = SamplingParams(
                        temperature=float(temperature),
                        top_p=float(top_p),
                        max_tokens=int(max_tokens),
                        stop_token_ids=[1]
                    )
    print(sampling_params)
    print("\n\n\n")
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)
    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")
    
    if stream:
        return StreamingResponse(stream_results())
    
    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output
        #print(request_output.outputs[0].text)

    assert final_output is not None
    prompt = final_output.prompt    
    text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    #print(text_outputs)
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="6712")
    parser.add_argument("--model_path", type=str, default="/home/0_models/google-gemma-1.1-7b-it")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--splitter_chunk_size", type=int, default=5000)
    parser.add_argument("--splitter_chunk_overlap", type=int, default=500)
        
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs", type=str, default=None, help="The CA certificates file")     
    parser.add_argument("--ssl-cert-reqs", type=int, default=int(ssl.CERT_NONE), help="Whether client certificate is required (see stdlib ssl module's)")
    parser.add_argument("--root-path", type=str, default=None, help="FastAPI root_path when app is behind a path based routing proxy")    
    args = parser.parse_args()
            
    engine_args = AsyncEngineArgs(model=args.model_path, max_model_len=args.max_model_len, dtype="float16")    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    #TODO : 비동기 토크나이저로 변경해야함.
    #허깅페이스 찾아봐야할듯..?
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=args.splitter_chunk_size,
            chunk_overlap=args.splitter_chunk_overlap
            )

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)