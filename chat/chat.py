import argparse
import json

import gradio as gr
import requests


def chat_response(msg, history) :
    headers = {"User-Agent": "vLLM Client"}
    pload = {
        "prompt": msg,
        "stream": True,
        "max_tokens": 1024,
    }
    history.append([msg, None])
    response = requests.post(args.model_url,
                             headers=headers,
                             json=pload,
                             stream=True)

    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            history[-1][1] = data["text"][0]
            yield msg, history

    #return "", history


def chat_clear(msg, chatbot) :
    return "", []


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# 아~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")    
        chatbot = gr.Chatbot(height=500, value="")
        msg = gr.Textbox(label="Input", container=False, placeholder="Enter text and press ENTER")
        send_btn = gr.Button(value="Send", interactive=True)                
        clear = gr.Button(value="clear", interactive=True)                    

        clear.click(fn=chat_clear, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(fn=chat_response, inputs=[msg, chatbot], outputs=[msg, chatbot])
        send_btn.click(fn=chat_response, inputs=[msg, chatbot], outputs=[msg, chatbot])    
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11181)
    parser.add_argument("--model-url",
                        type=str,
                        default="http://10.10.20.24:11182/gemma-generate")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue().launch(server_name=args.host,
                        server_port=args.port,
                        share=False)
