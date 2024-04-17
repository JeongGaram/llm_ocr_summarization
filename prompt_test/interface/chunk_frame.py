import requests
import gradio as gr
from module.llm_module import LLM_Module

def set_splitter(chunk_size, chunk_overlap) :
    LLM_Module.set_splitter_option(chunk_size, chunk_overlap)
    return chunk_size, chunk_overlap

def clear_chunk() :
    return "", ""

    
def get_chunk(input_text) :
    if len(input_text) < 5 :
        return "ERROR", "ERROR"
    
    data_dict = LLM_Module.get_chunk_text(input_text)    
    chunk_info_data = """"""
    chunk_info_data += "chunk list num : " + str(data_dict["chnuk_num"]) + "\n\n"
    
    chunk_data = """"""

    for i, chunk_text in enumerate(data_dict["chunk_text"]) :
        token_dict = LLM_Module.get_token_num(chunk_text)         
        token_num = token_dict["token_num"]
        
        chunk_info_data += str(i) + " - chunk \n char_num : " + str(len(chunk_text)) +"\n token_num : " + str(token_num) +"\n\n"        
        chunk_data += chunk_text + "\n\n"
    print(chunk_data)
    return gr.Textbox(value=chunk_info_data), gr.Textbox(value=chunk_data)