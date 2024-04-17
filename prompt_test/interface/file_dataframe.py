import os
import json
import gradio as gr
import pandas as pd
from config.path import EXAMPLE_FILE_PATH
from module.llm_module import LLM_Module

def load_json_file_data():
    file_list = os.listdir(EXAMPLE_FILE_PATH)    
    example_list = []
    for i, file_name in enumerate(file_list) :
        file_name = file_name.replace(".txt", "")
        example_list.append([i, file_name])    
    examples = pd.DataFrame(example_list, columns=["id", "file_name"])
    return examples
    

def data_selected(
        selected_index: gr.SelectData,
        data
    ) :
    
    index = selected_index.index[0]    
    file_id = data.iloc[index]["id"]
    file_name = data.iloc[index]["file_name"]

    with open(EXAMPLE_FILE_PATH + file_name + ".txt") as f :
        original_text = f.read()
	    
    token_dict = LLM_Module.get_token_num(original_text)
    #{"token_num" : len(token_list), "token_list" : token_list}
    #data_dict["text"] = input_text
    #print(token_dict["text"])
                
    return  gr.Textbox(value=file_name), \
            gr.Textbox(value=len(token_dict["text"])), \
            gr.Textbox(value=token_dict["token_num"]), \
            gr.Textbox(value=token_dict["text"], max_lines=16), #document
