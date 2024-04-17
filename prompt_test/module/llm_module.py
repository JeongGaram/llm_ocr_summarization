#-*- coding:utf-8 -*-
import re
import requests
import json
import unicodedata
from typing import Union
from config.llm_url import (
    MODEL_INFERENCE_URL, 
    TOKEN_URL, 
    CHUNK_URL, 
    SET_SPLITTER_URL
)


class LLM_Module :
    @classmethod
    def get_token_num(cls, input_text) : 
        input_text = cls.remove_special_char(input_text)                               
        response = requests.post(
                url=TOKEN_URL,
                data=json.dumps({"input_text" : input_text})
                )
        data_dict = eval(response.text)
        #{"token_num" : len(token_list), "token_list" : token_list}
        data_dict["text"] = input_text
        
        return data_dict


    @classmethod
    def remove_special_char(cls, text) :
        text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", str(text))
        return ' '.join(text.split())
    
    
    @classmethod
    def set_splitter_option(cls, chunk_size, chunk_overlap) :
        response = requests.post(
                url=SET_SPLITTER_URL,
                data=json.dumps({"chunk_size" : chunk_size, "chunk_overlap" : chunk_overlap})
                )
        print(response.text)
        return response.text
    
    
    @classmethod
    def get_chunk_text(cls, text) :
        response = requests.post(
                url=CHUNK_URL,
                data=json.dumps({"input_text" : text})
                )
        data_dict = eval(response.text)
        temp = []
        for chunk in data_dict["chunk_text"] :
            remove_pront_num = 0
            remove_back_num = 0
            for i in range(0, 3) :
                if unicodedata.name(chunk[i]) == "REPLACEMENT CHARACTER" :
                    remove_pront_num += 1                                
            for i in range(3, 0, -1) :
                if unicodedata.name(chunk[-i]) == "REPLACEMENT CHARACTER" :
                    remove_back_num += 1
            
            if remove_back_num != 0 :
                temp.append(chunk[remove_pront_num:-remove_back_num])
            else :
                temp.append(chunk[remove_pront_num:])
            
        data_dict["chunk_text"] = temp                
        return data_dict
        
    
    
    @classmethod
    def llm_inference_api(cls, text, input_dict) :
        input_dict["prompt"] = text        
        response = requests.post(
                url=MODEL_INFERENCE_URL,
                data=json.dumps(input_dict)
                )
        data_dict = eval(response.text)        
        return data_dict["text"]   
    
    
    @classmethod
    def make_prompt(cls, instruction, chunk, params_dict) :   
        input_text = instruction +"\n <문서>"+ chunk +"\n</문서>"  
        re_gen_num = 0             
                
        while True :            
            generated_text = cls.llm_inference_api(input_text, params_dict)[0]        
            re_gen_num += 1                            
            if re_gen_num == 2 or generated_text != "" :
                break
                                    
        return ' '.join(generated_text.split()) 


    @classmethod
    def make_chunk_summary(cls, summary_instruction : str, chunks : list, params_dict : dict) :
        #generate chunk summary        
        summary_list = []
        summary_document = """"""
        for i, chunk in enumerate(chunks) :
            if len(chunk) < 5 :
                continue                        
            generated_text = cls.make_prompt(summary_instruction, chunk, params_dict) 
            summary_list.append([chunk, generated_text])            
            summary_document += generated_text + "\n\n"            
            
        summary_document = re.sub("\*\*요약\*\*", "", summary_document)      
        summary_document = re.sub("\*\*요약:\*\*", "", summary_document)
        return summary_list, summary_document
    
    
    @classmethod
    def summary_module(        
            cls,
            summary_instruction : str,
            ori_doc : str,
            chunks : Union[list, str],
            summary_max_token_num : int,
            summary_total_step : int,
            params_dict : dict
        ) :
                
        chunk_list = chunks.split("\n\n")
        chunk_token = LLM_Module.get_token_num(ori_doc)["token_num"]
        if chunk_token < summary_max_token_num :                
            return ori_doc
        
        summary_step_num = 0            
        while True :
            #make summary
            summary_list, summary_doc = cls.make_chunk_summary(summary_instruction, chunk_list, params_dict)            
            summary_step_num += 1            
            #summary_doc_token_num
            token_num = cls.get_token_num(summary_doc)["token_num"]
            if token_num <= summary_max_token_num or summary_step_num == summary_total_step  : 
                break
                                        
            #new chunk_texts
            #concat summary list ele
            chunk_texts = []
            # 리스트의 각 요소에 대해 반복, 2개 단위로 짝을 지어 처리
            for i in range(0, len(summary_list), 2):
                # 인접한 두 요소를 연결
                if i+1 >= len(summary_list) :
                    chunk_texts.append(summary_list[-1][1])
                else :
                    token_num = cls.get_token_num(summary_list[i][1] + summary_list[i+1][1])["token_num"]
                    if token_num >= summary_max_token_num :
                        chunk_texts.append(summary_list[i][1])  
                        chunk_texts.append(summary_list[i+1][1])  
                    else :
                        chunk_texts.append(summary_list[i][1] + summary_list[i+1][1])  
        
        return summary_doc
    
    
    @classmethod
    def test_module(cls, instruction : str, summary_document : str, params_dict : dict) :
        max_gen_num = 4
        gen_num = 0
        while True :            
            gen_data = cls.make_prompt(instruction, summary_document, params_dict)
            if max_gen_num == gen_num or len(gen_data) >=3 :
                break
            gen_num += 1
            
        gen_data = gen_data.replace("<start_of_turn>model", "")
        gen_data = gen_data.replace("<eos>", "")
        
        return gen_data