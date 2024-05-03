import requests
import json
print("\n vllm test request \n")
url = "http://10.10.20.23:6711/gemma-generate"
response = requests.post(url=url, data=json.dumps(
                                {
                                    "prompt" : "육대범은 독립 운동가이다 yes or no ?",
                                }
                            )
                        )
print(eval(response.text))
