import openai
openai.api_type = "azure"
openai.api_base = "https://pengying.openai.azure.com/"

openai.api_version = "2023-05-15"
openai.api_key = "5587ea77923f4365ac2824aa971e4721"



path_prompt="Hello.Who are you?"
response = openai.ChatCompletion.create(
    engine="VoroNav", # engine = "deployment_name".
    messages=[
            {"role": "user", "content":path_prompt},
    ],
    temperature=0.8,
)
answer=response["choices"][0]["message"]["content"]
print(answer)