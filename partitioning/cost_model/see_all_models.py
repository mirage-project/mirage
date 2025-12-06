from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://ai-gateway.andrew.cmu.edu/")
models = client.models.list()
for m in models:
    print(m.id)