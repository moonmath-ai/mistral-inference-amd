from openai import OpenAI

# vLLM mimics the OpenAI API structure
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
  model="mistralai/Mistral-7B-Instruct-v0.3",
  messages=[{"role": "user", "content": "Can you explain Navier-Stokes equations for fluid dynamics?"}]
)

print(response.choices[0].message.content)
