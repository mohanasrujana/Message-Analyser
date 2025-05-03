import ollama

response = ollama.chat(
    model='mistral',
    messages=[{"role": "user", "content": "Hi, what’s 2+2?"}]
)

print(response['message']['content'])
