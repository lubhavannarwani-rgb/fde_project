from groq import Groq
from app.core.config import get_settings

s = get_settings()
print("Provider:", s.llm_provider)
print("Configured model:", repr(s.groq_model))
print("Key loaded:", bool(s.groq_api_key))
print("--- Models your key can use ---")
for m in sorted(Groq(api_key=s.groq_api_key).models.list().data, key=lambda m: m.id):
    print(m.id)
