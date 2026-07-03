import os
import sys

def test_openai(api_key):
    try:
        from openai import OpenAI
        print("Initializing OpenAI client...")
        client = OpenAI(api_key=api_key)
        print("Fetching accessible models list...")
        models = client.models.list()
        print("\nSUCCESS! Accessible Models:")
        for m in list(models)[:10]:
            print(f"- {m.id}")
    except Exception as e:
        print(f"\nFAILURE! Error details: {str(e)}")

def test_claude(api_key):
    try:
        from anthropic import Anthropic
        print("Initializing Anthropic client...")
        client = Anthropic(api_key=api_key)
        print("Sending test message to claude-3-5-sonnet-20241022...")
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'API Key is Valid!'"}]
        )
        print("\nSUCCESS! LLM Response:")
        print(response.content[0].text)
    except Exception as e:
        print(f"\nFAILURE! Error details: {str(e)}")

def test_groq(api_key):
    try:
        from groq import Groq
        print("Initializing Groq client...")
        client = Groq(api_key=api_key)
        print("Sending test message to llama-3.3-70b-versatile...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'API Key is Valid!'"}]
        )
        print("\nSUCCESS! LLM Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"\nFAILURE! Error details: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        choice = sys.argv[1].strip()
        api_key = sys.argv[2].strip()
    else:
        print("=== LLM API Key Validator ===")
        print("Select LLM Provider:")
        print("1. OpenAI")
        print("2. Anthropic Claude")
        print("3. Groq")
        
        choice = input("Enter choice (1-3): ").strip()
        if choice not in ["1", "2", "3"]:
            print("Invalid choice. Exiting.")
            sys.exit(1)
            
        api_key = input("Paste your API Key: ").strip()
        if not api_key:
            print("API Key cannot be empty. Exiting.")
            sys.exit(1)
            
    if choice == "1":
        test_openai(api_key)
    elif choice == "2":
        test_claude(api_key)
    elif choice == "3":
        test_groq(api_key)
