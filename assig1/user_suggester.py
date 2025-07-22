import asyncio
import streamlit as st
from dotenv import load_dotenv
import os

try:
    asyncio.get_running_loop() 
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")


from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

client = AsyncOpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",  # or "gemini-1.5-pro" etc
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

agent = Agent(
    name="Smart Product Agent",
    model=model,
    instructions=(
        "You are a helpful assistant that suggests medicine based on user needs.\n\n"
        "Example:\n"
        "- If the user says 'I have a headache', you should suggest a medicine or product and explain why it's suitable.\n"
        "- If the user says 'My skin is dry', you should recommend a skincare product with a reason.\n\n"
        "Keep your answers clear, friendly, and informative."
    )
)

st.set_page_config(page_title="Smart Product Agent", layout="centered")
st.title("💊 Smart Product Recommender")

user_input = st.text_input("🤖 Ask your question:", placeholder="e.g., I have a sore throat")

def main():
    if st.button("Get Suggestion") and user_input.strip() != "":
        with st.spinner("Thinking..."):
            result = Runner.run_sync(agent, user_input, run_config=config)
        st.markdown(f"### 💡 Suggestion:\n{result.final_output}")

if __name__ == "__main__":
    main()



# from dotenv import load_dotenv
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
# from agents.run import RunConfig
# import os

# load_dotenv()

# google_api_key = os.getenv("GEMINI_API_KEY")

# client = AsyncOpenAI(
#     api_key=google_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=client
# )

# config = RunConfig(
#     model=model,
#     model_provider=client,
#     tracing_disabled=True
# )

# agent = Agent(
#     name="Smart Product Agent",
#     model=model,
#     instructions=(
#         "You are a helpful assistant that suggests products or remedies based on user needs. If the user describes a problem (e.g., 'I have a headache'), suggest a suitable product or medicine and explain why it is appropriate."
#     )
# )

# def main():
#     prompt = input("Ask your question: ")
#     result = Runner.run_sync(agent, prompt, run_config=config)  
#     print(result.final_output)

# if __name__ == "__main__":
#     main()
