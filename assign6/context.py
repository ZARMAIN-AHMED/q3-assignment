import os
from dotenv import load_dotenv
from agents import Agent, RunContextWrapper, function_tool, Runner, OutputGuardrail
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents import AsyncOpenAI
from pydantic import BaseModel
from typing import Optional

# -------------------- Load API Key --------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("⚠ GEMINI_API_KEY is missing in .env")

# Gemini client
client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model wrapper for Agents SDK
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

class userinfo(BaseModel):
    name: str
    issue_type: str = "general"
    is_premium_user: bool = False

@function_tool
def refund(wrapper: RunContextWrapper[userinfo])-> str:
    """Process a refund for premium users only."""
    return (
        f"Refund processed for {wrapper.context.name}."
        if wrapper.context.is_premium_user else f"Refund denied. {wrapper.context.name} is not premium."
    )

refund.is_enabled = lambda ctx, agent=None: ctx.context.is_premium_user

@function_tool
def restart_service(wrapper: RunContextWrapper[userinfo]) -> str:
    """Restart the service for technical issues."""
    return (
        f"Service restarted for {wrapper.context.name}."
        if wrapper.context.issue_type == "technical"
        else "Restart service only for technical issues."
    )
restart_service.is_enabled = lambda ctx, agent=None: ctx.context.issue_type == "technical"

@function_tool
def answer_general(wrapper: RunContextWrapper[userinfo]) -> str:
    """Answer a general query."""
    return f"Here's the answer to your general question, {wrapper.context.name}."

def classify_issue(wrapper: RunContextWrapper[userinfo], message: str) -> str:
    msg = message.lower()
    if "refund" in msg or "payment" in msg:
        wrapper.context.issue_type = "billing"
        return "Billing issue detected."
    elif "service" in msg or "restart" in msg or "technical" in msg:
        wrapper.context.issue_type = "technical"
        return "Technical issue detected."
    else:
        wrapper.context.issue_type = "general"
        return "General issue detected."

billing = Agent(
    name="Billing agent",
    instructions="You are a helpful billing agent and you handle billing issues.",
    tools=[refund]
)

tech = Agent(
    name="Tech Agent",
    instructions="You are a helpful technical agent and you handle technical/service issues.",
    tools=[restart_service]
)

general = Agent(
    name="General Agent",
    instructions="You are a helpful general agent and you handle general queries.",
    tools=[answer_general]
)

agent = Agent(
    name="supportagent",
    instructions="Classify the query and switch to the correct agent.",
    tools=[refund, restart_service, answer_general]
)

def route_message(wrapper: RunContextWrapper[userinfo], message: str):
    classify_issue(wrapper, message)
    if wrapper.context.issue_type == "billing":
        return billing
    elif wrapper.context.issue_type == "technical":
        return tech
    else:
        return general

def main():
    print("🎓 Console-Based Support Agent System (Gemini)")
    name = input("Enter your name: ")
    premium = input("Are you a premium user? (yes/no): ").strip().lower() == "yes"

    ctx = userinfo(name=name, is_premium_user=premium)
    active_agent = agent

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        if active_agent == agent:
            new_agent = route_message(RunContextWrapper(ctx), user_input)
            print(f"🔄 Switching to {new_agent.name}")
            active_agent = new_agent
            continue

        result = Runner.run_sync(active_agent, context=ctx, input=user_input, run_config=config)
        print(f"{active_agent.name}: {(result.final_output)}")

if __name__ == "__main__":
    main()



