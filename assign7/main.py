import os
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    Agent, OpenAIChatCompletionsModel, RunConfig, Runner,
    function_tool, RunContextWrapper, input_guardrail,
    GuardrailFunctionOutput, ModelSettings, AsyncOpenAI
)

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# ---------- External client setup ----------
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

class Account(BaseModel):
    name: str
    pin: int

class Guardrail_output(BaseModel):
    is_not_bank_related: bool

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="check if the user is asking you bank related quries.",
    output_type=Guardrail_output,
)

@input_guardrail
async def check_bank_related(ctx: RunContextWrapper[None], agent: Agent, input: str) -> GuardrailFunctionOutput:
    result = await Runner.run(
        guardrail_agent,
        input,
        context=ctx.context,
        config=config   # ✅ Gemini run config pass karna zaroori hai
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_not_bank_related
    )

def check_user(ctx: RunContextWrapper[Account], agent: Agent) -> bool:
    return ctx.context.name == "Asharib" and ctx.context.pin == 1234

@function_tool(is_enabled=check_user)
def check_balance(account_number: str) -> str:
    return f"The balance of account is $1000000"

def dynamic_instruction(ctx: RunContextWrapper[Account], agent: Agent):
    return f"user name is {ctx.context.name}  check the users name if its correct use the balance check tool to check thier balance"

bank_agent = Agent(
    name="Bank Agent",
    instructions=dynamic_instruction,
    tools=[check_balance],
    input_guardrails=[check_bank_related], 
    model_settings=ModelSettings(
        temperature=0.2,
        tool_choice='required',
        max_tokens=1000,
        parallel_tool_calls=None,
        frequency_penalty=0.3,
        presence_penalty=0.2
    ),
    reset_tool_choice=False,
    tool_use_behavior='run_llm_again'
)

user_context = Account(name="Asharib", pin=1234)

# ✅ Run agent synchronously
result = Runner.run_sync(
    bank_agent,
    "what is my balance my acount 93849348",
    context=user_context,
    config=config
)

print(result.final_output)
