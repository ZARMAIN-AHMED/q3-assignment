from dotenv import load_dotenv
load_dotenv()
from agents import Agent, Runner , function_tool, RunContextWrapper,input_guardrail,GuardrailFunctionOutput,ModelSettings
from pydantic import BaseModel
from dotenv import load_dotenv




class Account(BaseModel):
    name: str
    pin: int

class Guardrail_output(BaseModel):
    isNot_bank_related:bool

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="check if the user is asking you bank related quries.",
    output_type=Guardrail_output,
)

@input_guardrail
async def check_bank_related(ctx:RunContextWrapper[None],agent:Agent,input:str)->GuardrailFunctionOutput:
   
   result = await Runner.run(guardrail_agent,input,context=ctx.context)
   return GuardrailFunctionOutput(
       output_info=result.final_output,
       tripwire_triggered=result.final_output.isNot_bank_related
    )
   
def check_user(ctx:RunContextWrapper[Account],agent:Agent)->bool:
    if ctx.context.name == "Zarmain" and ctx.context.pin == 1234:
        return True
    else:
        return False

@function_tool(is_enabled=check_user)
def check_balance(account_number: str) -> str:
    return f"The balance of account is $1000000"

def dynamic_instruction(ctx:RunContextWrapper[Account],agent:Agent):
    return f"user name is {ctx.context.name}  check the users name if its correct use the balance check tool to check thier balance"

bank_agent = Agent(
    name="Bank Agent",
    instructions=dynamic_instruction,
    tools=[check_balance],
    input_guardrails=[check_bank_related], 
)

user_context = Account(name="Zarmain", pin=1234)

result = Runner.run_sync(bank_agent, "what is my balance my acount 93849348", context=user_context)
print(result.final_output)