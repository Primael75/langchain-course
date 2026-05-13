import re 
import inspect


from  dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL= "qwen3:1.7b"



@traceable(run_type="tool")
def get_product_price(product: str)-> float:
    """Look up the price of a product in a catalogue """
    print(f" >>> Executing get_product_price(product='{product}')")
    prices= {"laptop": 1299.00, "headphones":156.65, "keyboard":234.23}
    return prices.get(product,0)

@traceable(run_type="tool")
def apply_discount( price: float, discount_tier: str)->float:
    """Apply a discount tier to a price and return the final price.
        Avalaible tiers: bronze, silver, gold  """
    print(f" >> Executing apply_discount(price={price}, discount_tier='{discount_tier})")
    price=float(price)
    discount_percentages={"bronze":5, "silver":12, "gold":23}
    discount= discount_percentages.get(discount_tier, 0)
    return round(price*(1 - discount/100),2)


tools={
    "get_product_price":get_product_price,
    "apply_discount":apply_discount 
    }

def get_tool_descriptions(tools_dict):
    descriptions=[]
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function,"__wrapped__",tool_function)
        signature=inspect.signature(original_function)
        docstring= inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature}-{docstring}")
    return "\n".join(descriptions)


tool_descriptions=get_tool_descriptions(tools)
tool_names=", ".join(tools.keys())

react_prompt= f""""
STRICT RULES - you must follow thesse exactly: \n"
"1. Never guess or assume any product price." \
"You must call the get_product_price first to get the real price .\n" \
"2. Only call the apply_discount AFTER you have received" \
"a price from get_product_price. Pass the exact price "
"returned by get_product_price - do  NOT pass a made-up number.\n"
"3. NEVER calculated disvounts yourself using math." \
"Always use the apply_disvcount tool.\n"
"4. If the user does not specify a discount tier," \
"ask them which tier to use - do NOT assume one. "

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought: """


                    





# tools_for_llm=[
#     {
#         "type":"function",
#         "function":{
#             "name":"get_product_price",
#             "description":"Look up the price of a product in a catalog",
#             "parameters":{
#                 "type":"object",
#                 "properties":{
#                     "product":{
#                         "type":"string",
#                         "description":"The product name, e.g. 'Laptop', 'headphones', 'keyboard' ",
#                     },
#                 },
#                 "required":["product"],
#             },
#         },
#     },
#      {
#         "type":"function",
#         "function":{
#             "name":"apply_discount",
#             "description":"Apply a discount tier to a price and return the final price.Avalaible tiers: bronze, silver, gold ",
#             "parameters":{
#                 "type":"object",
#                 "properties":{
#                     "price":{"type":"number","description":"The original price "},
#                     "discount_tier":{"type": "string", "description":"The discount tier: 'bronze','silver', 'gold' "},
#                 },
#                 "required":["price","discount_tier"],
#             },
#         },
#     },
    
    
    
# ]

#Helper: traced ollama call
# Without LangChain , we must manually trace LLM call for langsmith

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model,  messages=messages,options=options)
# --- Agent_Loop ---

@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    
    
    
    print(f"Question: {question}")
    print("="*60)


    prompt= react_prompt.format(question=question)
    scratchpad=""



   



    for iteration in range(1, MAX_ITERATIONS + 1 ):
        print(f"\n ---Iteration {iteration} ---")
        full_prompt=prompt + scratchpad

        response= ollama_chat_traced(
            model=MODEL,
            messages=[{"role":"user","content":full_prompt}],
            options={"stop":["\nObservation"], "temperature":0},
        )
        output=response.message.content
        print(f"LLM Output:\n{output}")


        print(f" [Parsing] Looking for the Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer: \s*(.+)", output)
        if final_answer_match:
            final_answer=final_answer_match.group(1).strip()
            print(f"[Parsed] Final Answer: {final_answer}")
            print("\n"+"="*60)
            print(f"Final answer:{final_answer}")
            return final_answer
        

        print(f"  [Parsing] Looking for Action and Action Input in LLM output...")

        action_match= re.search(r"Action:\s*(.+)", output)
        action_input_match=re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "[Parsing] ERROR: Could not parse or Action/Action Input for LLM ouput "
            )
            break

        tool_name= action_match.group(1).strip()
        tool_input_raw= action_input_match.group(1).strip()


        # if not tool_calls:
        #     print(f"\nFinal Answer: {ai_message.content}")
        #     return ai_message.content


        # # Process only the First tool call -force one tool per iteration 
        # tool_call = tool_calls[0]
        # # tool_name=  tool_call.get("name")
        # # tool_args= tool_call.get("args",{})
        # # tool_call_id= tool_call.get("id")

        # tool_name = tool_call.function.name
        # tool_args = tool_call.function.arguments
        # tool_call_id = None 

        print(f"[Tool selected] {tool_name} with args: {tool_input_raw}")

        # Split comma-separated args; strip key=prefix if LLM outputs key=value format 

        raw_args=[x.strip() for x in tool_input_raw.split(",")]
        args=[x.split("=",1)[-1].strip().strip("'\"") for x in raw_args]

        print(f" [Tool Executing] {tool_name}({args})....")
        if tool_name not in tools:
            observation=f"ERROR: Tool '{tool_name}' not found. Available tools: {list[str](tools.keys())}"
        else:
            observation=str(tools[tool_name](*args))
        # tool_to_use = tools_dict.get(tool_name)
        # if tool_to_use is None:
        #     raise ValueError(f"Tool '{tool_name}' not found")
        
        # observation = tool_to_use(**tool_args)

        print(f"[Tool Result] {observation}")

        # History is one growing string re-sent every iteration (replaces messages.append).
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

        # messages.append(ai_message)
        # messages.append(
        #     {
        #         "role":"tool",
        #         "content":str(observation),
        #     }

        #  )

    print("ERROR: Max iterations reached without a final answer")
    return None



        










    pass

if __name__=="__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result=run_agent("What is the price of a laptop after applying a gold discount?")


