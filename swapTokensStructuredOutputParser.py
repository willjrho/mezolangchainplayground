import os
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

#Load OpenAI API key
load_dotenv()

# Verify if key is loaded
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

# Define the structured output schema for swaps
swap_response_schemas = [
    ResponseSchema(name="amount", description="The amount of mUSD to swap."),
    ResponseSchema(name="from_currency", description="The token to swap from (should always be 'mUSD')."),
    ResponseSchema(name="to_currency", description="The token to receive (should always be 'BTC')."),
    ResponseSchema(name="router_address", description="The Dumpy Swap router address for executing the swap.")
]

# Create the StructuredOutputParser
swap_output_parser = StructuredOutputParser.from_response_schemas(swap_response_schemas)

# Define LLM and Swap Prompt for Parsing Transactions
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

swap_prompt_template = PromptTemplate(
    template="""
    Extract swap transaction details from this request:
    {input}

    - The token to swap from should always be 'mUSD'.
    - The token to receive should always be 'BTC'.
    - The router address should always be '0xC2E61936a542D78b9c3AA024fA141c4C632DF6c1'.
    
    {format_instructions}
    """,
    input_variables=["input"],
    partial_variables={"format_instructions": swap_output_parser.get_format_instructions()},
)


def extract_swap_details(prompt: str):
    """
    Uses LLM to extract structured swap transaction details from user input.
    """
    formatted_prompt = swap_prompt_template.format(input=prompt)
    response = llm.invoke(formatted_prompt)

    try:
        extracted_data = swap_output_parser.parse(response.content)
        return extracted_data
    except Exception as e:
        return f"Failed to extract swap details: {str(e)}"
    

user_input = "Swap 100 mUSD for BTC on Dumpy."
parsed_output = extract_swap_details(user_input)
print(parsed_output)

user_input = "Send it baller. We need to swap 1000 mUSD for BTC."
parsed_output = extract_swap_details(user_input)
print(parsed_output)
