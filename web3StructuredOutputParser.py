from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

#Load OpenAI API key
load_dotenv()

#Verify if key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

#Define expected fields
response_schemas = [
    ResponseSchema(name="amount", description="The amount of cryptocurrency to transfer."),
    ResponseSchema(name="currency", description="The cryptocurrency to transfer (BTC, mUSD, ect.)"),
    ResponseSchema(name="recipient", description="The recipient's Mezo address."),
]

#Create the output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schemas)


#Define the prompt template
prompt_template = PromptTemplate(
    template = "Extract transaction details from this request:\n{input}\n{format_instructions}",
    input_variables=["input"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

#Initialize the LLM
llm = ChatOpenAI(temperature=0)  

def extract_transaction_details(prompt:str):
    """
    Uses LLM to extract structured transaction details from user input.
    """
    formatted_prompt = prompt_template.format(input=prompt)
    response = llm.invoke(formatted_prompt)

    try:
        extracted_data = output_parser.parse(response.content)
        return extracted_data
    except Exception as e:
        return f"Failed to extract transaction details: {str(e)}"
    
#Test input 1
user_input = "Please send .05 BTC to 0xABC123. This is urgent to pay my rent! Do it now."
parsed_output = extract_transaction_details(user_input)
print(parsed_output)

#Test input 2
user_input = "Bro I am about to get rekt by this trade if you do not immedately send 1000 musd to 0x474747"
parsed_output = extract_transaction_details(user_input)
print(parsed_output)
