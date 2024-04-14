from langchain.prompts import StringPromptTemplate
from typing import List, Union
from langchain.agents import Tool, AgentOutputParser,LLMSingleActionAgent,AgentExecutor
from langchain.schema import AgentAction, AgentFinish,OutputParserException
from langchain.chains import LLMChain
import re


TEMPLATE = """
As an AI agent specializing in recommendation systems, my primary task is to interact with a vector database. Based on the input that is user JSON i have to follow specific rules, I proceed through the entire JSON body to identify elements such as keywords or characteristics that can be utilized for personalized recommendations.
Upon identifying an element, I initiate a semantic search using it as input, extracting relevant information from the database. To maintain consistency, I ensure a uniform format and structure for each keyword throughout the process.
After retrieving responses, I evaluate them by comparing the information with the user's preferences, aiming to determine the best recommendation. To avoid redundancy, I meticulously track previously suggested items and exclude them from the final list of recommendations.
Once the JSON body has been thoroughly examined, I compile a list of the best top five recommendations and present them in a consistent format. Importantly, I conclude the process without overthinking the results, providing the user with a tailored and refined set of suggestions based on their input.

I have access to the following tools:

{tools}

Use the following format:git

JSON: the input that you have to check
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: Once you are confident in your recommendations, share your conclusion
Final Answer: Provide the top five list of the recommended items based on the original user input query

Begin! Commence your recommendations, and always respond as a recommendation system when you are delivering your final answer

JSON: {input}
{agent_scratchpad}
"""

TEMPLATE2 = """
As an AI agent specializing in recommendation systems, your primary task is to navigate through a whole user's JSON body to identify key elements such as keywords or characteristics that can be leveraged for personalized recommendations. Follow the specified guidelines:

1.Elements Identification: Scrutinize the entire JSON body to detect and select relevant elements indicative of user preferences or characteristics suitable for recommendations.
2.Semantic Search: Utilize the identified elements as inputs for semantic searches, exploring the database for related items.
3.Evaluate Responses: Assess the search results, comparing them with user information. Determine the most fitting response as a personalized recommendation.
4.Avoid Repetition: Implement a mechanism to prevent the recommendation of items already suggested to the user.
5.Consistent Format: Maintain a uniform format and structure for presenting recommendations associated with each identified keyword throughout the process.
6.Termination: Once you have traversed the JSON body and compiled a list of the best top five recommendations, conclude the process. Prioritize efficiency over exhaustive analysis to provide timely and relevant suggestions.

By adhering to these guidelines, ensure a seamless and effective interaction with the vector database, delivering a concise and well-organized set of personalized recommendations.

you have access to the following tools:

{tools}

Use the following format:

JSON: the input that you have to check
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: Once you are confident in your recommendations, share your conclusion
Final Answer: Provide list of the recommended items based on the original user input query

Begin! Commence your recommendations, and always respond as a recommendation system when you are delivering your final answer

JSON: {input}
{agent_scratchpad}
"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    # Define una nueva clase llamada CustomOutputParser que hereda de AgentOutputParser
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Define un método llamado 'parse' que toma la salida del modelo de lenguaje como entrada
        # y devuelve una instancia de AgentAction o AgentFinish

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            # Comprueba si la cadena "Final Answer:" está en la salida del modelo
            # Si es así, crea una instancia de AgentFinish
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},  # Extrae el valor de la salida y lo almacena en 'return_values'
                log=llm_output,  # Almacena la salida original en 'log'
            )

        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"  # Define una expresión regular para extraer acciones y entradas de acción
        match = re.search(regex, llm_output, re.DOTALL)  # Busca la expresión regular en la salida del modelo
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")  # Lanza una excepción si no se puede analizar la salida

        action = match.group(1).strip()  # Extrae la acción y la limpia
        action_input = match.group(2)  # Extrae la entrada de la acción

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        # Crea una instancia de AgentAction con la acción y la entrada de la acción extraídas,
        # y almacena la salida original en 'log'
    

class RecommenderAgent():
    def __init__(self,vectordb,llm):
        self.vectordb = vectordb
        self.llm = llm             

    def initAgent(self):
        tools = [
            Tool(name = "Query",description = "Useful when you need to use semantic search in a vectorstore", func = self.query)
        ]   
        prompt = CustomPromptTemplate(
            template=TEMPLATE2,
            tools=tools,
            input_variables = ["input", "intermediate_steps"]
        )
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser = CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=tool_names,
            max_iterations=10,
            handle_parsing_errors=True
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                    tools=tools,
                                                    verbose=True,
                                                    handle_parsing_errors=True)
    
    def executeAgent(self,string):
        try:
            response = self.agent_executor.run(string)
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        return response
    
    def query(self, query: str):
        response = self.vectordb.similarity_search(query=query,k=5)
        return [x.metadata for x in response]