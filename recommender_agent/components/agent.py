from langchain.prompts import StringPromptTemplate
from typing import List
from langchain.agents import Tool, AgentExecutor, create_react_agent


TEMPLATE = """
As an AI agent specializing in recommendation systems, my primary task is to interact with a vector database. Based on the input that is user JSON i have to follow specific rules, I proceed through the entire JSON body to identify elements such as keywords or characteristics that can be utilized for personalized recommendations.
Upon identifying an element, I initiate a semantic search using it as input, extracting relevant information from the database. To maintain consistency, I ensure a uniform format and structure for each keyword throughout the process.
After retrieving responses, I evaluate them by comparing the information with the user's preferences, aiming to determine the best recommendation. To avoid redundancy, I meticulously track previously suggested items and exclude them from the final list of recommendations.
Once the JSON body has been thoroughly and completely examined, I compile a list of the best top five recommendations and present them in a consistent format. Importantly, I conclude the process without overthinking the results, providing the user with a tailored and refined set of suggestions based on their preferences.

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
6.Termination: Once you have traversed completely the JSON body, your final task is to compile a list of the best top five recommendations, conclude the process. Prioritize efficiency over exhaustive analysis to provide timely and relevant suggestions.

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

   

class RecommenderAgent():
    def __init__(self,vectordb,llm):
        self.vectordb = vectordb
        self.llm = llm             

    def initAgent(self):
        tools = [
            Tool(name = "Query",description = "Useful when you need to use semantic search in a vectorstore", func = self.query)
        ]   
        prompt = CustomPromptTemplate(
            template=TEMPLATE,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables = ["input", "intermediate_steps",'tools', 'tool_names', 'agent_scratchpad']
        )
        agent = create_react_agent(
            self.llm,
            tools,
            prompt
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    def executeAgent(self,string):
        try:
            response = self.agent_executor.invoke({"input": string})
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        return response
    
    def query(self, query: str):
        response = self.vectordb.similarity_search(query=query,k=5)
        return [x.metadata for x in response]