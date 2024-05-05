from langchain_community.embeddings import HuggingFaceEmbeddings
from recommender_agent.components.vectorStoreClient import Client
from recommender_agent.components.agent import RecommenderAgent
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import torch

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
ENCODE_KWARGS = {'normalize_embeddings': True} # set True to compute cosine similarity
MODEL_KWARGS = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}#si esta cuda habilitado se crearan los embeddings con este

HUGGINFACE_MODEL_NAME = "TheBloke/Mistral-7B-OpenOrca-GGUF"
HUGGINFACE_BASENAME = "mistral-7b-openorca.Q4_K_M.gguf"

OPENAI_MODEL_NAME = "gpt-3.5-turbo-1106"

GROQ_MODEL_NAME = "llama3-70b-8192"

class CoreRecommendation:
    def __init__(self,openai_key='',groq_key='') -> None:
        self.openai_key=openai_key
        self.embeddingModel = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs=ENCODE_KWARGS,
            model_kwargs=MODEL_KWARGS
        )
        if(openai_key!=''):
            print(f"OpenAi model name = {OPENAI_MODEL_NAME}")
            self.llm = ChatOpenAI(temperature=0.3, model_name=OPENAI_MODEL_NAME,api_key = openai_key)
        if(groq_key!=''):
            print(f"Groq model name = {GROQ_MODEL_NAME}")
            self.llm = ChatGroq(temperature=0.5, groq_api_key=groq_key, model_name=GROQ_MODEL_NAME)
            

    def init_components(self,collection_name:str,resources:list[dict]):
        self.resourcesClient = Client(resources=resources,collection_name=collection_name)
        vectordb = self.resourcesClient.create_collection(embedding_model=self.embeddingModel)
        self.agent = RecommenderAgent(vectordb=vectordb,llm=self.llm)
        self.agent.initAgent()
    
    def get_recommendation(self,user):
        text = self.get_text_for_user(user=user)
        response = self.agent.executeAgent(string=text)
        
        user_rec = {
            "user":user, 
            "rec":response['output']
        }
        return user_rec
    
    def get_recommendatios(self,users):
        responses = []
        for user in users:
            response = self.get_recommendation(user=user)
            responses.append(response)
        return responses


    def get_text_for_user(self,user:dict):
        text = "{\n"
        for (key,value) in user.items():
            text+=f"{key}: {value}\n"
        text+="}\n"
        return text
    
    def deleteCollection(self):
        self.resourcesClient.delete_collection()

  