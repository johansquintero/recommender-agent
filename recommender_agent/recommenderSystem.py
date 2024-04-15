from langchain.embeddings import HuggingFaceEmbeddings
from recommender_agent.components.vectorStoreClient import Client
from recommender_agent.components.agent import RecommenderAgent
from langchain.chat_models import ChatOpenAI

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
ENCODE_KWARGS = {'normalize_embeddings': True} # set True to compute cosine similarity
MODEL_KWARGS = {'device': 'cpu'}

HUGGINFACE_MODEL_NAME = "TheBloke/Mistral-7B-OpenOrca-GGUF"
HUGGINFACE_BASENAME = "mistral-7b-openorca.Q4_K_M.gguf"

OPENAI_MODEL_NAME = "gpt-3.5-turbo-1106"


class CoreRecommendation:
    def __init__(self,openai_key='') -> None:
        self.openai_key=openai_key
        self.embeddingModel = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs=ENCODE_KWARGS,
            model_kwargs=MODEL_KWARGS
        )
        if(openai_key!=''):
            print(f"model name = {OPENAI_MODEL_NAME}")
            self.llm = ChatOpenAI(temperature=0.3, model_name=OPENAI_MODEL_NAME,api_key = openai_key)
        else:
            from langchain.callbacks.manager import CallbackManager
            from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
            from langchain.llms import LlamaCpp
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(repo_id=HUGGINFACE_MODEL_NAME, filename=HUGGINFACE_BASENAME)
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            self.llm = LlamaCpp(
                model_path=model_path,
                callback_manager=callback_manager,
                verbose=True, # Verbose is required to pass to the callback manager
                temperature=0.0,
                n_batch=2,
                n_ctx=8000#15000
            )
            

    def init_components(self,collection_name:str,resources:list[dict]):
        self.resourcesClient = Client(resources=resources,collection_name=collection_name)
        vectordb = self.resourcesClient.create_collection(embedding_model=self.embeddingModel)
        self.agent = RecommenderAgent(vectordb=vectordb,llm=self.llm)
        self.agent.initAgent()
    
    def get_recommendation(self,user):
        text = self.get_text_for_user(user=user)
        response = self.agent.executeAgent(string=text)
        return tuple((user, response))
    
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

  