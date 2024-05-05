from langchain_community.vectorstores import Chroma
import os
import time

class Client:
    def __init__(self, resources:list[dict],collection_name):
        self.resources = resources      
        self.collection_name=collection_name

    """ 
        Evaluador de existencia de la colleccion en la base de datos vectorial
    """
    def has_data(self):
        ruta = f"recommender_agent\persists\{self.collection_name}"
        return os.path.exists(ruta)
    
   
    """
        metodo para formatear un recurso a texto
    """
    def get_text_for_resource(self,resource:dict):
        text = ""
        for (key,value) in resource.items():
            text+=f"{key}: {value}\n"
        return text

    """
    def get_documents_for_resources(self):
        docs = []
        for resource in self.resources:
            doc =  Document(page_content=self.get_text_for_product(resource), metadata={'source':self.get_text_for_product(resource)})
            docs.append(doc)
        return docs
    """

    def get_texts_for_resources(self):
        texts = []
        for resource in self.resources:
            text =  self.get_text_for_resource(resource)
            texts.append(text)
        return texts       
    
    def create_collection(self,embedding_model):
        path = f"recommender_agent\persists\{self.collection_name}"
        if(self.has_data()):
            print("Embeddings already existis, persisting the path...")
            self.vectordb = Chroma(persist_directory = path, collection_name=self.collection_name, embedding_function= embedding_model)
        else:
            texts = self.get_texts_for_resources()
            ids = [str(resourse['id']) for resourse in self.resources]
            start = time.time()
            print('creating embeddings...')
            self.vectordb = Chroma.from_texts(texts=texts,persist_directory = path, metadatas=self.resources,embedding = embedding_model, collection_name=self.collection_name, ids = ids)
            end = time.time()
            print(f'Embeddings completed! total time={end-start}')
        return self.vectordb
    
    def delete_collection(self):
        path = f"recommender_agent\persists\{self.collection_name}"
        self.delete_path_r(path=path)
        
    def delete_path_r(self,path):
        # Verificar si la ruta existe
        if not os.path.exists(path):
            print(f"La ruta {path} no existe.")
            return

        # Verificar si la ruta es un archivo
        if os.path.isfile(path):
            os.remove(path)
            print(f"Archivo {path} eliminado.")
            return

        # Si la ruta es un directorio, eliminar recursivamente su contenido
        for elemento in os.listdir(path):
            # Obtener la ruta completa del elemento
            path_element = os.path.join(path, elemento)
            # Recursivamente eliminar el elemento
            self.delete_path_r(path_element)

        # Eliminar el directorio vacío
        os.rmdir(path)
        print(f"Directorio {path} eliminado.")