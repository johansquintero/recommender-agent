# Sistema de recomendación con agente langchain

## Acerca del proyecto
En el presente proyecto muestro la construcción de una sistema de recomendación, el sistema permite recomendar recursos a través del uso de mecanismos de Procesamiento de lenguaje natura como agentes langchain, IA generatica y embeddings.


## Procesos 

* Configuración y creacion del modulo de la base de datos vectorial con ChromaDb
* Creación del módulo y configuracion del agente Langchain
* Módulo principal de recomendación el cual gestiona e integrar los dos anteriores modulos

## Tecnologías

* Python
* ChromaDB
* Langchain
* Llama
* HugginFace
* pytorch
* OpenAI
* Groq
* Embeddings
* LLM


## Prerequisitos
* Instalar Python (<a href="https://www.python.org/downloads/">https://www.python.org/downloads/</a>).
* Obtener una API KEY de OpenAI (opcional si se va a utilizar Groq) (<a href="https://openai.com/">https://openai.com/</a>).
* Obtener una API KEY de groq (opcional si se va a utilizar OpenAI) (<a href="https://console.groq.com/keys">https://console.groq.com/keys</a>)

## Instalacion manual desde el repositorio

* Clonar el repositorio
  ```sh
  git clone https://github.com/johansquintero/recommender-agent
  ```

* Crear el entorno virtual
  ```sh
  python -m venv env
  ```
* Activar entorno virtual (windows):
  ```sh
  env\Scripts\activate
  ```
* Instalar las dependencias del proyecto:
  ```sh
  pip install -r requirements.txt
  ```

## Instalando con pip
```sh
pip install recommender-agent
```
# Ejemplo de implementacion del sistema
#### Imports 
```python
from recommender_agent.recommenderSystem import CoreRecommendation
import pandas as pd
```
#### Se importa el dataset de la ruta donde este ubicado
```python
books = pd.read_csv("../dataset_books/Books2.csv")
books = books.fillna('')
books.drop(columns=["isbn13","thumbnail","subtitle"],axis=1,inplace=True)
```
#### Un requisito es que cada uno de los elementos tengan un id
```python
books["id"] = books.index + 1
```
#### Se convierte a diccionario el dataset
```python
books_dict = books.to_dict(orient='records') 
```
### la libreria dispone de dos alternativas para aplicar un LLM uno por la api de OpenAI y el otro por la api de Groq, cabe recalcar que es obligatorio contar con alguna de estas 2 apis

#### Si se cuenta con una api key de openai este seria el proceso
```python
openai_api_key = ""
```
#### Se inicializa la instancia el recomendador
```python
recommender_open_ai = CoreRecommendation(openai_key=openai_api_key)
```
#### De la intancia anterior se crea ruta de persistencia de los recursos
```python
recommender_open_ai.init_components(collection_name="books",resources=books_dict)
```
#### El el caso de usar groq este es el metodo
```python
groq_api_key=""
recommender_groq = CoreRecommendation(groq_key=groq_api_key)
recommender_groq.init_components(collection_name="books",resources=books_dict)
```
#### Usuario de prueba en formato JSON
```python
user = {
  "user": {
    "id": 234567,
    "name": "Carlos Rodriguez",
    "email": "carlos_rodriguez@email.com",
    "address": {
      "street": "Calle Principal",
      "city": "Barcelona",
      "state": "Spain",
      "postal_code": "08001"
    },
    "preferences": {
      "genres": ["Philosophy","Novels", "History"],
      "favorite_authors": ["Dan Brown", "Ken Follett", "René Descartes"]
    },
    "purchase_history": [
      {
        "book": "The Da Vinci Code",
        "author": "Dan Brown",
        "price": 17.99,
        "purchase_date": "2023-12-12"
      },
      {
        "book": "The Pillars of the Earth",
        "author": "Ken Follett",
        "price": 22.99,
        "purchase_date": "2024-01-05"
      }
    ]
  }
}
```
#### Se ejecuta el recomendador con la informacion del usuario
```python
response = recommender_groq.get_recommendation(user=user)
print(f"Usuario={response['user']} \n") 
print(f"Recomendaciones= {response['rec']}")
```



