# Sistema de recomendación con agente langchain

<!-- ABOUT THE PROJECT -->
## Acerca del proyecto
En el presente proyecto muestro la construcción de una sistema de recomendación, el sistema permite recomendar recursos a través del uso de mecanismos de Procesamiento de lenguaje natura como agentes langchain, IA generatica y embeddings.


## Procesos 

* Configuración de ChromaDb - Vector Similarity
* Creación del módulo de generación de embeddings ya sea por openAi o algun otro modelo
* Módulo de recomendación generativo con agentes langchain y llms

<!-- GETTING STARTED -->
## Tecnologías

* Python
* ChromaDB
* Langchain
* Llama
* HugginFace
* pytorch
* OpenAI
* Embeddings
* Mistral
* GPT


## Prerequisitos

* Instalar Python (<a href="https://www.python.org/downloads/">https://www.python.org/downloads/</a>).
* Obtener el API KEY de OpenAI (opcional) (<a href="https://openai.com/">https://openai.com/</a>).

## Inicializacion

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

#### Se convierte a diccionario el dataset
```python
books_dict = books.to_dict(orient='records') 
```
#### Si se cuenta con una api key de openai se agrega en este apartado
```python
openai_api_key = ""
```
#### Se inicializa la instancia el recomendador
```python
recommender = CoreRecommendation(openai_key=openai_api_key)
```
#### De la intancia anterior se crea ruta de persistencia de los recursos
```python
recommender.init_components(collection_name="books",resources=books_dict)
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
print(recommender.get_recommendation(user=user))
```



