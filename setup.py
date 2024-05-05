from setuptools import setup, find_packages

with open("README.md", "r",encoding="utf-8") as d:
    description = d.read()

setup(
    name='recommender-agent',
    version='0.3.0',
    license= 'MIT',
    description="En el presente proyecto muestro la construcción de una sistema de recomendación, el sistema permite recomendar recursos a través del uso de mecanismos de Procesamiento de lenguaje natura como agentes langchain, IA generatica y embeddings.",
    long_description= description,
    long_description_content_type="text/markdown",
    author="Johan sebastian quintero rojas",
    install_requires= [
        'numpy==1.25.2',
        'pandas==2.0.3',
        'typer==0.9.4',
        'scipy==1.11.4',
        'langchain==0.1.17',
        'langchain-community==0.0.36',
        'chromadb==0.5.0',
        'openai==1.25.1',
        'langchain_openai==0.1.6',
        'groq==0.5.0',
        'langchain-groq==0.1.3',
        'sentence-transformers==2.7.0',
        'torch==2.3.0'
    ],
    author_email="metalium144@gmail.com",
    packages=find_packages(),
    url='https://github.com/johansquintero/recommender-agent'
)