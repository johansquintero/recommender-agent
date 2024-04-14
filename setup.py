from setuptools import setup, find_packages

setup(
    name='recommender-agent',
    version='0.0.1',
    license= 'MIT',
    description="En el presente proyecto muestro la construcción de una sistema de recomendación, el sistema permite recomendar recursos a través del uso de mecanismos de Procesamiento de lenguaje natura como agentes langchain, IA generatica y embeddings.",
    long_description= open('README.md').read(),
    author="Johan sebastian quintero rojas",
    install_requires= [
        'numpy<1.26.0',
        'langchain==0.0.334',
        'chromadb==0.4.17',
        'requests==2.31.0',
        'huggingface_hub==0.16.4',
        'scipy==1.9.3',
        'pandas<2.0.0',
        'sentence-transformers==2.2.2',
        'openai<1.0.0',
        'llama-cpp-python==0.2.11',
    ],
    author_email="metalium144@gmail.com",
    packages=find_packages(),
    url='https://github.com/johansquintero/recommender-agent'
)