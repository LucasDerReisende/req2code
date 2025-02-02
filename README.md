# req2code

To get started locally, make sure you have poetry setup and install all required dependecies.

If you want to analyze a Javascript repository, make sure to have a recent version of NodeJS installed and run `npm i`.
Also make sure you import the correct analyzer in `setup_repository.py`, either `analyzer_js` or `analyzer_py`.

Next you want to make sure to install the ollama tool from their official website and pull a desired LLM using the command line tool.

Set the value for the model you want to use in `utils.py`.

If you want to use OpenAI for querying, make sure to store your OPENAI_API key in an `.env` file.

Also make sure to adapt `get_llm_query_result` to your needs (whether you want to query locally or using OPENAI).


## Usage of command line tool

### Setup
To create the data for project setup run the script with the `init` command.

projects and their according directory have to be specified in `main.py.`

`poetry run python main.py /project/ init`

with arguments 

`--analyse --summarize --vectorize-summaries --vectorize-content`.

### Retrieve
To query the RAG use the `retrieve` command.

`retrieve --query "question?"` will query the RAG for files it associates with the question.

`retrieve --stats` will print stats about the data for the specified project.