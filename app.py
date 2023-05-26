from flask import Flask, render_template, request

import os
import io

from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

from pprint import pprint

app = Flask(__name__)

def index_documents(directory):
    document_store = InMemoryDocumentStore(use_bm25=True)
    files_to_index = [directory + "/" + file for file in os.listdir(directory)]
    indexing_pipeline = TextIndexingPipeline(document_store=document_store)
    indexing_pipeline.run_batch(file_paths=files_to_index)

    return indexing_pipeline, document_store


def search(query):
    indexing_pipeline, document_store = index_documents('../data/bulletins')
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
    retriever = BM25Retriever(document_store=document_store)
    pipe = ExtractiveQAPipeline(reader, retriever)

    prediction = pipe.run(
                            query=query, 
                            params={
                                "Retriever": {"top_k": 1}, # pick single document
                                "Reader": {"top_k": 1} # only one answer
                            }
                        )
    
    result = [
        prediction['answers'][0].answer,
        prediction['answers'][0].score,
        prediction['answers'][0].context
        ]
    
    return result


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        results = search(query)
        return render_template('results.html', results=results)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)