from llama_index.readers.json import JSONReader

json_reader= JSONReader(
    levels_back=0,
    collapse_length=100,
    clean_json=True
)