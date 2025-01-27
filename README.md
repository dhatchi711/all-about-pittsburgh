# CMU Advanced NLP Assignment 2: End-to-end NLP System Building


## Steps
1. Run the data scraping using the following command:
```bash
python src/scraping/web_scrape.py

```

2. Generate the data annotations
```bash
python src/data_annotation/meta_api.py
```

3. Run the model finetuning using the following command:
```bash
tune run lora_finetune_single_device --config finetune/8B_lora_single_device.yaml
```

4. Example of how to run the RAG pipeline using the following command:
```bash
python rag_pipeline.py ${model_name} ${retriever_name} ${input query filepath} ${outputf filepath}
```

5. Evaluation Metrics
```bash
python src/rag/eval_matrics.py
```