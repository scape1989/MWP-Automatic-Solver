from bert_embedding import BertEmbedding

# Use the pretrained model
bert_embedding = BertEmbedding(model='bert_24_1024_16',
                               dataset_name='book_corpus_wiki_en_cased')


def bert_embedding_from_text(text):
    # Get the BERT embeddings from the Google pretrained model
    # Parameter text must be one sentence per line
    # Split the text into sentences for embedding
    split_text = text.split("\n")

    # Use the Google pretrained model to retreive embeddings
    return bert_embedding(split_text)


def main():
    # Run a quick demo for understanding of the BERT encodings
    print("This is the BERT embeddings demo.")

    demo_text = """We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.
Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. 
BERT is conceptually simple and empirically powerful. 
It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7 (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5% absolute improvement), outperforming human performance by 2.0%."""

    print(f"\nFrom the following text body:\n{demo_text}\n\nThe result is:")

    result = bert_embedding_from_text(demo_text)

    print(result)


if __name__ == "__main__":
    main()
