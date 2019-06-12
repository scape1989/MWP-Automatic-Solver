from bert_embedding import BertEmbedding

MAX_DECIMAL_PLACES = 6

# Use the pretrained model
bert_embedding = BertEmbedding(model='bert_24_1024_16',
                               dataset_name='book_corpus_wiki_en_cased')


def bert_embedding_from_text(text):
    text_encoding_list = []
    # Get the BERT embeddings from the Google pretrained model
    # Parameter text must be one sentence per line

    # Split the text into sentences for embedding
    # split_text = text.split("\n")

    for encoding in bert_embedding(text):
        if len(encoding[1]) > 0:
            for number in encoding[1][0].tolist():
                shortened_representation = f"%.{MAX_DECIMAL_PLACES}f" % number
                text_encoding_list.append(float(shortened_representation))

    # Use the Google pretrained model to retreive embeddings
    return text_encoding_list


if __name__ == "__main__":
    # Run a quick demo for understanding of the BERT encodings
    print("This is the BERT embeddings demo.")

    demo_text = "hello this is a test .\n"

    print(f"\nFrom the following text body:\n{demo_text}\n\nThe result is:")

    result = bert_embedding_from_text(demo_text)

    print(result)
