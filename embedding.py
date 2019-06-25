import numpy as np
from keras_transformer import get_model, decode

source_tokens = [
    'i need more power'.split(' '),
    'eat jujube and pill'.split(' '),
]
target_tokens = [
    list('我要更多的抛瓦'),
    list('吃枣💊'),
]

# Generate dictionaries


def build_token_dict(token_list):
    token_dict = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
    }
    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    return token_dict


source_token_dict = build_token_dict(source_tokens)
target_token_dict = build_token_dict(target_tokens)
target_token_dict_inv = {v: k for k, v in target_token_dict.items()}


# Add special tokens
encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]

# Padding
source_max_len = max(map(len, encode_tokens))
target_max_len = max(map(len, decode_tokens))

encode_tokens = [tokens + ['<PAD>'] *
                 (source_max_len - len(tokens)) for tokens in encode_tokens]
decode_tokens = [tokens + ['<PAD>'] *
                 (target_max_len - len(tokens)) for tokens in decode_tokens]
output_tokens = [tokens + ['<PAD>'] *
                 (target_max_len - len(tokens)) for tokens in output_tokens]


encode_input = [list(map(lambda x: source_token_dict[x], tokens))
                for tokens in encode_tokens]
decode_input = [list(map(lambda x: target_token_dict[x], tokens))
                for tokens in decode_tokens]
decode_output = [list(map(lambda x: [target_token_dict[x]], tokens))
                 for tokens in output_tokens]
print(decode_input)
exit()
