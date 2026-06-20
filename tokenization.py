import tiktoken 

encoder = tiktoken.encoding_for_model('gpt-4o')

print(f"Vocabulary size: {encoder.n_vocab}") #1,00,019 200K+

text = "The cat sat on the mat."

# Encode the text into tokens
tokens = encoder.encode(text)
print(f"Tokens: {tokens}") #[976, 9059, 10139, 402, 290, 2450, 13]

tokens = [976, 9059, 10139, 402, 290, 2450, 13]
# Decode the tokens back into text
decoded_text = encoder.decode(tokens)
print(f"Decoded text: {decoded_text}") #The cat sat on the mat.