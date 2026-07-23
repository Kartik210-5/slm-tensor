from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained(
    "google-t5/t5-small",
    use_fast=False
)

tokenizer.save_pretrained("./t5_tokenizer")