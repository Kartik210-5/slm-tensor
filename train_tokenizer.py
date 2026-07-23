import sentencepiece as spm

input_file = "dataset/corpus.txt"
model_prefix = "slm_tokenizer"
vocab_size = 16000

spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type="bpe",
    character_coverage=1.0,

    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,

    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<bos>",
    eos_piece="<eos>",

    normalization_rule_name="nmt_nfkc",
    input_sentence_size=1000000,
    shuffle_input_sentence=True
)


print("Tokenizer training complete!")