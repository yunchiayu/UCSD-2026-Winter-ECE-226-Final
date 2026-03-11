import torch
def get_fake_prompt(tokenizer, batch_size: int, seq_len: int, device: str = "cuda"):
    # avoid special tokens (usually <10 ids)
    vocab_size = tokenizer.vocab_size

    min_token_id = 10
    max_token_id = vocab_size - 1

    input_ids = torch.randint(
        min_token_id,
        max_token_id,
        (batch_size, seq_len),
        device=device,
    )

    attention_mask = torch.ones_like(input_ids)


    return input_ids, attention_mask



