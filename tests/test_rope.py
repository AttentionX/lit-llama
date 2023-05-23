import torch

@torch.no_grad()
def test_rope(lit_llama, orig_llama) -> None:
    torch.manual_seed(1)

    # batch_size, sequence_length, number_of_heads, embedding_size
    bs, seq_len, n_head, n_embed = 1, 6, 2, 8
    # make random input
    x = torch.randint(0, 10000, size=(bs, seq_len, n_head, n_embed // n_head)).float()

    # check that the rope caches are same
    # what is RoPE Cache?
    # RoPE (Rotary Positional Embedding) adds positional information to the input by rotating the input tensor proportionally to the index.
    # So, RoPE Cache is the precaculated rotational matrix for RoPE.
    
    # why n_embed // n_head?
    # RoPE is applied to query and key vectors
    freqs_cis = orig_llama.precompute_freqs_cis(n_embed // n_head, seq_len)
    llama_rope_cache = lit_llama.build_rope_cache(seq_len, n_embed // n_head, dtype=x.dtype, device=x.device)
    torch.testing.assert_close(freqs_cis, torch.view_as_complex(llama_rope_cache))

    # check that the inputs with RoPE from both models are same
    llama_x_rope = lit_llama.apply_rope(x.transpose(1, 2), llama_rope_cache).transpose(1, 2)
    # original llama's rope is added to query and key at the same time
    orig_llama_x_rope, _ = orig_llama.apply_rotary_emb(x, x, freqs_cis)
    torch.testing.assert_close(llama_x_rope, orig_llama_x_rope)
