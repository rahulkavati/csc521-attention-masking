# CSC 521 — Attention Masking in Transformer Models

**Team:** Rahul Kavati · Vignesh Vijayalayan · Vinit Lad  
**Course:** CS 521 · Final Presentation  
**Paper:** Attention Is All You Need — Vaswani et al., 2017

## Topic
Controls which tokens are allowed to attend to 
which other tokens inside the Transformer attention mechanism.

## Repository Used
[hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)  
A clean PyTorch reimplementation of the original paper.
We used this as our reference implementation to understand 
how masking is structured internally.

## Our Code
We wrote our own analysis notebook to isolate and visualize 
the masking mechanism independently:

| Cell | What it does |
|------|-------------|
| Cell 2 | Generates causal mask tensor using torch.tril() |
| Cell 3 | Three-panel heatmap — encoder, decoder, cross-attention |
| Cell 4 | Shows masked_fill operation step by step |
| Cell 5 | Attention weights heatmap with exact values |

## Results
- `attention_masks.png` — three mask types side by side
- `attention_weights.png` — before/after softmax heatmap

## How to Run
1. Open `attention_masking_csc521.ipynb` in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells in order
4. PNG files will be saved automatically

## Demo Video
[Watch 4-minute Loom demo](https://www.loom.com/share/24519be43ba346499532e6d1bfae62ad)
