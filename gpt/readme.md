<h2>Bigram Model</h2>

- **Batch size:** 65  

- **Block size / Context window:** 256  

- **Layers:** 6  

- **Heads:** 6  

- **Vector dimension:** 65  

- **Learning rate:** 0.0003  
  - On such data, attention does not perform well with a high learning rate  

- **Dropout:** 0.2  

- **Max iterations:** 5k  

- **Evaluation interval:** 500  

- **Evaluation iterations:** 200  


<h2>GPT Model</h2>

*(Not the same — different number of layers and hyperparameters)*

- **Batch size:** 32  

- **Block size / Context window:** 8  

- **Layers:** 1  

- **Heads:** 1  
  - Self-attention  

- **Vector dimension:** 32  

- **Learning rate:** 0.001  

- **Max iterations:** 3k  

- **Evaluation iterations:** 300  


<h2>Dataset</h2>

- Shakespeare  


<h2>Notes</h2>

- Do not run the GPT model unless you have a good GPU.  

- GPT model: ~10M corpus.  

- Bigram model can run on CPU.  


<h2>Credit</h2>

- Andrej Karpathy  
