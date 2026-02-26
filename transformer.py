# Toy transformer w/ excessive logging to help explore underlying mechanisms. 
# NOTE: Support from https://jalammar.github.io/illustrated-transformer/ on 
# architecture. Support frm chatGPT5.3 for syntax (called out where applicable) 
# and Q&A on transformer fundamental concepts (e.g. layernorm, positional encoding, ...)

import os 
import logging
import numpy as np 
import argparse
import torch 

# Silence spurious load warnings, must occur before import; syntax courtesy of Gemini
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import AutoTokenizer, AutoModel

bert_model_string = "google-bert/bert-base-uncased"

def log_if(s, condition): 
    if condition: 
        print(s)

def tokenize(a, verbose): 
    """
    Decompose our language input into suitable number of tokens 
    """
    log_if(f" ▶ Tokenizing {len(a)} characters ", verbose) 

    tokenizer = AutoTokenizer.from_pretrained(bert_model_string, local_files_only=True)
    
    ids = tokenizer.encode(a) 

    literals = [x for x in zip(ids[0:5], tokenizer.convert_ids_to_tokens(ids[0:5]))]
    log_if(f"  → Resulting sequence is {len(ids)} tokens (first five ids/tokens: {literals[0:5]})", verbose)

    return ids, len(tokenizer._vocab)

def detokenize(id, verbose): 
    """
    Map from an integer value back to a token 
    """
    log_if(f" ▶ Mapping ID {id} back to token...", verbose) 
    tokenizer = AutoTokenizer.from_pretrained(bert_model_string, local_files_only=True) 
    token = tokenizer.decode(id) 
    log_if(f"  → Resulting token is {token}", verbose)

    return token 

def embed(ids, verbose): 
    """
    Leverage pre-trained word embeddings to map our tokens into a high-D space
    """
    
    log_if(f" ▶ Embedding sequence of {len(ids)} characters.. ", verbose) 

    log_if(f"  → Loading donor model ({bert_model_string})", verbose) 
    model = AutoModel.from_pretrained(bert_model_string, local_files_only=True)
    embeddings = model.get_input_embeddings()

    # Accumulate the embeddings for each token ID 
    sequence = [embeddings.weight.H[:,x].detach().numpy().copy() for x in ids]

    log_if(f"  → Embedded {len(ids)}", verbose)
    log_if(f"  → New sequence is size {len(sequence)}, {len(embeddings.weight[0])}", verbose)

    return np.array(sequence)

def unembed(a, verbose): 
    """
    Implement a shallow linear layer to project from d_model to d_vocab
    """
    pass

def encode_position(a, verbose):
    """
    Encode the order of each element and add it to the respective token representation
    in the sequence (which must be a numpy array)
    """
            
    log_if(f" ▶ Encoding position into sequence of length {len(a)}... ", verbose) 

    # Load the BERT model and leverage its fixed positional encodings, note this 
    # has a limit 512 encodings by default, which our toy shouldn't exceed
    if len(a) > 512: 
        raise ValueError("Input sequence exceeds positional encoding vector size!")

    log_if(f"  → Loading donor model ({bert_model_string})", verbose) 
    model = AutoModel.from_pretrained(bert_model_string, local_files_only=True)

    log_if(f"  → Retrieving fixed position embeddings...", verbose)

    # Extract a suitable number of position encodings and add to our source vector, 
    # detaching from the pytorch graph to avoid any gradient baggage
    ix = torch.tensor(range(0, len(a)), dtype=torch.int32)
    position = model.embeddings.position_embeddings(ix)
    position = position.detach().numpy().copy()

    log_if(f"  → Adding absolute positional encoding @{position.shape}) to sequence @{a.shape} ", verbose)

    return a + position 

def self_attention(weights, r, verbose, Q=0, K=1, V=2): 
    """
    Perform self-attention on the provided sequence using the given weights for Q, K, V 
    projections, returning the output
    """
    
    # For every head, project through our Q, K weights to get to d_head, then
    # apply the dot product to arrive at a similarity measure... this scalar tells us 
    # the degree to which our projections are aligned in d_head space. Here we implement
    # this as a matrix multiplication (which is just a composition of dot products on 
    # corresponding vectors)
    
    log_if(f" ▶ Applying self attention to residual matrix r@{r.shape}", verbose)
    log_if(f"  → Projecting residual through Q, K - q@{weights[Q].shape} k@{weights[K].shape}", verbose)

    # E.g. (n, 768) * (768, d_k) -> (n, d_k)
    q = np.matmul(r, weights[Q])
    k = np.matmul(r, weights[K])
    v = np.matmul(r, weights[V])
    d_k = weights[Q].shape[1] 

    log_if(f"  → q@{q.shape}, k@{k.shape}, v@{v.shape}", verbose)

    # Compute attention, yield scores for every token, from every token's perspective 
    # (square matrix). So (n, d_k) * (n, d_k).T = (n,n)
    s = np.matmul(q, k.T) 
    log_if(f"  → Computed raw attention scores on q,k - s@{s.shape}", verbose)
    
    # Now softmax those scores to give us a normalized factor (actually a probability 
    # simplex - all values sum to 1) that preserves relative order and is also trivially
    # differentiable. Scale the dot product scalars by the square root of the head 
    # size to avoid large positive or negative inputs that push softmax to the 
    
    log_if(f"  → Scale by √d_head ...", verbose)
    a = softmax(s/d_k)

    # Mask all attention to the right of the query token. 
    # NOTE: Triu syntax to build our triangle causality mask courtesy of ChatGPT5.3
    mask = np.triu(np.ones((a.shape[0],a.shape[0])), k=0)
    a = a * mask.T
    log_if(f"  → Masked out attention scores on future tokens", verbose)

    # Scale our v vector, amplifying or suppressing by the corresponding score,  then 
    # add the results to get our new residual stream z
    log_if(f"  → Scaling v@{v.shape} by attention scores...", verbose)
    z = np.matmul(a, v)
    log_if(f"  → Attention head output  z@{z.shape}", verbose)

    return z 
    
def softmax(a, verbose=True): 
    """
    Poor-man's softmax (eˣ/Σeˣ) that presumes a 2d matrix with the values to operate on in the last 
    dimension. 
    
    :param logits: Description
    :param verbose: Description
    """
    if len(a.shape) != 2: 
        raise ValueError(f"Expected 2d matrix, got {len(a.shape)}")
    
    out = np.zeros(a.shape)
    for i in range(0, a.shape[0]): 
        e_x = np.e ** a[i]
        out[i] = e_x/np.sum(e_x)
    
    log_if(f"  → Softmax (eˣ/Σeˣ) applied to residuals@{a.shape}", verbose)

    return out

def rms_norm(a, verbose): 
    """
    Apply an RMS-based normalization operation to the provided embeddings/residuals
    """
    epsilon = 1e-6

    rms = np.sqrt(np.sum(np.square(a), axis=1))/a.shape[1]
    rms = rms.reshape(-1,1)
    rmsnorm = a/(rms + epsilon)

    log_if(f"  → RMSNorm applied to residuals@{a.shape}", verbose)

    return rmsnorm

def fc(a, weights, verbose=True): 
    """
    Apply the provided array of weight matrices to a provided residual
    """    
    log_if(f" ▶ Applying fully connected layers to residual matrix r@{a.shape}", verbose)
    log_if(f"  → Projecting r@{a.shape} through FC layer one@{weights[0].shape}", verbose)  
    x = np.matmul(a, weights[0])
    
    log_if(f"  → Applying ReLU activation to r@{a.shape}", verbose)
    x = np.maximum(0, x)

    log_if(f"  → Projecting a@{a.shape} through FC layer two @{weights[1].shape}", verbose)    
    x = np.matmul(x, weights[1])
    
    return x 

def forward(ids, vocab_size, verbose=False):
    """
    Forward pass of our toy transformer
    
    We're ignoring a host of things here, not the least of which are 
    actual learned weights, multiple attention heads and multiple transformer blocks. This is 
    in service of understanding so it feels defensible to rationalize these deficiences away. 
    That said, the toy implementation does all the basic manipulations we would expect
    to find in a basic decoder-onyl transformer, and can log those out if the verbose 
    flag is set.
    """
    
    # Map tokens into a richer space, imparting learned semantics on the way 
    # Note we could learn the embeddings here from scratch, bootstrap with an existing 
    # network or just do a lookup on embeddings using prior results.
    seq = embed(ids, verbose)
    
    # Our model dimension is the embedding dimension, we could project up or down, but 
    # unclear what value this would add, up would sparsify semantics and down would compress
    d_model = seq.shape[1]
    
    # Encode the position information and concatenate to ensure token position is 
    # captured and made available to the downstream learning process
    residual = encode_position(seq, verbose)

    # Initialize random weights to emulate learned weights
    # NOTE: The convention has become multiple attention heads to improve the richness
    # of learned relationships and mappings. For our toy we have just a single 
    # attention head of d_model size, one each for Q, K and V matrices
    attn_weights = np.random.rand(3, d_model, d_model)

    z = self_attention(attn_weights, residual, verbose)

    # Model our skip connection, adding residual stream and attention outputs, 
    # before normalizing to a fixed scale w/ RMSNorm.
    residual = rms_norm(residual + z, verbose)

    # Initialize a random set of weights to stand in for trained weights in our 
    # fully-connected layers, then pass the residual stream for each token through
    fc_weights = [np.random.rand(d_model, d_model*4)]
    fc_weights.append(np.random.rand(d_model*4, d_model))
    fc_residual = fc(residual, fc_weights, verbose)
    
    # Second skip connection - add the attention residual to the MLP output and normalize
    residual = rms_norm(residual + fc_residual, verbose)

    # Unembedding/projection to vocabulary 
    # - Regardless of how many transformer blocks we have elected to stack (here just one), we
    #   must project from the residual stream (d = d_model) to the token vocabulary 
    # - This is purely a linear projection vocab space
    vocab_projection = np.random.rand(d_model, vocab_size)
    logits = np.matmul(residual, vocab_projection)

    # Armed with a pile of logits, we compress the associated scalars into the [0,1] range which 
    # will herd the resulting values toward real token probabilities that are chasing our loss 
    # function and hopefully represent a close approximation of the real conditional probs in natural 
    # language  
    probs = softmax(logits, verbose)
    return probs

def main(): 
    """
    Main entrypoint and arg handling
    """
    parser = argparse.ArgumentParser(description="Toy implementation of a transformer architecture.")     
    parser.add_argument("--verbose", action="store_true", default=True, help="Log all operations to aid understanding")
    parser.add_argument("--dimension", type=int, default=16, help="Model dimension")
    parser.add_argument("text", type=str, help="An input string to process")
    args = parser.parse_args()   

    log_if(f"Starting transformer emulator on sequence '{args.text[0:10]}' ...", args.verbose)

    ids, vocab_size = tokenize(args.text, args.verbose)   
    probs = forward(ids, vocab_size, args.verbose) 
    
    # Greedily grab the highest probability token and report it 
    # Note: This is where we could introduce important features like... 
    # - Sampling: gives us a more authentic variation in selected tokens but might risk some gibberish 
    # - Top-k sampling: constraints our sampling operation to the top K probs, but might admit some
    #   low probability tokens or exclude relatively high probability due to the sharp cutoff @ K
    # - Top-p/nucleus sampling: Variable number of probs based on a target cumulative probability, perhaps
    #   does a better job of excluding braindead tokens while expanding options
    # - Beam search : we would have to roll out multiple tokens through associated forward passes and then 
    #   pick the higher cumulative probability path
    next_token = detokenize(np.argmax(probs), args.verbose) 
    print(next_token)

if __name__ == "__main__":
    main() 