import numpy as np 
import argparse
import torch 
from transformers import AutoTokenizer, AutoModel



def log_if(s, condition): 
    if condition: 
        print(s)

def dot(a,b): 
    return np.dot(a,b) 

def matmul(a,b): 
    """
    Wrapped around a matrix multiplication to ensure all transformations
    are logged and annotated consistently. 
    
    :param a: Description
    :param b: Description
    """
    return np.matmul(a,b)

def tokenize(a, verbose): 
    """
    Decompose our language input into suitable number of tokens 
    """
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    ids = tokenizer.encode(a) 

    log_if(f" ▶ Tokenizing {len(a)} characters ", verbose) 

    literals = [x for x in zip(ids[0:5], tokenizer.convert_ids_to_tokens(ids[0:5]))]
    log_if(f"  → Resulting sequence is {len(ids)} tokens (first five ids/tokens: {literals[0:5]})", verbose)

    return ids

def detokenize(a, verbose): 
    """
    Map from an integer value back to a token 
    """
    pass 

def embed(ids, verbose): 
    """
    Leverage pre-trained word embeddings to map our tokens into a high-D space
    """
    
    log_if(f" ▶ Embedding sequence of {len(ids)} characters.. ", verbose) 

    model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
    embeddings = model.get_input_embeddings()
    sequence = [embeddings.weight.H[:,x].detach().numpy().copy() for x in ids]

    log_if(f"  → Embedded {len(ids)}", verbose)
    log_if(f"  → New sequence is size {len(sequence)}, {len(embeddings.weight[0])}", verbose)

    return np.array(sequence)

def unembed(a, verbose): 
    """
    Implement a shallow linear layer to project from d_model to d_vocab
    """
    pass

def add(a, b, verbose): 
    """
    Add to matrices together 
    """

def encode_position(a, verbose):
    """
    Encode the order of each element and add it to the respective token representation
    in the sequence (which must be a numpy array)
    """
            
    log_if(f" ▶ Encoding position into sequence of {len(a)}... ", verbose) 

    # Load the BERT model and leverage its fixed positional encodings, note this 
    # has a limit 512 encodings by default, which our toy shouldn't exceed
    if len(a) > 512: 
        raise ValueError("Input sequence exceeds positional encoding vector size!")

    model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

    log_if(f"  → Retrieving fixed position embeddings...", verbose)

    # Extract a suitable number of encodings and add to our source vector, 
    # detaching from the pytorch graph to avoid any gradient baggage
    ix = torch.tensor(range(0, len(a)), dtype=torch.int32)
    position = model.embeddings.position_embeddings(ix)
    position = position.detach().numpy().copy()

    log_if(f"  → Adding absolute positional encoding ({position.shape}) to sequence ({a.shape}) ", verbose)

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
    
    log_if(f" ▶ Applying self attentiong to residual matrix (r@{r.shape})", verbose)
    log_if(f"  → Projecting residual through Q, K (q@{weights[Q].shape} k@{weights[K].shape})", verbose)

    # E.g. (n, 768) * (768, 64) -> (n, 64)
    q = matmul(r, weights[Q])
    k = matmul(r, weights[K])
    log_if(f"  → q@{q.shape}, k@{k.shape}", verbose)

    # E.g. (n, 64) (n, 64)T -> (64, 64)
    s = matmul(q, k.T)
    log_if(f"  → Computed raw attention scores on q,k (s@{s.shape})", verbose)
    
    # Now softmax those scores to give us a normalized factor (actually a probability 
    # simplex - all values sum to 1) that preserves relative order and is also trivially
    # differentiable. 
    a = softmax(s)

    # Scale our v vector, amplifying or suppressing by the corresponding score,  then 
    # add the results to get our new residual stream z
    log_if(f"  → Scaling v@{v.shape} by attention scores...", verbose)
    v = matmul(r, weights[V])

    log_if(f"  → Attention head output  z@{z.shape}", verbose)
    z = v * a

    return z 
    
def softmax(logits, verbose=True): 
    """
    Poor-man's softmax that presumes a 2d matrix with the values to operate on in the last 
    dimension. 
    
    :param logits: Description
    :param verbose: Description
    """
    if len(logits.shape) != 2: 
        raise ValueError(f"Expected 2d matrix, got {len(logits.shape)}")
    
    out = np.zeros(logits.shape)
    for i in range(0, logits.shape[0]): 
        e_x = np.e ** logits[i]
        out[i] = e_x/np.sum(e_x)
    
    return out

def norm(a, verbose): 
    pass 

def fc(a, verbose=True): 
    pass 

def forward(tokens, verbose=False):
    """
    Forward pass of our toy transformer

    The 
    Embedding 

    Pos encoding 

    Attention 

    Normalization 

    Skip connection  

    FC layer  

    Logits  

    TODO: include where we would replicate transformer blocks 
    TODO: include where we would scale out attention heads
    """
    
    # Map tokens into a richer space, imparting learned semantics on the way 
    # Note we could learn the embeddings here from scratch, bootstrap with an existing 
    # network or just do a lookup on embeddings using prior results.
    seq = embed(tokens, verbose)
    
    # Our model dimension is the embedding dimension, we could project up or down, but 
    # unclear what value this would add, up would sparsify semantics and down would compress
    d_model = seq.shape[1]
    
    # Encode the position information and concatenate to ensure token position is 
    # captured and made available to the downstream learning process
    residual = encode_position(seq, verbose)

    # Initialize random weights to emulate learned weights
    # NOTE: The convention has become multiple attention heads to improve the richness
    # of learned relationships and mappings. For our toy we have just a single 
    # attention head of d_model size
    attn_weights = np.random.rand(3, d_model, d_model)

    z = self_attention(attn_weights, residual, verbose)

    # Join residual and attention outputs
    residual = add(residual, z, verbose)

    # Normalize
    residual = norm(residual, verbose)

    # Pass through a fully-connected deep network 
    residual = fc(residual, verbose)
    
    # Unembedding/projection to vocabulary 
    # - Regardless of how many transformer blocks we have elected to stack (here just one), we
    #   must project from the residual stream (d = d_model) to the model's vocabulary. 
    # - We implement above projection with a shallow linear layer and no fancy activations 
    logits = unembed(residual, verbose) 

    # Armed with a pile of logits, we compress the associated scalars into the [0,1] range which 
    # will herd the resulting values toward real token probabilities that are chasing our loss 
    # function and hopefully represent a close approximation of the real conditional probs in natural 
    # language  
    probs = softmax(logits, verbose)
    return probs

def main(): 
    """
    """
    parser = argparse.ArgumentParser(description="Toy implementation of a transformer architecture.")     
    parser.add_argument("--verbose", action="store_true", default=True, help="Log all operations to aid understanding")
    parser.add_argument("--dimension", type=int, default=16, help="Model dimension")
    parser.add_argument("text", type=str, help="An input string to process")
    args = parser.parse_args()

    log_if(f"Starting transformer emulator on sequence '{args.text[0:10]}' ...", args.verbose)

    tokens = tokenize(args.text, args.verbose)   
    probs = forward(tokens, args.verbose) 
    
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