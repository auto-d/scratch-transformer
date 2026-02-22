import numpy as np 
import argparse

def matmul(a,b): 
    """
    Wrapped around a matrix multiplication to ensure all transformations
    are logged and annotated consistently. 
    
    :param a: Description
    :param b: Description
    """
    pass

def tokenize(a, verbose): 
    """
    Decompose our language input into suitable number of tokens 
    """
    pass 

def detokenize(a, verbose): 
    """
    Map from an integer value back to a token 
    """
    pass 

def embed(a, verbose): 
    """
    Doc1string for embed
    """

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
    in the sequence
    """
    pass

def self_attention(a, verbose): 
    """
    Perform self-attention on the provided matrix, returning the output
    """
    pass 

def softmax(logits, verbose): 
    pass 

def norm(a, verbose): 
    pass 

def fc(a, verbose): 
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
    seq = embed(tokens, verbose)
    
    # Encode the position information and concatenate to ensure token position is 
    # captured and made available to the downstream learning process
    residual = encode_position(seq, verbose)

    # Perform 
    attn = self_attention(residual, verbose)

    # Join residual and attention outputs
    residual = add(residual, attn, verbose)

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

    tokens = tokenize(args.text, args.verbose)   
    probs = forward(args.verbose) 
    
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