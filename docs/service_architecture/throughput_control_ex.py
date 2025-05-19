import time
import random
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# -------------------------------------------------------------------
# 1. Load or Initialize Your Model and Tokenizer
#    (In audio scenarios, replace with your streaming audio model)
# -------------------------------------------------------------------
model_name = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# -------------------------------------------------------------------
# 2. Throughput Measurement (Stub)
#    In reality, you'll get this info from your networking stack,
#    an edge server’s metrics, or a custom function.
# -------------------------------------------------------------------
def measure_throughput():
    """
    Returns an integer or float indicating current network throughput,
    e.g., in Kbps or Mbps, or a latency measure in ms.
    This is a stub that randomly simulates throughput changes.
    """
    return random.randint(100, 1000)  # Simulated throughput between 100–1000 Kbps

# -------------------------------------------------------------------
# 3. Dynamic Token Generation
#    - For demonstration, we generate text tokens using GPT-2.
#    - For audio, you'd have a function generate_audio_tokens(...) that
#      returns the next chunk of audio tokens.
# -------------------------------------------------------------------
def generate_next_tokens(prompt_ids, max_new_tokens=1):
    """
    Generates a small batch of tokens (text example).
    For audio-based models, replace with your TTS or ASR inference logic.
    """
    # Use the `generate` method with your chosen number of new tokens
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    # Only return the newly generated tokens, not the entire sequence
    new_token_ids = output_ids[0, -max_new_tokens:]
    return new_token_ids


def dynamic_stream_generation(
    initial_prompt: str,
    low_latency_batch_size: int = 1,
    high_throughput_batch_size: int = 6,
    throughput_threshold: int = 500,
    max_total_tokens: int = 50
):
    """
    Continuously generate tokens using either a small batch size
    (for low-latency) or a larger batch size (for high-throughput),
    based on real-time throughput measurements.
    """
    # Tokenize initial prompt
    prompt_ids = tokenizer.encode(initial_prompt, return_tensors="pt")

    generated_tokens = []
    total_generated = 0

    while total_generated < max_total_tokens:
        # Check throughput (could be done asynchronously or in intervals)
        current_throughput = measure_throughput()
        
        # Decide batch size based on throughput
        if current_throughput < throughput_threshold:
            # Low throughput detected → use larger batch to reduce round-trips
            batch_size = high_throughput_batch_size
        else:
            # High throughput or stable → generate fewer tokens at a time
            batch_size = low_latency_batch_size
        
        # Generate tokens
        new_token_ids = generate_next_tokens(prompt_ids, max_new_tokens=batch_size)
        
        # Append the new tokens to the ongoing sequence
        prompt_ids = tokenizer.encode(
            tokenizer.decode(new_token_ids), add_special_tokens=False, return_tensors="pt",
        )
        generated_tokens.extend(new_token_ids.tolist())
        
        # Convert to string (for demonstration)
        current_output_str = tokenizer.decode(new_token_ids)
        total_generated += batch_size

        # Stream or “send” the partial output
        # In a real app, you might transmit these tokens over WebSocket, gRPC, etc.
        print(f"[Throughput={current_throughput}][Batch={batch_size}] Generated: {current_output_str}")
        
        # Artificial small sleep to emulate streaming intervals
        time.sleep(0.2)

    # Return the full generated text
    return tokenizer.decode(generated_tokens)

# -------------------------------------------------------------------
# 4. Run the Dynamic Streaming Generation
# -------------------------------------------------------------------
if __name__ == "__main__":
    prompt = "Once upon a time in a land far, far away, there was a great"
    
    final_output = dynamic_stream_generation(
        initial_prompt=prompt,
        low_latency_batch_size=2,          # small # of tokens in "fast" mode
        high_throughput_batch_size=5,      # bigger # of tokens in "slow" mode
        throughput_threshold=600,          # threshold to switch
        max_total_tokens=40                # total tokens to generate
    )
    
    print("\n=== Final Generated Output ===")
    print(final_output)
