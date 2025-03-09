import torch
import os
from safetensors.torch import save_file

def convert_lora_pth_to_safetensors(input_path, output_path):
    # Load the .pth file
    print(f"Loading LoRA from {input_path}")
    state_dict = torch.load(input_path, map_location="cpu")
    
    # Create a new state dict with updated keys
    new_state_dict = {}
    
    for old_key, tensor in state_dict.items():
        # Example transformation:
        # From: double_blocks.6.img_attn.proj.lora.lora_A
        # To: diffusion_model.double_blocks.6.img_attn.proj.lora_A.weight
        
        if "lora.lora_A" in old_key:
            # Replace "lora.lora_A" with "lora_A.weight"
            new_key = old_key.replace("lora.lora_A", "lora_A.weight")
        elif "lora.lora_B" in old_key:
            # Replace "lora.lora_B" with "lora_B.weight"
            new_key = old_key.replace("lora.lora_B", "lora_B.weight")
        else:
            # For other keys, just add ".weight" if it's a tensor
            new_key = old_key + ".weight"
        
        # Prepend "diffusion_model." to all keys
        new_key = "diffusion_model." + new_key
        
        print(f"Mapping: {old_key} -> {new_key}")
        new_state_dict[new_key] = tensor
    
    # Save as safetensors
    print(f"Saving converted LoRA to {output_path}")
    save_file(new_state_dict, output_path)
    
    print("Conversion completed successfully!")
    return new_state_dict

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert LoRA .pth file to safetensors with updated key pattern")
    parser.add_argument("--input", type=str, required=True, help="Path to input .pth file")
    parser.add_argument("--output", type=str, help="Path to output .safetensors file (default: same as input but with .safetensors extension)")
    
    args = parser.parse_args()
    
    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        # Replace .pth extension with .safetensors
        output_path = os.path.splitext(input_path)[0] + ".safetensors"
    
    convert_lora_pth_to_safetensors(input_path, output_path)