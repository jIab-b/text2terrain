import argparse
import os
from together import Together

DEFAULT_MODEL = "sammac446_f3cf/Meta-Llama-3.1-8B-Instruct-Reference-4c9c0b0c"

def main():
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned LoRA model")
    parser.add_argument("--prompt", required=True, help="Input prompt text")
    parser.add_argument("--model", default=os.getenv("TOGETHER_MODEL", DEFAULT_MODEL), help="Model name (defaults to fine-tuned adapter)")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum tokens to generate")
    args = parser.parse_args()

    client = Together()

    resp = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a terrain generator. Reply ONLY with the JSON object for the generate_heightmap function call and nothing else. The object MUST include: tile_u, tile_v, world_x, world_y, global_seed, module_ids, parameters, and seeds (one seed per module)."},
            # One-shot example from training data
            {"role": "user", "content": "sharp moderate winding valleys with weathering gently curved"},
            {"role": "assistant", "content": "{\"tile_u\":-3,\"tile_v\":-6,\"world_x\":-768,\"world_y\":-1536,\"global_seed\":0,\"module_ids\":[0,2,3],\"parameters\":{\"frequency\":0.04,\"octaves\":6,\"persistence\":0.5,\"lacunarity\":2.0,\"ridge_sharpness\":0.0,\"warp_amplitude\":140.0,\"warp_frequency\":0.005,\"warp_octaves\":2,\"iterations\":200,\"rain_amount\":0.6,\"evaporation\":0.05,\"capacity\":0.3},\"seeds\":[0,1013,2026]}"},
            {"role": "user", "content": args.prompt}
        ],
        max_tokens=args.max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
    )

    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
