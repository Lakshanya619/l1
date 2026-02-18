from huggingface_hub import InferenceClient
from datetime import datetime
from PIL import Image
from config import HF_API_KEY  # make sure file name is correct

MODELS = [
    "ByteDance/SDXL-Lightning",
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-xl-base-1.0"
]

client = InferenceClient(api_key=HF_API_KEY)

print(f"Primary model: {MODELS[0]}")
print("Type 'quit' to exit\n")

while True:
    prompt = input("Enter prompt: ").strip()

    if prompt.lower() in ["quit", "exit", "q"]:
        break

    if not prompt:
        continue

    print("Generating...")
    image = None

    for model in MODELS:
        try:
            image = client.text_to_image(
                prompt=prompt,
                model=model
            )
            print(f"Generated using {model}")
            break

        except Exception as e:
            print(f"{model} failed, trying next...")
            print(e)

    if image is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        image.save(filename)
        print(f"Saved: {filename}")

        image.show()
        print()
    else:
        print("Error: All models failed. Check your API key.\n")

print("Goodbye!")


