import torch
from model import EntityMatcher
from dataset import get_llm_embedding, normalize_embedding
import numpy as np
# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EntityMatcher().to(device)
model.load_state_dict(torch.load("entity_matcher.pth", map_location=device))  # Load trained weights
model.eval()  # Set model to evaluation mode

def predict(text1, text2):
    """Predict whether two entity names are a match."""
    embedding1 = normalize_embedding(get_llm_embedding(text1))
    embedding2 = normalize_embedding(get_llm_embedding(text2))

    if embedding1.shape[0] != 3072 or embedding2.shape[0] != 3072:
        raise ValueError(f"Unexpected embedding size: {embedding1.shape}, {embedding2.shape}")

    input_vector = np.concatenate((embedding1, embedding2))
    # **Ensure embeddings are concatenated correctly**
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(device)

    print(f"Input vector shape: {input_tensor.shape}")  # Debugging step: should be (1, 6144)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
        confidence, predicted = torch.max(probabilities, 1)

    return "Match" if predicted.item() == 1 else "No Match", confidence.item()

# Example predictions
# ✅ Should Predict "Match"
prediction_samples = [
    ("Apple iPhone 14", "iPhone 14 Pro"),  # Same model, minor variation
    ("Samsung Galaxy S22", "Galaxy S22 Ultra"),  # Same series
    ("Nike Air Max 270", "Nike Air Max 270 Sneakers"),  # Same product line
    ("Dell XPS 13", "Dell XPS 13 9310"),  # Different models but same product
    ("MacBook Air M2", "Apple MacBook Air M2"),  # Brand name variation
    ("Sony WH-1000XM5", "Sony Noise Cancelling Headphones XM5"),  # Same headphones
    ("GoPro Hero 11", "GoPro Hero 11 Black"),  # Same camera, minor variation
    ("JBL Charge 5", "JBL Portable Speaker Charge 5"),  # Speaker model
    ("Microsoft Surface Laptop 4", "Surface Laptop 4"),  # Laptop model
    ("Fitbit Charge 5", "Fitbit Smartwatch Charge 5"),  # Wearable fitness tracker
]

# ❌ Should Predict "No Match"
no_match_samples = [
    ("Apple iPhone 14", "Samsung Galaxy S22"),  # Different brands
    ("Nike Air Max 270", "Adidas Ultraboost"),  # Different shoe brands
    ("Dell XPS 13", "MacBook Pro 16"),  # Different laptop brands
    ("Sony PlayStation 5", "Xbox Series X"),  # Competing gaming consoles
    ("Samsung Galaxy Tab S8", "Apple iPad Air 5"),  # Different tablet brands
    ("Canon EOS R6", "Nikon Z7"),  # Different camera brands
    ("Bose QuietComfort 45", "Sony WH-1000XM5"),  # Competing headphones
    ("GoPro Hero 10", "DJI Osmo Action Camera"),  # Different action cameras
    ("Tesla Model S", "Ford Mustang Mach-E"),  # Different car brands
    ("Microsoft Surface Pro 8", "Samsung Galaxy Book Pro")  # Different laptops
]


for text1, text2 in prediction_samples + no_match_samples:
    result, confidence = predict(text1, text2)
    print(f"Input: '{text1}' vs '{text2}' → Prediction: {result} (Confidence: {confidence:.2f})")
