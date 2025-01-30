import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import EntityMatchingDataset
from model import EntityMatcher

import random

def evaluate(model, data_loader):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return accuracy, precision, recall, f1

match_pairs = [
    ("Apple iPhone 14", "iPhone 14 Pro"),
    ("Samsung Galaxy S22", "Galaxy S22 Ultra"),
    ("Nike Air Max 270", "Nike Air Max 270 Sneakers"),
    ("MacBook Air M2", "Apple MacBook Air M2"),
    ("GoPro Hero 10", "GoPro Hero 10 Action Camera"),
    ("Sony WH-1000XM5", "Sony Noise Cancelling Headphones XM5"),
    ("Bose QuietComfort 45", "Bose QC 45 Headphones"),
    ("JBL Flip 6", "JBL Flip 6 Bluetooth Speaker"),
    ("Microsoft Surface Pro 8", "Surface Pro 8 Tablet"),
    ("Canon EOS R5", "Canon R5 Camera"),
    ("Google Pixel 7", "Google Pixel 7 Pro"),
    ("Sony PlayStation 5", "PS5"),
    ("Apple MacBook Pro 16", "MacBook Pro M1 Max 16-inch"),
    ("Samsung Galaxy Watch 4", "Galaxy Watch 4 Classic"),
    ("Dell XPS 13", "Dell XPS 13 9310"),
    ("Lenovo ThinkPad X1 Carbon", "ThinkPad X1 Carbon Gen 9"),
    ("Adidas Ultraboost 22", "Adidas Running Shoes Ultraboost 22"),
    ("Dyson V11 Vacuum", "Dyson V11 Cordless Vacuum Cleaner"),
    ("Tesla Model S", "Tesla Model S Plaid"),
    ("Xbox Series X", "Microsoft Xbox Series X"),
    ("Nintendo Switch OLED", "Nintendo Switch OLED Model"),
    ("Razer BlackWidow Keyboard", "Razer BlackWidow Elite Mechanical Keyboard"),
    ("HP Spectre x360", "HP Laptop Spectre x360"),
    ("Garmin Fenix 7", "Garmin Smartwatch Fenix 7"),
    ("Oculus Quest 2", "Meta Quest 2 VR Headset"),
] * 10  # Repeat list to reach 250

# ❌ 250 No-Match Pairs (Completely Different Products)
no_match_pairs = [
    ("Apple iPhone 14", "Samsung Galaxy S22"),
    ("Nike Air Max 270", "Adidas Ultraboost 22"),
    ("Dell XPS 13", "MacBook Pro 16"),
    ("Sony PlayStation 5", "Xbox Series X"),
    ("Samsung Galaxy Watch 4", "Apple Watch Series 7"),
    ("Google Pixel 7", "Samsung Galaxy S22"),
    ("HP Spectre x360", "Lenovo ThinkPad X1 Carbon"),
    ("Tesla Model S", "Ford Mustang Mach-E"),
    ("Bose QuietComfort 45", "Sony WH-1000XM5"),
    ("Garmin Fenix 7", "Fitbit Versa 3"),
    ("JBL Flip 6", "Bose SoundLink Mini"),
    ("Microsoft Surface Pro 8", "Samsung Galaxy Tab S8"),
    ("Oculus Quest 2", "Valve Index VR Headset"),
    ("GoPro Hero 10", "DJI Osmo Action Camera"),
    ("Razer BlackWidow Keyboard", "Corsair K95 RGB Mechanical Keyboard"),
    ("Samsung Galaxy Tab S7", "Amazon Fire HD 10"),
    ("Canon EOS R5", "Nikon Z7 Camera"),
    ("Dyson V11 Vacuum", "Shark Rocket Cordless Vacuum"),
    ("Apple MacBook Pro 16", "Microsoft Surface Laptop 4"),
    ("Google Nest Mini", "Amazon Echo Dot 4th Gen"),
    ("Fitbit Charge 5", "Apple Watch SE"),
    ("Sony Bravia OLED TV", "LG C1 OLED TV"),
    ("Samsung QLED 4K TV", "Vizio P-Series Quantum TV"),
    ("Lenovo Yoga 7i", "HP Pavilion x360"),
    ("Bose SoundLink Revolve", "UE Boom 3"),
] * 10  # Repeat list to reach 250

# Shuffle dataset to mix match and no-match pairs
random.shuffle(match_pairs)
random.shuffle(no_match_pairs)

# Combine and create labels
pairs = match_pairs + no_match_pairs
labels = [1] * 250 + [0] * 250  # 1 for Match, 0 for No Match

# Shuffle final dataset
combined = list(zip(pairs, labels))
random.shuffle(combined)
pairs, labels = zip(*combined)

pairs = list(pairs)  # Convert tuple to list
labels = list(labels)  # Convert tuple to lis


# Split into training and validation sets
train_pairs, temp_pairs, train_labels, temp_labels = train_test_split(
    pairs, labels, test_size=0.3, random_state=42
)
val_pairs, test_pairs, val_labels, test_labels = train_test_split(
    temp_pairs, temp_labels, test_size=0.5, random_state=42
)

# Create dataset and dataloader
train_dataset = EntityMatchingDataset(train_pairs, train_labels)
val_dataset = EntityMatchingDataset(val_pairs, val_labels)
test_dataset = EntityMatchingDataset(test_pairs, test_labels)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EntityMatcher().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluate on validation set
    val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Accuracy: {val_acc:.2f}, F1-score: {val_f1:.2f}")

# 📌 Evaluate on Test Set
test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc:.2f}, Precision: {test_prec:.2f}, Recall: {test_rec:.2f}, F1-score: {test_f1:.2f}")

# Save trained model
torch.save(model.state_dict(), "entity_matcher.pth")
print("Model saved successfully!")




