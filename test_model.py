import torch

print("Loading model file...")

state = torch.load(
    "models/transit_classifier.pt",
    map_location="cpu"
)

print("Loaded successfully")
print(type(state))