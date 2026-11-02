
import json
import torch

from model import ImprovedRelationExtractionRGCN

MODEL_DIR = "model"

# Load configuration
with open(f"{MODEL_DIR}/model_config.json", "r") as f:
    config = json.load(f)

# Load POS vocabulary
with open(f"{MODEL_DIR}/pos_vocab.json", "r") as f:
    pos_vocab = json.load(f)

# Initialize model using saved configuration
model = ImprovedRelationExtractionRGCN(
    num_pos=len(pos_vocab),
    num_relations=config["num_relations"],
    num_classes=config["num_classes"],
    pos_embedding_dim=config["pos_embedding_dimension"],
    hidden_dim=config["hidden_dimension"],
    dropout=config["dropout"]
)

# Load trained weights
state_dict = torch.load(
    f"{MODEL_DIR}/final_rgcn_model.pt",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict, strict=True)

model.eval()

print("Checkpoint loaded successfully!")
print("Model:", config["model"])
print("Number of classes:", config["num_classes"])
print("POS vocabulary:", len(pos_vocab))
print("Dependency relations:", config["num_relations"])
print("Device: CPU")
