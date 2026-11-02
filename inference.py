
import json
import re
from pathlib import Path

import torch
import spacy

from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data, Batch

from model import ImprovedRelationExtractionRGCN


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

DEVICE = torch.device("cpu")

BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128


# ==================================================
# 2. LOAD VOCABULARIES AND CONFIGURATION
# ==================================================

def load_json(filename):

    path = MODEL_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


pos_vocab = load_json("pos_vocab.json")
dependency_vocab = load_json("dependency_vocab.json")
relation_labels = load_json("relation_labels.json")
config = load_json("model_config.json")


# ==================================================
# 3. LOAD SPACY
# ==================================================

nlp = spacy.load("en_core_web_sm")


# ==================================================
# 4. LOAD BERT TOKENIZER AND MODEL
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(
    BERT_MODEL_NAME,
    use_fast=True
)

bert_model = AutoModel.from_pretrained(
    BERT_MODEL_NAME
).to(DEVICE)

bert_model.eval()

for parameter in bert_model.parameters():
    parameter.requires_grad = False


# ==================================================
# 5. LOAD TRAINED R-GCN MODEL
# ==================================================

model = ImprovedRelationExtractionRGCN(
    num_pos=len(pos_vocab),
    num_relations=config["num_relations"],
    num_classes=config["num_classes"],
    pos_embedding_dim=config["pos_embedding_dimension"],
    hidden_dim=config["hidden_dimension"],
    dropout=config["dropout"]
).to(DEVICE)

checkpoint_path = MODEL_DIR / "final_rgcn_model.pt"

state_dict = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict, strict=True)

model.eval()


# ==================================================
# 6. FIND ENTITY CHARACTER POSITIONS
# ==================================================

def find_entity_span(sentence, entity):

    entity = entity.strip()

    if not entity:
        raise ValueError("Entity cannot be empty.")

    matches = list(
        re.finditer(
            re.escape(entity),
            sentence,
            flags=re.IGNORECASE
        )
    )

    if len(matches) == 0:
        raise ValueError(
            f"Entity '{entity}' was not found in the sentence."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Entity '{entity}' occurs multiple times. "
            "Please provide an unambiguous entity mention."
        )

    return matches[0].start(), matches[0].end()


# ==================================================
# 7. BUILD DEPENDENCY GRAPH
# ==================================================

def build_inference_graph(sentence, entity_1, entity_2):

    e1_start, e1_end = find_entity_span(
        sentence, entity_1
    )

    e2_start, e2_end = find_entity_span(
        sentence, entity_2
    )

    # Check entity overlap
    if max(e1_start, e2_start) < min(e1_end, e2_end):
        raise ValueError(
            "Entity 1 and Entity 2 cannot overlap."
        )

    doc = nlp(sentence)

    if len(doc) == 0:
        raise ValueError("The sentence is empty.")

    # ----------------------------------------------
    # POS FEATURES
    # ----------------------------------------------

    node_features = []

    for token in doc:

        pos_id = pos_vocab.get(
            token.pos_,
            pos_vocab["<UNK>"]
        )

        node_features.append([pos_id])

    x = torch.tensor(
        node_features,
        dtype=torch.long
    )

    # ----------------------------------------------
    # ENTITY NODE IDENTIFICATION
    # ----------------------------------------------

    entity_1_nodes = [
        token.i
        for token in doc
        if token.idx < e1_end
        and token.idx + len(token.text) > e1_start
    ]

    entity_2_nodes = [
        token.i
        for token in doc
        if token.idx < e2_end
        and token.idx + len(token.text) > e2_start
    ]

    if not entity_1_nodes:
        raise ValueError(
            "Entity 1 could not be aligned with spaCy tokens."
        )

    if not entity_2_nodes:
        raise ValueError(
            "Entity 2 could not be aligned with spaCy tokens."
        )

    # ----------------------------------------------
    # DEPENDENCY EDGES
    # ----------------------------------------------

    edge_list = []
    edge_types = []

    for token in doc:

        if token.head != token:

            source = token.i
            target = token.head.i

            forward_relation = token.dep_
            reverse_relation = token.dep_ + "_reverse"

            forward_id = dependency_vocab.get(
                forward_relation,
                dependency_vocab["<UNK>"]
            )

            reverse_id = dependency_vocab.get(
                reverse_relation,
                dependency_vocab["<UNK>"]
            )

            edge_list.append([source, target])
            edge_types.append(forward_id)

            edge_list.append([target, source])
            edge_types.append(reverse_id)

    # ----------------------------------------------
    # CREATE EDGE TENSORS
    # ----------------------------------------------

    if edge_list:

        edge_index = torch.tensor(
            edge_list,
            dtype=torch.long
        ).t().contiguous()

        edge_type = torch.tensor(
            edge_types,
            dtype=torch.long
        )

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

        edge_type = torch.empty(
            (0,),
            dtype=torch.long
        )

    # ----------------------------------------------
    # ENTITY MASKS
    # ----------------------------------------------

    entity_1_mask = torch.zeros(
        len(doc),
        dtype=torch.bool
    )

    entity_2_mask = torch.zeros(
        len(doc),
        dtype=torch.bool
    )

    entity_1_mask[entity_1_nodes] = True
    entity_2_mask[entity_2_nodes] = True

    # ----------------------------------------------
    # CREATE PYTORCH GEOMETRIC DATA
    # ----------------------------------------------

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        entity_1_mask=entity_1_mask,
        entity_2_mask=entity_2_mask
    )

    # Store character offsets for BERT alignment
    graph.token_offsets = [
        (token.idx, token.idx + len(token.text))
        for token in doc
    ]

    graph.token_texts = [
        token.text for token in doc
    ]

    return graph


# ==================================================
# 8. GENERATE BERT EMBEDDINGS
# ==================================================

def generate_bert_embeddings(sentence, graph):

    encoded = tokenizer(
        [sentence],
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    offsets = encoded.pop("offset_mapping")

    inputs = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.inference_mode():

        outputs = bert_model(**inputs)

    hidden_states = outputs.last_hidden_state.cpu()

    word_embeddings = []

    # Align BERT subwords with spaCy tokens
    for word_start, word_end in graph.token_offsets:

        matching_indices = []

        for j, (sub_start, sub_end) in enumerate(
            offsets[0].tolist()
        ):

            if sub_end > sub_start:

                if (
                    sub_start < word_end
                    and sub_end > word_start
                ):
                    matching_indices.append(j)

        if not matching_indices:

            raise ValueError(
                f"No BERT token found for "
                f"'{sentence[word_start:word_end]}'. "
                "The sentence may exceed BERT's 128-token limit."
            )

        # Average all matching subword embeddings
        embedding = hidden_states[
            0, matching_indices, :
        ].mean(dim=0)

        word_embeddings.append(embedding)

    graph.bert_x = torch.stack(word_embeddings)

    return graph


# ==================================================
# 9. PREDICT RELATION
# ==================================================

def predict_relation(sentence, entity_1, entity_2):

    if not sentence.strip():
        raise ValueError("Please enter a sentence.")

    # Build dependency graph
    graph = build_inference_graph(
        sentence,
        entity_1,
        entity_2
    )

    # Generate BERT embeddings
    graph = generate_bert_embeddings(
        sentence,
        graph
    )

    # Convert single graph into a PyG Batch
    # This provides data.num_graphs = 1
    graph_batch = Batch.from_data_list([graph])

    graph_batch = graph_batch.to(DEVICE)

    # Model prediction
    with torch.inference_mode():

        logits = model(graph_batch)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    class_id = predicted_class.item()

    return {
        "relation": relation_labels[str(class_id)],
        "class_id": class_id,
        "confidence": confidence.item(),
        "probabilities": probabilities[0].tolist()
    }


print("Inference pipeline loaded successfully!")
