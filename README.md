# Relation-Extraction-RGCN
BERT-enhanced R-GCN for semantic relation extraction

# Relation Extraction using Enhanced R-GCN

An NLP-based relation extraction system that identifies semantic relationships between two entities in a sentence using **BERT and Relational Graph Convolutional Networks (R-GCN)**.

The project combines contextual language representations from BERT with syntactic dependency information represented as graphs to classify relationships between entity pairs.

## Project Overview

Relation Extraction is an important Natural Language Processing (NLP) task that identifies semantic relationships between entities mentioned in a sentence.

For example:

**Input Sentence:**
> The storm caused damage to the building.

**Entity 1:** storm

**Entity 2:** damage

**Predicted Relation:** Cause-Effect(e1,e2)

This project implements an enhanced R-GCN architecture to classify entity relationships into 19 classes using the SemEval-2010 Task 8 dataset.

## Key Features

- BERT-based contextual word representations.
- Dependency graph construction using spaCy.
- Relational Graph Convolutional Network for graph-based learning.
- Entity-aware feature extraction.
- Classification into 19 semantic relation classes.
- Prediction confidence scores and probability distribution.
- Interactive Streamlit web application.
- CPU-based inference support.

## Model Architecture

The proposed system combines three major components:

1. **BERT (Bidirectional Encoder Representations from Transformers)**

   - Uses `bert-base-uncased`.
   - Extracts contextual representations for sentence tokens.
   - BERT parameters remain frozen during R-GCN training.
   - BERT subword embeddings are aligned with spaCy tokens using character offsets and mean pooling.

2. **Dependency Graph Construction**

   - Uses spaCy's `en_core_web_sm` model.
   - Each sentence token represents a graph node.
   - Part-of-speech (POS) tags are used as node features.
   - Dependency relationships form directed graph edges.
   - Reverse dependency edges are added to capture bidirectional information.

3. **Enhanced R-GCN**

   - Two R-GCN convolutional layers.
   - 256-dimensional hidden representations.
   - 30 relational bases.
   - POS embeddings of dimension 32.
   - Dropout of 0.3.

   The final entity-pair representation combines:

   - Entity 1 representation
   - Entity 2 representation
   - Absolute difference between entity representations
   - Element-wise product of entity representations

   These features are concatenated and passed through a classification network to predict one of 19 relation classes.

### Architecture Workflow

```text
Input Sentence
      |
      v
Entity Identification
      |
      v
BERT Contextual Embeddings
      |
      v
spaCy Dependency Parsing
      |
      v
Dependency Graph Construction
      |
      v
POS Embeddings + BERT Features
      |
      v
Enhanced R-GCN
      |
      v
Entity Pair Representation
      |
      v
Fully Connected Classifier
      |
      v
19 Relation Classes
      |
      v
Predicted Relation + Confidence
```

## Dataset

**Dataset:** SemEval-2010 Task 8

The dataset is used for semantic relation classification between two marked entities in a sentence.

| Dataset split | Number of samples |
|---|---:|
| Training | 7,200 |
| Validation | 800 |
| Testing | 2,717 |
| **Total** | **10,717** |

The task contains 19 classes:

- 9 semantic relation types, each represented in two directional forms.
- 1 additional `Other` class.

The directional labels distinguish the relationship between Entity 1 and Entity 2.

## Model Configuration

| Parameter | Configuration |
|---|---|
| Pretrained model | BERT-base-uncased |
| BERT embedding dimension | 768 |
| POS embedding dimension | 32 |
| Combined input dimension | 800 |
| R-GCN layers | 2 |
| Hidden dimension | 256 |
| Number of dependency relations | 89 |
| Number of relation classes | 19 |
| Number of bases | 30 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Best epoch | 14 |

## Experimental Results

The enhanced model was compared with the original R-GCN architecture.

| Model | Test Accuracy | Macro F1-score |
|---|---:|---:|
| Original R-GCN | 80.64% | 77.27% |
| Enhanced R-GCN | **81.78%** | **78.53%** |

### Performance Improvement

- Accuracy improvement: approximately 1.14 percentage points.
- Macro F1-score improvement: approximately 1.26 percentage points.

The enhanced entity-pair representation improved both accuracy and macro F1-score in the reported comparison.

**Evaluation note:** The test results were also used during model comparison and selection. They should therefore be interpreted as reported experimental results rather than an untouched final holdout estimate.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| PyTorch Geometric | Graph neural network implementation |
| Hugging Face Transformers | BERT model and tokenizer |
| spaCy | Dependency parsing and POS tagging |
| NumPy | Numerical operations |
| Pandas | Data processing |
| Streamlit | Interactive web application |
| Scikit-learn | Evaluation metrics |

## Project Structure

```text
Relation-Extraction-RGCN/
│
├── model/
│   ├── final_rgcn_model.pt
│   ├── model_config.json
│   ├── pos_vocab.json
│   ├── dependency_vocab.json
│   └── relation_labels.json
│
├── app.py
├── inference.py
├── model.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Important Files

- `app.py`: Streamlit application interface.
- `inference.py`: Builds dependency graphs, generates BERT embeddings, and predicts relations.
- `model.py`: Defines the enhanced R-GCN architecture.
- `final_rgcn_model.pt`: Trained model checkpoint.
- `model_config.json`: Model configuration.
- `pos_vocab.json`: POS tag vocabulary.
- `dependency_vocab.json`: Dependency relation vocabulary.
- `relation_labels.json`: Relation class mappings.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Khyathiyadav/Relation-Extraction-RGCN.git
```

### 2. Navigate to the project directory

```bash
cd Relation-Extraction-RGCN
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

## How to Use the Application

1. Open the Streamlit application.
2. Enter a sentence containing two entities.
3. Enter Entity 1 and Entity 2 exactly as they appear in the sentence.
4. Click **Predict Relation**.
5. View the predicted semantic relationship.
6. Explore the model confidence score and probability distribution across all 19 classes.

### Example

**Sentence:**

```text
The bicycle has two wheels.
```

**Entity 1:** bicycle

**Entity 2:** wheels

**Predicted Relation:**

```text
Component-Whole(e2,e1)
```

The predicted relationship indicates that the wheels are components of the bicycle.

## Deployment

The application is deployed using Streamlit Community Cloud.

**GitHub Repository:**

https://github.com/Khyathiyadav/Relation-Extraction-RGCN

**Deployment:** Streamlit Community Cloud.

## Limitations

- The model is trained on the SemEval-2010 Task 8 dataset and may not generalize equally well to every domain.
- Rare relation classes can be difficult to predict due to limited training examples.
- The `Other` class can be confused with specific semantic relations.
- Softmax confidence scores are not calibrated probabilities and do not guarantee prediction correctness.
- BERT inference requires computational resources and may be slower on CPU.
- The system expects entity mentions to occur unambiguously in the input sentence.

## Future Improvements

- Improve performance on rare relation classes.
- Explore attention-based entity representations.
- Investigate graph attention networks and alternative GNN architectures.
- Improve confidence calibration.
- Evaluate generalization on additional relation extraction datasets.
- Explore domain-specific fine-tuning.

## Author

**Khyathi Yadav**

B.Tech – Artificial Intelligence and Data Science

IIIT Sri City

**GitHub:** https://github.com/Khyathiyadav

---

*This project was developed as an academic and practical exploration of combining pretrained language models with graph neural networks for semantic relation extraction.*
