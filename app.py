
import streamlit as st
import pandas as pd

from inference import predict_relation, relation_labels


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Relation Extraction using R-GCN",
    page_icon="🔗",
    layout="wide"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .main-title {
        color: #163a70;
        font-size: 34px;
        font-weight: 700;
        text-align: center;
    }

    .subtitle {
        color: #64748b;
        text-align: center;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .result-card {
        background-color: #eaf2ff;
        padding: 22px;
        border-radius: 12px;
        border-left: 5px solid #2563eb;
        margin-top: 15px;
    }

    .result-label {
        color: #475569;
        font-size: 15px;
    }

    .result-value {
        color: #163a70;
        font-size: 24px;
        font-weight: 700;
    }

    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.markdown(
    '<div class="main-title">Relation Extraction using Enhanced R-GCN</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Identify semantic relationships between two entities in a sentence '
    'using a BERT-enhanced Relational Graph Convolutional Network.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 4. SIDEBAR INFORMATION
# ==========================================

with st.sidebar:
    st.title("About the Project")

    st.write(
        """
        This application predicts the semantic relationship
        between two entities in a given sentence.

        **Model architecture**
        - BERT: Contextual word representations
        - Dependency Graph: Syntactic relationships
        - R-GCN: Graph-based representation learning
        - Classifier: 19 relation classes

        **Dataset:** SemEval-2010 Task 8
        """
    )

    st.divider()

    st.caption(
        "This is a research prototype. Predictions and confidence "
        "scores should not be treated as guaranteed correct."
    )


# ==========================================
# 5. INPUT SECTION
# ==========================================

st.subheader("Enter Your Sentence")

sentence = st.text_area(
    "Sentence",
    placeholder="Example: The storm caused damage to the building.",
    height=110
)

col1, col2 = st.columns(2)

with col1:
    entity_1 = st.text_input(
        "Entity 1",
        placeholder="Example: storm"
    )

with col2:
    entity_2 = st.text_input(
        "Entity 2",
        placeholder="Example: damage"
    )

st.caption(
    "Enter the two entity mentions exactly as they appear in the sentence."
)

predict_button = st.button("Predict Relation")


# ==========================================
# 6. PREDICTION
# ==========================================

if predict_button:

    if not sentence.strip():
        st.warning("Please enter a sentence.")

    elif not entity_1.strip() or not entity_2.strip():
        st.warning("Please enter both entities.")

    else:

        try:
            with st.spinner(
                "Analyzing sentence and predicting the relation..."
            ):

                result = predict_relation(
                    sentence,
                    entity_1,
                    entity_2
                )

            st.divider()
            st.subheader("Prediction Result")

            relation = result["relation"]
            confidence = result["confidence"]
            class_id = result["class_id"]

            # Main prediction
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Relation</div>
                    <div class="result-value">{relation}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric(
                    "Class ID",
                    class_id
                )

            with metric2:
                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

            st.info(
                "The confidence score represents the model's softmax "
                "output. It is not a guarantee that the prediction is correct."
            )

            # ==================================
            # 7. PROBABILITY DISTRIBUTION
            # ==================================

            st.divider()
            st.subheader("Relation Probability Distribution")

            probabilities = result["probabilities"]

            probability_data = []

            for index, probability in enumerate(probabilities):

                label = relation_labels[str(index)]

                probability_data.append({
                    "Relation": label,
                    "Probability (%)": probability * 100
                })

            probability_df = pd.DataFrame(probability_data)

            probability_df = probability_df.sort_values(
                by="Probability (%)",
                ascending=False
            )

            st.write("Top 5 predicted relations")

            top5 = probability_df.head(5).copy()

            top5["Probability (%)"] = top5[
                "Probability (%)"
            ].round(2)

            st.dataframe(
                top5,
                use_container_width=True,
                hide_index=True
            )

            st.write("Probability distribution across all 19 classes")

            chart_df = probability_df.set_index("Relation")

            st.bar_chart(
                chart_df,
                horizontal=True
            )

            # ==================================
            # 8. INPUT SUMMARY
            # ==================================

            st.divider()
            st.subheader("Input Summary")

            st.write("**Sentence:**", sentence)
            st.write("**Entity 1:**", entity_1)
            st.write("**Entity 2:**", entity_2)

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "An unexpected error occurred while predicting the relation."
            )
            st.exception(error)


# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.caption(
    "Relation Extraction using Enhanced R-GCN | "
    "BERT + Dependency Graph + R-GCN"
)
