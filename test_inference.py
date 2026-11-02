
from inference import predict_relation

test_cases = [
    {
        "sentence": "The storm caused damage to the building.",
        "entity_1": "storm",
        "entity_2": "damage"
    },
    {
        "sentence": "The scientist discovered a new chemical compound.",
        "entity_1": "scientist",
        "entity_2": "compound"
    },
    {
        "sentence": "The laptop contains a powerful processor.",
        "entity_1": "laptop",
        "entity_2": "processor"
    },
    {
        "sentence": "The book is written by the author.",
        "entity_1": "book",
        "entity_2": "author"
    },
    {
        "sentence": "The student reads a newspaper.",
        "entity_1": "student",
        "entity_2": "newspaper"
    }
]

for i, case in enumerate(test_cases, 1):

    print(f"\n{'=' * 55}")
    print(f"TEST CASE {i}")
    print(f"{'=' * 55}")

    try:
        result = predict_relation(
            case["sentence"],
            case["entity_1"],
            case["entity_2"]
        )

        print("Sentence:", case["sentence"])
        print("Entity 1:", case["entity_1"])
        print("Entity 2:", case["entity_2"])
        print("Prediction:", result["relation"])
        print(
            "Confidence:",
            round(result["confidence"] * 100, 2),
            "%"
        )

    except Exception as e:
        print("ERROR:", e)
