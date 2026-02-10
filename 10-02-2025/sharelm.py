from collections import Counter
from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("shachardon/ShareLM")["train"]
    dataset = dataset.flatten()
    print(len(dataset))
    dataset = dataset.filter(
        lambda x: x["conversation_metadata.language"] == "English"
    )
    print(len(dataset))
    for metadata in ["location", "age", "gender"]:
        filtered = dataset.filter(
            lambda x: x[f"user_metadata.{metadata}"] != ""
        )
        print(len(filtered))
        unique_items = Counter(filtered[f"user_metadata.{metadata}"])
        print(unique_items, len(unique_items))
