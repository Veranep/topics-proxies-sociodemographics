from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("shachardon/ShareLM")["train"]
    dataset = dataset.flatten()
    print(len(dataset))
    for metadata in ["location", "age", "gender"]:
        filtered = dataset.filter(
            lambda x: x[f"user_metadata.{metadata}"] != ""
        )
        print(len(filtered))
