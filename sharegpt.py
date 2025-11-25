from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("shachardon/ShareLM")
    print(dataset.features)
    dataset = dataset.flatten()
    print(dataset.features)
