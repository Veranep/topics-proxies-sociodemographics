if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to evaluate",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=None,
        help="Number of samples",
    )
    parser.add_argument(
        "-d",
        "--demo",
        type=str,
        default="gender",
        help="Demographic group to evaluate",
    )
    args = parser.parse_args()
    np.random.seed(42)
    stereotypes = pd.read_csv("stereotypes.csv").drop(columns=["source"])
