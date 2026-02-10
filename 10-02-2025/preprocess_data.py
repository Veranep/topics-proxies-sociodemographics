def get_prism_convos(df):
    convos = [
        get_convo(df.iloc[i], args.mitigation, args.conversations_dataset)
        for i in range(len(df))
    ]
    for convo in convos:
        to_remove = []
        for i in range(len(convo)):
            if i > 0 and convo[i]["role"] == convo[i - 1]["role"]:
                to_remove.append(i)
        to_remove = to_remove[::-1]
        for idx in to_remove:
            del convo[idx]
    return convos


def get_cad_convos(df):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="prism",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        help="Folder dataset is stored in",
        default="",  # "/scratch/vneplen/sociodemographics-interpretability-mitigation"
    )
    if args.dataset == "prism":
        conversations = load_dataset(
            "HannahRoseKirk/prism-alignment", "conversations"
        )["train"].to_pandas()
        survey = load_dataset("HannahRoseKirk/prism-alignment", "survey")[
            "train"
        ].to_pandas()

        df = pd.merge(
            conversations,
            survey,
            on=["user_id"],
        )
        to_simplify = ["religion", "ethnicity"]
        for column in to_simplify:
            df[column] = df[column].apply(
                lambda x: (
                    dict(x)["simplified"]
                    if type(x) == dict
                    else "Prefer not to say"
                )
            )
        regions = ["birth_region", "reside_region"]
        for region in regions:
            df[region] = df["location"].apply(
                lambda x: (
                    dict(x)[region] if type(x) == dict else "Prefer not to say"
                )
            )

        df.to_pickle(f"{args.folder}/prism_preprocessed.gz")
