import argparse
from pyliwc import Liwc
import pandas as pd
import pickle

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
        "-df",
        "--data_folder",
        type=str,
        default="",
    )
    args = parser.parse_args()
    df = pd.read_pickle(
        f"{args.data_folder}/{args.dataset}{'_utterances' if args.dataset != 'chen' else ''}_linguistic.gz"
    )
    if "cad" in args.dataset:
        language = args.dataset.split("_")[1]
    else:
        language = "en"

    liwc = Liwc("/opt/liwc-22/bin/LIWC-22-cli")
    if language == "en":
        liwc_dict = "LIWC22"
    elif language == "it":
        liwc_dict = "LIWC2007 Dictionary - Italian.dicx"
    elif language == "fr":
        liwc_dict = "LIWC2007 Dictionary - French.dicx"
    elif language == "pt":
        liwc_dict = "LIWC2015 Dictionary - Brazilian Portuguese.dicx"

    for column in ["user_prompt", "model_response"]:
        if column not in df:
            continue
        df = pd.concat(
            [
                df,
                liwc.analyze_df(df[column], liwc_dict=liwc_dict).add_prefix(
                    f"{column}_liwc_"
                ),
            ],
            axis=1,
        )
        df.to_pickle(
            f"{args.data_folder}/{args.dataset}{'_utterances' if args.dataset != 'chen' else ''}_linguistic.gz"
        )
