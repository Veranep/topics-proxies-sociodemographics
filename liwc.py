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
    args = parser.parse_args()
    df = pd.read_pickle(f"data/{args.dataset}_utterances_linguistic.gz")
    liwc = Liwc("/opt/liwc-22/bin/LIWC-22-cli")
    liwc_dict = "LIWC22"

    for column in ["user_prompt", "model_response"]:
        df = pd.concat(
            [
                df,
                liwc.analyze_df(df[column], liwc_dict=liwc_dict).add_prefix(
                    f"{column}_liwc_"
                ),
            ],
            axis=1,
        )
        df.to_pickle(f"data/{args.dataset}_utterances_linguistic.gz")
