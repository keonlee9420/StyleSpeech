import os
import numpy as np
import yaml
import argparse

def main(preprocess_config, model_config):
    preprocessed_path = preprocess_config["path"]["preprocessed_path"]
    max_seq_len = model_config["max_seq_len"]

    with open(
        os.path.join(preprocessed_path, "train.txt"), "r", encoding="utf-8"
    ) as f:
        filtered_list = []
        for i, line in enumerate(f.readlines()):
            basename, speaker, *_ = line.strip("\n").split("|")
            mel_path = os.path.join(
                preprocessed_path,
                "mel",
                "{}-mel-{}.npy".format(speaker, basename),
            )
            mel = np.load(mel_path)
            if mel.shape[0] <= max_seq_len:
                filtered_list.append(line)

    # Write Filtered Filelist
    with open(os.path.join(preprocessed_path, "train_filtered.txt"), "w", encoding="utf-8") as f:
        for line in filtered_list:
            f.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)

    main(preprocess_config, model_config)