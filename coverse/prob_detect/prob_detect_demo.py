from coverse.cli.main import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "topic",
                "prob-detect",
                "--model-path",
                "data/models/google-bert/bert-base-chinese",
                "--target",
                "嚼馒头",
                "--text",
                "学习，就像[MASK][MASK][MASK]，因为久了方觉甜",
            ]
        )
    )
