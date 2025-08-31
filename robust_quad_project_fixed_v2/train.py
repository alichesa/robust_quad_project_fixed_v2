# Convenience launcher so you can run: python train.py --config configs\base.yaml
from scripts.train import main
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    args = ap.parse_args()
    main(args)
