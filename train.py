import argparse
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings")
    parser.add_argument("--model", default="alexnet")
    parser.add_argument("--dataset", default="fasion-mnist")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cuda", default=0, type=int)

    args = parser.parse_args()
