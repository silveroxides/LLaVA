from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="eager")
