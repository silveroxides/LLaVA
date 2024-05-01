from llava.train.train import train

# This line checks whether the current script is being executed as the main program (i.e., directly) 
# or being imported as a module into another script.
# directly run in using the deepspeed command line
if __name__ == "__main__":
    train(attn_implementation="eager")
