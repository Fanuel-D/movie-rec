import argparse
import sys
import pandas as pd
import pickle



def main():
    parser = argparse.ArgumentParser(description="Example using choices in argparse")

    parser.add_argument(
    "mode", 
    choices=["1", "2", "3"],  
    help="Operating mode: train, test, or evaluate"
)
    args = parser.parse_args()

    with open("df.pkl", 'rb') as f:
        df = pickle.load(f) 

    print(df.head())


    if args.mode == "1":
        print("hi")
        
        


main()