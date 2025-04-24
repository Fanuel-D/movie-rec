import argparse
import sys
import pandas as pd







def main():
    parser = argparse.ArgumentParser(description="Example using choices in argparse")

    parser.add_argument(
    "mode", 
    choices=["Top5movies", "Favorite actress", "personality"],  
    help="Operating mode: train, test, or evaluate"
)
    args = parser.parse_args()




    # with open('/data/cs91r-s25/gutenberg/mirror/metadata/metadata.csv', newline='') as csvfile:
    #     ids = []
    #     spamreader = csv.DictReader(csvfile)
    #     for row in spamreader:
    #         ids.append(row['id'])
                    
    #     unq_words  = set()
    #     for f_id in ids:
    #         if not os.path.exists(f"/data/cs91r-s25/gutenberg/mirror/data/counts/{f_id}_counts.txt"):
    #             continue
    #         f = open(f"/data/cs91r-s25/gutenberg/mirror/data/counts/{f_id}_counts.txt")
    #         for line in f:
    #             if not line.strip():
    #                 print("No counts for this file")
    #                 break
                
    #             word, count = line.strip().split()

            
    #             if word in dnry:
    #                 unq_words.add(word)
    basics = pd.read_csv("/scratch/fdana1/movies_genres.csv", sep = '\t')
    print(basics)


    # print(data)

    print(args.mode)




main()