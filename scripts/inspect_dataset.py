import argparse 
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",default="train")
    parser.add_argument("--samples",type=int,default=3)
    args=parser.parse_args()


    ds=load_dataset(
        "ai4bharat/indicdlp",
        split=args.split
    )

    print("Dataset:",ds)
    print("Columns:", ds.column_names)
    print("Features",ds.features)
    print("Rows",len(ds))

    for i in range(min(args.samples,len(ds))):
        row=ds[i]

        for key, value in row.items():
            if key=="image":
                print("Image:",type(value),getattr(value,"size",None))

            elif isinstance(value,list):
                print(key,f"list[{len(value)}]",value[:2])

            else:
                print(key,type(value),value)

    meta = ds.select_columns(['category_ids'])
    category_ids = set()

    for row in meta:
        category_ids.update(int (x) for x in row['category_ids'])

    print("\nUnique category IDs:", sorted(category_ids))
    print("Number of category IDs:", len(category_ids))
    
    
if __name__ == "__main__":
    main()
