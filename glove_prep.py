import cnn_base_extension as base
from itertools import chain
import re

if __name__ == "__main__":
    # read in data
    test_en = base.read_data("data/pandata/test/en/*.xml")
    train_en = base.read_data("data/pandata/train/en/*.xml")

    # remove file names
    for i in range(len(test_en)):
        test_en[i] = test_en[i][1:]
    for i in range(len(train_en)):
        train_en[i] = train_en[i][1:]

    # create one list
    all_in = test_en + train_en
    flattened = list(chain.from_iterable(all_in))

    # create single string
    big_str = "".join(flattened)

    # convert to lower case
    big_str = big_str.lower()

    # split into list of unique words
    # all_words = re.split("[\w']+|[.,!?;]", big_str)
    all_words = re.findall(r"\w+|[^\w\s]", big_str, re.UNICODE)
    # unique = list(set(all_words))


    # print(len(unique), len(all_words))
    # print(all_words[:50])
    # print("---")
    # print(big_str[:500])
    # print("---")
    # print(unique[:500])

    # # write to file
    with open('full_pandata_corpus.txt', 'w') as f:
        for item in all_words:
            f.write("%s " % item)
