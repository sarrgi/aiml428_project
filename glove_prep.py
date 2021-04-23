import cnn_base_extension_part_3 as base
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
    all_in = train_en + test_en
    # flattened = list(chain.from_iterable(all_in))

    # flatten
    all_in_flattened = base.flatten_input(all_in)

    # lowercase
    all_in_flattened = base.to_lower(all_in_flattened)

    # remove urls
    all_in_flattened = base.remove_urls(all_in_flattened)

    # remove mentions
    all_in_flattened = base.convert_mentions(all_in_flattened)

    # remove stop words
    all_in_flattened = base.remove_stopwords(all_in_flattened)

    # remove stop words
    all_in_flattened = base.remove_punctuation(all_in_flattened)

    # stem check
    all_in_flattened = base.fix_stemming(all_in_flattened)

    exit(1)

    # remove punctuation
    # flattened = base.remove_punctuation(flattened)
    flattened = list(chain.from_iterable(all_in_flattened))

    # create single string
    big_str = "".join(flattened)


    # split into list of unique words
    # all_words = re.split("[\w']+|[.,!?;]", big_str)
    all_words = re.findall(r"\w+|[^\w\s]", big_str, re.UNICODE)
    # unique = list(set(all_words))


    # # print(len(unique), len(all_words))
    print(all_words[:500])

    print("---")
    print(len(all_words[0]), all_words[0])
    # print("---")
    # print(big_str[:500])
    # print("---")
    # print(unique[:500])

    # exit(1)

    # # write to file
    with open('full_pandata_corpus.txt', 'w') as f:
        for item in all_words:
            f.write("%s " % item)
