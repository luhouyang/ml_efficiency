from memp_metrics import BaseExtractor, SlottedExtractor

if __name__ == "__main__":

    # url for word list (huge)
    url = 'https://raw.githubusercontent.com/dwyl/english-words/master/words.txt'

    # word list in array
    array = ['one', 'two', 'three', 'four', 'five']

    test_base = False

    if test_base:
        print("Base Class\n")

        # initializing BaseExtractor object
        extractor = BaseExtractor()

        # calling parse_url function
        extractor.parse_url(url)

        # calling pasrse_list function
        # extractor.parse_list(array)

    else:
        print("\nSlotted Class\n")

        # initializing BaseExtractor object
        slotted_extractor = SlottedExtractor()

        # calling parse_url function
        slotted_extractor.parse_url(url)

        # calling pasrse_list function
        # slotted_extractor.parse_list(array)
