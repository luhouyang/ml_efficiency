from memory_profiler import profile
import requests


class BaseExtractor():
    # decorator which specifies which
    # function to monitor
    @profile
    def parse_list(self, array):

        # create a file object
        f = open('tests/memory/words.txt', 'w')
        for word in array:

            # writing words to file
            f.writelines(word)

    # decorator which specifies which
    # function to monitor
    @profile
    def parse_url(self, url):

        # fetches the response
        response = requests.get(url).text
        with open('tests/memory/url.txt', 'w') as f:

            # writing response to file
            f.writelines(response)


class SlottedExtractor():
    __slots__ = []

    # decorator which specifies which
    # function to monitor
    @profile
    def parse_list(self, array):

        # create a file object
        f = open('tests/memory/words.txt', 'w')
        for word in array:

            # writing words to file
            f.writelines(word)

    # decorator which specifies which
    # function to monitor
    @profile
    def parse_url(self, url):

        # fetches the response
        response = requests.get(url).text
        with open('tests/memory/url.txt', 'w') as f:

            # writing response to file
            f.writelines(response)
