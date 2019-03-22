import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
from heapq import nlargest
from collections import defaultdict

'''
Inspired by: https://towardsdatascience.com/write-a-simple-summarizer-in-python-e9ca6138a08e
Siddhant Sharma, 2019
'''
class Summarizer:
    def __init__(self):
        """
        Initializes Summarizer object and creates summary of input file
        """
        args = self.parse_arguments()
        content = self.read_file(args.filepath)
        content = self.clean_whitespace(content)
        sentence_tokens, word_tokens = self.tokenize_text(content)
        sentence_ranks = self.score_tokens(word_tokens, sentence_tokens)
        self.summary = self.compile_summary(sentence_ranks, sentence_tokens, args.length)
    
    def parse_arguments(self):
        """
        Parses the command line arguments
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('filepath', help = 'File name of text to summarize')
        parser.add_argument('-l', '--length', default = 7, help = 'Number of sentences to return')
        return parser.parse_args()
       
    def read_file(self, path):
        """
        Reads the file at the given path
        Throws error if unable to find or read file
        """
        try:
            with open(path, 'r') as file:
                return file.read()

        except IOError as e:
            print("Fatal Error: File ({}) could not be located or is not readable.".format(path))

    def clean_whitespace(self, text):
        """
        Removes unnecessary whitespace from input text
        """
        replace = {
            ord('\f') : ' ',
            ord('\t') : ' ',
            ord('\n') : ' ',
            ord('\r') : None
        }
        return text.translate(replace)

    def tokenize_text(self, text):
        """
        Filters out stop words from NLTK corpus and Python String class punctuation to
        create a list of tokenized sentences and tokenized words
        """
        stop_words = set(stopwords.words('english') + list(punctuation))
        words = word_tokenize(text.lower())
        return [sent_tokenize(text), [word for word in words if word not in stop_words]]

    def score_tokens(self, filtered_words, sentence_tokens):
        """
        Creates a frequency map based on filtered words
        Uses this frequency map to then make map of each sentence and its total score
        """
        word_freq = FreqDist(filtered_words)
        ranking = defaultdict(int)
        for i, sentence in enumerate(sentence_tokens):
            for word in word_tokenize(sentence.lower()):
                if word in word_freq:
                    ranking[i] += word_freq[word]
        return ranking

    def compile_summary(self, ranks, sentences, length):
        """
        Uses a ranking map made by score_tokens function to extract highest ranking
        sentences in order and converts from array to string
        """
        if int(length) > len(sentences):
            print("Error, more sentences requested than available. Use --l (--length) flag to adjust.")
            exit()
        indices = nlargest(length, ranks, key = ranks.get)
        final_sentences = [sentences[i] for i in sorted(indices)]
        return ''.join(final_sentences)
    
    def main(self):
        """
        Returns the summary
        """
        return self.summary
    
if __name__ == "__main__":
    summarizer = Summarizer()
    summary = summarizer.main()
    print(summary)
