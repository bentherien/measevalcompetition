import spacy

nlp = spacy.load("en_core_web_lg") # same output for 'en_core_sci_scibert' and 'en_core_sci_lg'

example = "The maximum temperature varies from 10,000 K \
to 13,200 K."

# "As ηnet varies from 0.1 to 1, the peak \
# shifts from 1.4Rp (0.5 nbar) to 1.9Rp (0.1 nbar) \
# and the maximum temperature varies from 10,000 K \
# to 13,200 K. It is interesting to note that the \
# temperature profile depends strongly on the heating \
# efficiency but the location of the peak and the \
# maximum temperature depend only weakly on ηnet."


doc = nlp(example)
print([token.text for token in doc])
print("\n\nHere \"K.\" should be \"K\",\".\"\n\n")


suffixes = nlp.Defaults.suffixes + [r'''.''']
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp.tokenizer.suffix_search = suffix_regex.search

doc = nlp(example)
print([token.text for token in doc])