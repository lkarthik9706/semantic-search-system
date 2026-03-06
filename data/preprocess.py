import re

def clean_text(text):

    # convert to lowercase
    text = text.lower()

    # remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()