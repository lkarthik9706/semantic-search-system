from sklearn.datasets import fetch_20newsgroups
from preprocess import clean_text

def load_data():

    dataset = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes')
    )

    documents = dataset.data
    labels = dataset.target
    categories = dataset.target_names

    # clean documents
    clean_docs = [clean_text(doc) for doc in documents]

    return clean_docs, labels, categories


if __name__ == "__main__":

    docs, labels, categories = load_data()

    print("Total documents:", len(docs))
    print("Total categories:", len(categories))

    print("\nSample cleaned document:\n")
    print(docs[0][:500])