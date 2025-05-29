import argparse
from typing import List


def add_context(orig_txt: List[str], context: List[str], doc_ids: List[str], sep_token: str = "</s>",
                ws: int = 2) -> List[str]:
    """Function that adds the previous sentences as context to the current sentence, respecting document boundaries
    :param orig_txt: the original text
    :param context: the text from which the context will be taken (same as orig_txt for source/reference)
    :param doc_ids: the document where each segment belongs to
    :param sep_token: the separator token of the tokenizer for the specific model
    :param ws: the window size, maximum of the previous sentences to be considered as context
    :return: the original text augmented with context
    """
    if not (len(orig_txt) == len(context) == len(doc_ids)):
        raise Exception(f'Lengths should match: len(orig_txt)={len(orig_txt)}, len(context)={len(context)}, len(doc_ids)={len(doc_ids)}')
    i, k = 0, 0
    augm_txt = []
    doc_id = doc_ids[0]
    while i < len(orig_txt):
        if doc_ids[i] == doc_id:
            context_window = context[i - min(k, ws):i]
            augm_txt.append(" {} ".format(sep_token).join(context_window + [orig_txt[i]]))
            i += 1
        else:
            doc_id = doc_ids[i]
            k = -1
        k += 1
    return augm_txt

def merge_document(text: List[str], doc_ids: List[str]):
    """Function that merges the text into documents
    :param text: the text to be merged
    :param doc_ids: the document where each segment belongs to
    :param sep_token: the separator token of the tokenizer for the specific model
    :return: the text merged into documents
    """
    if not (len(text) == len(doc_ids)):
        raise Exception(f'Lengths should match: len(text)={len(text)}, len(doc_ids)={len(doc_ids)}')
    doc_names = set(doc_ids)

    documents = []
    
    for i, doc_name in enumerate(doc_names):
        doc_text = [text[j] for j in range(len(text)) if doc_ids[j] == doc_name]
        doc_text = " </s> ".join(doc_text)
        documents.append(doc_text)    
    
    
    assert len(documents) == len(doc_names)
    return documents
