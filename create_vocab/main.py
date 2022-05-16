import json

if __name__ == '__main__':
    addIndices = True

    vocabulary: set = set()

    json_string = open("../JerichoWorld-main/data/train.json", "r")
    vocab_file = open("graph_encoder_vocabulary_training.txt", "w")

    # pre-processing
    python_obj: dict = json.load(json_string)

    for liste_index in range(len(python_obj)):
        for state_trans_index in range(len(python_obj[liste_index])):
            trans = python_obj[liste_index][state_trans_index]
            state = trans["state"]["graph"]
            next_state = trans["next_state"]["graph"]
            graphs_lists = state + next_state

            graphs_strings = [".".join(triple) for triple in graphs_lists]
            vocabulary.update(graphs_strings)

    if addIndices:
        index_list = (range(0, len(vocabulary)))
        vocab_dict = dict(zip(vocabulary, index_list))
        vocab_str = str(vocab_dict)
    else:
        vocab_str = str(vocabulary)
    vocab_file.write(vocab_str)
