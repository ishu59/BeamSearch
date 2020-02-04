from collections import defaultdict, OrderedDict
from math import log
from typing import List, Dict
from pprint import pprint
import json
class ExtractGraph:

    # key is head word; value stores next word and corresponding log_probability.
    graph:Dict[str,Dict[str,float]] = {}

    sentences_add = "data/assign1_sentences.txt"
    # sentences_add = "data/small_assign1_sentences.txt"
    new_line_char = '\n'
    empty_char = ''

    def __str__(self):
        return str(self.graph)

    def __repr__(self):
        return str(self.graph)

    def __init__(self, data_path = None):
        # Extract the directed weighted graph, and save to {head_word, {tail_word, log_probability}}
        if data_path is not None:
            self.sentences_add = data_path
        sent_list = self._extract_text_as_list(self.sentences_add)
        self.graph = self._create_graph(sent_list)

    def getProb(self, head_word, tail_word):
        child = self.graph.get(head_word, None)
        if child is None:
            return float('-inf')
        else:
            if child.get(tail_word) is None:
                return float('-inf')
            else:
                return log(child.get(tail_word))
        # return 0.0

    def _create_graph(self, sentence_list: List[str]):
        '''
        Graph has a structure of {k = prefix word, v = { k = word suffix , v = log_probability }
        :param sentence_list:
        :return: {k = prefix word, v = { k = word suffix , v = log_probability }
        '''
        local_graph: Dict[str, Dict[str, float]] = {}
        print('Extracting graph')
        for sentence in sentence_list:
            words = sentence.strip()
            words = words.split()
            for i, single_word in enumerate(words):
                if i >= len(words) - 1:
                    break
                if single_word not in local_graph:
                    local_graph[single_word] = defaultdict(int)
                local_graph[single_word][words[i + 1]] += 1
        for first_word, second_word_dict in local_graph.items():
            total_word_count = sum(second_word_dict.values())
            for sec_word, value in second_word_dict.items():
                local_graph[first_word][sec_word] = value / total_word_count
        # pprint(local_graph)
        #Debugging
        local_list_graph = {}
        for first_word, second_word_dict in local_graph.items():
            ord_dict = OrderedDict(sorted(second_word_dict.items(), key=lambda kv: kv[1], reverse=True))
            for k,v in ord_dict.items():
                ord_dict[k] = log(v)
            local_list_graph[first_word] = ord_dict
        # with open('graph.txt', 'w') as j:
        #     j.write(json.dumps(local_list_graph))

        # for first_word, second_word_dict in local_graph.items():
        #     local_graph[first_word] = {k,v}
        return local_graph

    def _extract_text_as_list(self, file_path) -> List[str]:
        with open(file_path, 'r') as file:
            data =  file.read()
        data = data.split(self.new_line_char)
        # [print(s) for s in data]
        return data



def test_extract_graph():
    g = ExtractGraph()
    print(g.getProb('yen','at'))
    current_word = 'zones'
    words = ['are','fell','have','in','included','such', 'hello', 'may','.']
    [print(current_word,w,g.getProb(current_word, w)) for w in words]
    current_word = '.'
    words = ['are','fell','have','in','included','such', 'hello', 'may','</s>']
    [print(current_word,w,g.getProb(current_word, w)) for w in words]

    print(g.getProb('helloh','worldz'))
    print(g.getProb(None, None))
    # with open('temp.txt', 'w') as wf:
    #     wf.write(str(g.graph))
    # print(g.graph.values())
    #
    # pprint(g)
    # pprint(g.graph)

if __name__ == '__main__':
    print('Hello world!!')
    test_extract_graph()


































# def unusued():
#     with open('data/assign1_sentences.txt', 'r') as file:
#         data = file.read()
#     data = data.replace('\n', '')
#     data = data.split('</s>')
#     data = [s.replace('<s>','').strip() for s in data]
#     ctr = 0
#     for line in data:
#         print(line)
#         ctr += 1
#         if ctr > 10:
#             break