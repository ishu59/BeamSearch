from math import log, exp
from pprint import pprint
from typing import Dict, List
import StringDouble
from ExtractGraph import ExtractGraph
from queue import PriorityQueue


class _Token:
    """
    Class to keep track of words to be added in the sentence
    also implemented les than for comparision between them which will enable in sorting and selecting max

    """

    def __init__(self, curr_word: str, sentence: List[str], log_probability: float = 0):
        self.curr_word = curr_word
        self.sentence = sentence
        self.log_probability = log_probability

    def __lt__(self, other):
        return self.log_probability < other.log_probability


class BeamSearch:
    graph: Dict[str, Dict[str, float]] = {}

    def __init__(self, input_graph: ExtractGraph):
        # Dict[str, Dict[str, float]]
        self._extract_graph_obj = input_graph
        self.graph = input_graph.graph
        self.EOS = '</s>'
        self.BOS = '<s>'
        return

    def beamSearchV1(self, pre_words: str, beamK: int, maxToken: int) -> StringDouble.StringDouble:
        """
        Method computes best sentence starting from pre_words having the highest score
        Here the deafault lambda is set to 0 so there is no normalization and delegates the task to beamSearchV2
        :param pre_words: starting words of the sentence
        :param beamK: width of the beam for beam search
        :param maxToken: depth of the search in beam search
        :return: sentence with best score and it log probability
        """
        return self.beamSearchV2(pre_words, beamK, param_lambda=0, maxToken=maxToken)

    def beamSearchV2(self, pre_words, beamK, param_lambda, maxToken) -> StringDouble.StringDouble:
        '''
        Memory efficient and computationally efficient implementation of Beam search
        It maintains the prority queue of size beam with to keep track of search tree
        No duplicate tree is required as we update the sentence in the prority queue
        Method computes best sentence starting from pre_words having the highest score based on lambda
        :param pre_words: starting words of the sentence
        :param beamK: width of the beam for beam search
        :param param_lambda: hyper parameter which can be tuned always between 0 and 1
        :param maxToken: depth of the search in beam search
        :return: sentence with best score and it log probability
        '''
        # Beam search with sentence length normalization.
        # Basic beam search.

        pre_words_list = pre_words.split()
        search_depth = maxToken - len(pre_words_list)

        # Raise exception if the starting string is not BOS tag or invalid
        if len(pre_words_list) < 1 or pre_words_list[0] != self.BOS:
            raise Exception('Not a valid starting point')
        if len(pre_words_list) == 1:
            curr_log_prob = 0
        else:
            curr_log_prob = 0
            for index in range(len(pre_words_list) - 1):
                curr_log_prob += (self._extract_graph_obj.getProb(pre_words_list[index], pre_words_list[index + 1]))
        search_start = pre_words_list[-1]
        start_node = _Token(curr_word=search_start, sentence=pre_words_list[:-1], log_probability=curr_log_prob)
        seed_list: List[_Token] = [start_node]

        def insert_pq(pq: PriorityQueue, item: _Token):
            '''
            Helper function to insert in the priority queue. It will always keep the max values of size beam width
            Since we always pop the first value the order is O(1) and inserting is log(beamWidth)
            :param pq: priority queue object
            :param item: item to be inserted in priority queue
            :return: updated priority queue based on maximum value.
            '''
            if pq.full():
                pq.put_nowait(max(pq.get_nowait(), item))
            else:
                pq.put_nowait(item)
            return pq

        # Actual computation start here
        # Iterate over the depth of the beam search
        while search_depth >= 0:
            search_depth -= 1

            # Initialize the priority queue with max size of beam width
            p_que_search = PriorityQueue(maxsize=beamK)

            # iterate over all the candidates in the beam
            for seed in seed_list:
                items_to_search = self.graph.get(seed.curr_word)

                # If item is end of string just add it to the prority queue
                if items_to_search is None or seed.curr_word == self.EOS:
                    sent = seed.sentence.copy()
                    if seed.curr_word is not None:
                        sent.append(seed.curr_word)
                    normalzied_prob = seed.log_probability

                    tok = _Token(curr_word=None,
                                 sentence=sent,
                                 log_probability=normalzied_prob)
                    p_que_search = insert_pq(p_que_search, tok)
                else:

                    # Iterate over the new candidates to fill and compute the new prority queue based on probability
                    for item, prob in items_to_search.items():
                        sent = seed.sentence.copy()
                        b = seed.curr_word
                        sent.append(b)

                        # Every time we fill the token in the priority queue we and the current word to the sentence
                        # This enables us to keep track of sentence and save memeory by not manintaing the graph as we
                        # save all the required information for sentence generation
                        tok = _Token(curr_word=item,
                                     sentence=sent,
                                     log_probability=(seed.log_probability + log(prob))
                                     # log_probability=(seed.log_probability + log(prob))/(len(sent) ** param_lambda)
                                     )
                        p_que_search = insert_pq(p_que_search, tok)

            # Empty the create list to store new candidates
            seed_list = []

            # Pop all items from queue to fill the seeds
            while not p_que_search.empty():
                seed_list.append(p_que_search.get_nowait())
            # [print(s.sentence, s.log_probability, s.curr_word) for s in seed_list]
        # [print(len(item.sentence), item.log_probability, item.sentence) for item in seed_list]

        # Normalize the priority queue based on lambda
        for item in seed_list:
            item.log_probability = item.log_probability / (len(item.sentence) ** param_lambda)
        # [print(len(item.sentence), item.log_probability, item.sentence) for item in seed_list]

        # Get the maximum value from the prority queue
        main_item = max(seed_list)

        try:
            final_val = main_item
        except:
            final_val = start_node = _Token(curr_word=search_start, sentence=[self.BOS, self.EOS], log_probability=float('-inf'))

        # fill the sentence object with maximum sentence
        sentence = " ".join(final_val.sentence)
        probability = (final_val.log_probability)
        return StringDouble.StringDouble(sentence, probability)


# Below are my custom function to test the program
# Not required in the assignment but still helps to verify the implementataion

def test_beamSearchV2():
    g = ExtractGraph()
    graph = g
    param_lambda = 0.7
    beam_search = BeamSearch(graph)
    sentence_prob = beam_search.beamSearchV2("<s>", 10, param_lambda, 20)
    print(str(sentence_prob.score) + "\t" + sentence_prob.string)
    sentence_prob = beam_search.beamSearchV2("<s> Israel and Jordan signed the peace", 10, param_lambda, 40)
    print(str(sentence_prob.score) + "\t" + sentence_prob.string)
    sentence_prob = beam_search.beamSearchV2("<s> It is", 10, param_lambda, 15)
    print(str(sentence_prob.score) + "\t" + sentence_prob.string)

def test_other():
    graph = ExtractGraph()
    beam_search = BeamSearch(graph)
    sentence_prob = beam_search.beamSearchV2('', 10, 0, 20)

def test_main():
    graph = ExtractGraph()
    # Test extraction correctness.
    head_word = "<s>"
    tail_word = "Water"
    print("The log_probability of \"" + tail_word + "\" appearing after \"" + head_word + "\" is " + str(
        graph.getProb(head_word, tail_word)))
    head_word = "Water"
    tail_word = "<s>"
    print("The log_probability of \"" + tail_word + "\" appearing after \"" + head_word + "\" is " + str(
        graph.getProb(head_word, tail_word)))
    head_word = "planned"
    tail_word = "economy"
    print("The log_probability of \"" + tail_word + "\" appearing after \"" + head_word + "\" is " + str(
        graph.getProb(head_word, tail_word)))
    head_word = "."
    tail_word = "</s>"
    print("The log_probability of \"" + tail_word + "\" appearing after \"" + head_word + "\" is "
          + str(graph.getProb(head_word, tail_word)))

    # Find the sentence with highest log_probability using basic beam search.
    beam_search = BeamSearch(graph)
    sentence_prob = beam_search.beamSearchV1("<s>", 10, 20)
    print(str(sentence_prob.score) + "\t" + sentence_prob.string)
    sentence_prob = beam_search.beamSearchV1("<s> Israel and Jordan signed the peace", 10, 40)
    print(str(sentence_prob.score) + "\t" + sentence_prob.string)
    sentence_prob = beam_search.beamSearchV1("<s> It is", 10, 15)
    print(str(sentence_prob.score) + "\t" + sentence_prob.string)

    # Find the sentence with highest log_probability using beam search with sentence length-normalzation.
    param_lambda = 0.7
    beam_search = BeamSearch(graph)
    sentence_prob = beam_search.beamSearchV2("<s>", 50, param_lambda, 20)
    print(str(sentence_prob.score) + "\t" + sentence_prob.string)
    sentence_prob = beam_search.beamSearchV2("<s> Israel and Jordan signed the peace", 50, param_lambda, 40)
    print(str(sentence_prob.score) + "\t" + sentence_prob.string)
    sentence_prob = beam_search.beamSearchV2("<s> It is", 50, param_lambda, 15)
    print(str(sentence_prob.score) + "\t" + sentence_prob.string)

if __name__ == '__main__':
    print('=' * 50)
    # test_beamSearchV1()
    test_main()
    # test_other()




######################################################################################################################
######################################################################################################################
####### End of program ##########
######################################################################################################################
######################################################################################################################



    # def beamSearchV2(self, pre_words, beamK, param_lambda, maxToken):
    #     # Beam search with sentence length normalization.
    #     # Basic beam search.
    #
    #     pre_words_list = pre_words.split()
    #     search_depth = maxToken - len(pre_words_list)
    #     search_start = pre_words_list[-1]
    #     if pre_words_list[0] != '<s>':
    #         raise Exception('Not a valid starting point')
    #     if len(pre_words_list) == 1:
    #         curr_log_prob = 0
    #     else:
    #         curr_log_prob = 0
    #         for index in range(len(pre_words_list) - 1):
    #             curr_log_prob += (self._extract_graph_obj.getProb(pre_words_list[index], pre_words_list[index + 1]))
    #
    #     start_node = _Token(curr_word=search_start, sentence=pre_words_list[:-1], log_probability=curr_log_prob)
    #     seed_list: List[_Token] = [start_node]
    #
    #     def insert_pq(pq: PriorityQueue, item: _Token):
    #         if pq.full():
    #             pq.put_nowait(max(pq.get_nowait(), item))
    #         else:
    #             pq.put_nowait(item)
    #         return pq
    #
    #     while search_depth >= 0:
    #         search_depth -= 1
    #         p_que_search = PriorityQueue(maxsize=beamK)
    #         for seed in seed_list:
    #             # print(seed.curr_word)
    #             # if seed.curr_word is None:
    #             #     print('hello None')
    #             items_to_search = self.graph.get(seed.curr_word)
    #             if items_to_search is None:
    #                 sent = seed.sentence.copy()
    #                 if seed.curr_word is not None:
    #                     sent.append(seed.curr_word)
    #                 if seed.curr_word == '</s>':
    #                     normalzied_prob = seed.log_probability / (len(sent) ** param_lambda)
    #                 else:
    #                     normalzied_prob = seed.log_probability
    #                 tok = _Token(curr_word=None,
    #                              sentence=sent,
    #                              log_probability=normalzied_prob)
    #                 p_que_search = insert_pq(p_que_search, tok)
    #             else:
    #                 for item, prob in items_to_search.items():
    #                     sent = seed.sentence.copy()
    #                     b = seed.curr_word
    #                     sent.append(b)
    #                     tok = _Token(curr_word=item,
    #                                  sentence=sent,
    #                                  log_probability=(seed.log_probability + log(prob)) / (len(sent) ** param_lambda)
    #                                  )
    #                     p_que_search = insert_pq(p_que_search, tok)
    #         seed_list = []
    #         while not p_que_search.empty():
    #             seed_list.append(p_que_search.get_nowait())
    #         # [print(s.sentence, s.log_probability, s.curr_word) for s in seed_list]
    #     # [print(len(item.sentence), item.log_probability, item.sentence) for item in seed_list]
    #     try:
    #         final_val = seed_list[-1]
    #     except IndexError:
    #         final_val = start_node = _Token(curr_word=search_start, sentence=['</s>'], log_probability=0)
    #     sentence = " ".join(final_val.sentence)
    #     probability = (final_val.log_probability)
    #     return StringDouble.StringDouble(sentence, probability)

    # def beamSearchV1(self, pre_words:str, beamK:int, maxToken:int, param_lambda = 0):
    # 	# Basic beam search.
    #
    #     pre_words_list = pre_words.split()
    #     search_depth = maxToken - len(pre_words_list)
    #     search_start = pre_words_list[-1]
    #     if pre_words_list[0] != '<s>':
    #         raise Exception('Not a valid starting point')
    #     if len(pre_words_list) == 1:
    #         curr_log_prob = 0
    #     else:
    #         curr_log_prob = 0
    #         for index in range(len(pre_words_list)-1):
    #             curr_log_prob += log(self._extract_graph_obj.getProb(pre_words_list[index],pre_words_list[index+1]))
    #     start_node = _Token(curr_word=search_start,sentence=pre_words_list[:-1], log_probability=curr_log_prob)
    #     seed_list:List[_Token] = [start_node]
    #     def insert_pq(pq:PriorityQueue, item:_Token):
    #         if pq.full():
    #             pq.put_nowait(max(pq.get_nowait(),item))
    #         else:
    #             pq.put_nowait(item)
    #         return pq
    #     while search_depth > 0:
    #         search_depth -= 1
    #         p_que_search = PriorityQueue(maxsize=beamK)
    #         for seed in seed_list:
    #             # print(seed.curr_word)
    #             items_to_search = self.graph.get(seed.curr_word)
    #             if items_to_search is None :
    #                 sent = seed.sentence.copy()
    #                 if seed.curr_word is not None:
    #                     sent.append(seed.curr_word)
    #                 if seed.curr_word == '</s>':
    #                     normalzied_prob = seed.log_probability / (len(sent) ** param_lambda)
    #                 tok = _Token(curr_word=None,
    #                              sentence=sent,
    #                              log_probability=seed.log_probability)
    #                 p_que_search = insert_pq(p_que_search, tok)
    #                 continue
    #             for item, prob in items_to_search.items():
    #                 a = seed.sentence.copy()
    #                 b = seed.curr_word
    #                 a.append(b)
    #                 tok = _Token(curr_word=item,
    #                              sentence=a,
    #                              log_probability=seed.log_probability + log(prob))
    #                 p_que_search = insert_pq(p_que_search,tok)
    #         seed_list = []
    #         while not p_que_search.empty():
    #             seed_list.append(p_que_search.get_nowait())
    #         # [print(s.sentence, s.log_probability, s.curr_word) for s in seed_list]
    #     [print(len(item.sentence),item.log_probability ,item.sentence ) for item in seed_list]
    #     try:
    #         final_val = seed_list[-1]
    #     except IndexError:
    #         final_val = start_node = _Token(curr_word=search_start,sentence=['</s>'], log_probability=0)
    #     sentence = " ".join(final_val.sentence)
    #     probability = final_val.log_probability
    #     return StringDouble.StringDouble(sentence, probability)

    # def beamSearchV2_Temp1(self, pre_words, beamK, param_lambda, maxToken):
    # 	# Beam search with sentence length normalization.
    #     # Basic beam search.
    #
    #     pre_words_list = pre_words.split()
    #     search_depth = maxToken - len(pre_words_list)
    #     search_start = pre_words_list[-1]
    #     if pre_words_list[0] != '<s>':
    #         raise Exception('Not a valid starting point')
    #     if len(pre_words_list) == 1:
    #         curr_log_prob = 0
    #     else:
    #         curr_log_prob = 0
    #         for index in range(len(pre_words_list) - 1):
    #             curr_log_prob += (self._extract_graph_obj.getProb(pre_words_list[index], pre_words_list[index + 1]))
    #
    #     start_node = _Token(curr_word=search_start, sentence=pre_words_list[:-1], log_probability=curr_log_prob)
    #     seed_list: List[_Token] = [start_node]
    #
    #     def insert_pq(pq: PriorityQueue, item: _Token):
    #         if pq.full():
    #             pq.put_nowait(max(pq.get_nowait(), item))
    #         else:
    #             pq.put_nowait(item)
    #         return pq
    #
    #     while search_depth >= 0:
    #         search_depth -= 1
    #         p_que_search = PriorityQueue(maxsize=beamK)
    #         for seed in seed_list:
    #             # print(seed.curr_word)
    #             # if seed.curr_word is None:
    #             #     print('hello None')
    #             items_to_search = self.graph.get(seed.curr_word)
    #             if items_to_search is None:
    #                 sent = seed.sentence.copy()
    #                 if seed.curr_word is not None:
    #                     sent.append(seed.curr_word)
    #                 if seed.curr_word == '</s>':
    #                     normalzied_prob = seed.log_probability / (len(sent) ** param_lambda)
    #                 else:
    #                     normalzied_prob = seed.log_probability
    #                 tok = _Token(curr_word=None,
    #                              sentence=sent,
    #                              log_probability=normalzied_prob)
    #                 p_que_search = insert_pq(p_que_search, tok)
    #             else:
    #                 for item, prob in items_to_search.items():
    #                     sent = seed.sentence.copy()
    #                     b = seed.curr_word
    #                     sent.append(b)
    #                     tok = _Token(curr_word=item,
    #                                  sentence=sent,
    #                                  log_probability=(seed.log_probability + log(prob))/(len(sent) ** param_lambda)
    #                                  )
    #                     p_que_search = insert_pq(p_que_search, tok)
    #         seed_list = []
    #         while not p_que_search.empty():
    #             seed_list.append(p_que_search.get_nowait())
    #         # [print(s.sentence, s.log_probability, s.curr_word) for s in seed_list]
    #     # [print(len(item.sentence), item.log_probability, item.sentence) for item in seed_list]
    #     try:
    #         final_val = seed_list[-1]
    #     except IndexError:
    #         final_val = start_node = _Token(curr_word=search_start, sentence=['</s>'], log_probability=0)
    #     sentence = " ".join(final_val.sentence)
    #     probability = (final_val.log_probability)
    #     return StringDouble.StringDouble(sentence, probability)
#
# def test_lambda():
#     graph = ExtractGraph()
#     beam_search = BeamSearch(graph)
#     param_lambda_list = [x*0.01 for x in range(55,100,5)]
#     param_lambda_list.insert(0,0.0)
#     sentences = ['<s> In the past year',
#                  '<s> It was at the',
#                  '<s> He stressed that Indonesia',
#                  '<s> It is estimated that the power reserve',
#                  '<s> It particularly highlighted the fact that the external debt',
#                  '<s> Secretary of State Warren Christopher']
#     data_dict = dict(SentenceNum = [], LambdaValue = [], SentenceLength = [])
#     for index, line in enumerate(sentences,1):
#         for lam in param_lambda_list:
#             # print(lam, line)
#             sentence_prob = beam_search.beamSearchV2(line, 10, lam, 20)
#             print('=' * 100)
#             print(f"Parameter Lambda is {lam} and Context sentence is: [{line}]")
#             print(f'Log Probability of sentence is {sentence_prob.score},\nCompleted sentence is [{sentence_prob.string}]')
#             print(f'Length of sentence is {len(sentence_prob.string)}')
#             print('='*100)
#             data_dict['SentenceNum'].append(index)
#             data_dict['LambdaValue'].append(lam)
#             data_dict['SentenceLength'].append(len(sentence_prob.string))
#         # if index > 1:
#         #     break
#
#     # pprint(data_dict)
#     df = pd.DataFrame(data_dict)
#     # print(df)
#     df = df.pivot(index = 'LambdaValue', values='SentenceLength', columns='SentenceNum')
#     # print(df)
#     df.plot()
#     plt.xlabel('Lambda Values')
#     plt.ylabel('Sentence length')
#     plt.title('Lambda vs sentence length')
#     plt.savefig('Comparision.png')
#     # plt.show()
