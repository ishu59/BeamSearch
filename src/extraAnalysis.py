import pandas as pd
import matplotlib.pyplot as plt
from BeamSearch import BeamSearch
from ExtractGraph import ExtractGraph

'''
Used matplotlib and pandas for extra analysis
Not required in the main program
'''


def test_lambda():
    graph = ExtractGraph()
    beam_search = BeamSearch(graph)
    param_lambda_list = [x*0.01 for x in range(55,100,5)]
    param_lambda_list.insert(0,0.0)
    sentences = ['<s> In the past year',
                 '<s> It was at the',
                 '<s> He stressed that Indonesia',
                 '<s> It is estimated that the power reserve',
                 '<s> It particularly highlighted the fact that the external debt',
                 '<s> Secretary of State Warren Christopher']
    data_dict = dict(SentenceNum = [], LambdaValue = [], SentenceLength = [])
    for index, line in enumerate(sentences,1):
        for lam in param_lambda_list:
            # print(lam, line)
            sentence_prob = beam_search.beamSearchV2(line, 10, lam, 20)
            print('=' * 100)
            print(f"Parameter Lambda is {lam} and Context sentence is: [{line}]")
            print(f'Log Probability of sentence is {sentence_prob.score},\nCompleted sentence is [{sentence_prob.string}]')
            print(f'Length of sentence is {len(sentence_prob.string)}')
            print('='*100)
            data_dict['SentenceNum'].append(index)
            data_dict['LambdaValue'].append(lam)
            data_dict['SentenceLength'].append(len(sentence_prob.string))
        # if index > 1:
        #     break

    # pprint(data_dict)
    df = pd.DataFrame(data_dict)
    # print(df)
    df = df.pivot(index = 'LambdaValue', values='SentenceLength', columns='SentenceNum')
    # print(df)
    df.plot()
    plt.xlabel('Lambda Values')
    plt.ylabel('Sentence length')
    plt.title('Lambda vs sentence length')
    plt.savefig('Comparision.png')
    # plt.show()


if __name__ == '__main__':
    print('='*50)
    test_lambda()
