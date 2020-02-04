import ExtractGraph
import BeamSearch

# Extract graph from assign1_sentences.txt.
graph = ExtractGraph.ExtractGraph()
# Test extraction correctness.
head_word = "<s>"
tail_word = "Water"
print("The log_probability of \"" + tail_word + "\" appearing after \"" + head_word + "\" is " + str(graph.getProb(head_word, tail_word)))
head_word = "Water"
tail_word = "<s>"
print("The log_probability of \"" + tail_word + "\" appearing after \"" + head_word + "\" is " + str(graph.getProb(head_word, tail_word)))
head_word = "planned"
tail_word = "economy"
print("The log_probability of \"" + tail_word + "\" appearing after \"" + head_word + "\" is " + str(graph.getProb(head_word, tail_word)))
head_word = "."
tail_word = "</s>"
print("The log_probability of \"" + tail_word + "\" appearing after \"" + head_word + "\" is "
				+  str(graph.getProb(head_word, tail_word)))

# Find the sentence with highest log_probability using basic beam search.
beam_search = BeamSearch.BeamSearch(graph)
sentence_prob = beam_search.beamSearchV1("<s>", 10, 20)
print(str(sentence_prob.score) + "\t" + sentence_prob.string)
sentence_prob = beam_search.beamSearchV1("<s> Israel and Jordan signed the peace", 10, 40)
print(str(sentence_prob.score) + "\t" + sentence_prob.string)
sentence_prob = beam_search.beamSearchV1("<s> It is", 10, 15)
print(str(sentence_prob.score) + "\t" + sentence_prob.string)

# Find the sentence with highest log_probability using beam search with sentence length-normalzation.
param_lambda = 0.7
beam_search = BeamSearch.BeamSearch(graph)
sentence_prob = beam_search.beamSearchV2("<s>", 10, param_lambda, 20)
print(str(sentence_prob.score) + "\t" + sentence_prob.string)
sentence_prob = beam_search.beamSearchV2("<s> Israel and Jordan signed the peace", 10, param_lambda, 40)
print(str(sentence_prob.score) + "\t" + sentence_prob.string)
sentence_prob = beam_search.beamSearchV2("<s> It is", 10, param_lambda, 15)
print(str(sentence_prob.score) + "\t" + sentence_prob.string)


#########################################################################################
## MY Work
#########################################################################################
# Find the sentence with highest log_probability using beam search with sentence length-normalzation.

# param_lambda = 0.7
# bwidth = 5
# beam_search = BeamSearch.BeamSearch(graph)
# sentence_prob = beam_search.beamSearchV2("<s>", bwidth, param_lambda, 20)
# print(str(sentence_prob.score) + "\t" + sentence_prob.string)
# sentence_prob = beam_search.beamSearchV2("<s> Israel and Jordan signed the peace", bwidth, param_lambda, 40)
# print(str(sentence_prob.score) + "\t" + sentence_prob.string)
# sentence_prob = beam_search.beamSearchV2("<s> It is", bwidth, param_lambda, 15)
# print(str(sentence_prob.score) + "\t" + sentence_prob.string)
