# MetaFTM

MetaFTM is a focused topic model that leverages metadata such as document labels and word embeddings, which enjoys improved performance on short text topic modelling.

The code of MetaFTM extends the paper "A Word Embeddings Informed Focused Topic Model" in ACML 2017 [PDF](http://proceedings.mlr.press/v77/zhao17a/zhao17a.pdf).
The original paper introduced focusing on the topic-word distributions, i.e., Phi, and the code applies a similar model construction that leverages document metadata to introduce focusing on the document-topic distributions, i.e., Theta.


# Run MetaFTM

1. Requirements: Matlab 2016b (or later).

3. We have offered the WS dataset used in the paper, which is stored in MAT format, with the following contents:
- train_doc: the sparse representation of the training documents, each row is a triplet [a, b, c], where a, b, c are document index, word index, and word occurrences, respectively.
- train_label: the one-hot representation of the document labels.
- test_doc_A/B: the first/second half of the testing documents, with the same format of train_doc
- voc: vocabulary of the dataset
- word_embeddings: the pretrained word embeddings of [GloVe](https://nlp.stanford.edu/projects/glove/)
- label_name: the name of the document labels

Please prepare your own documents in the above format. If you want to use this dataset, please cite the original papers, which are cited in our paper.

4. Run ```demo.m```.

5. Important parameters:
- alpha_option: use 'meta' if you want to turn on focusing of the document-topic distributions informed by the document labels.
- beta_option: use 'meta' if you want to turn on focusing of the topic-word distributions informed by the word embeddings.
- K: the number of topics

6. Outputs:

The code saves the metric of perplexity on the training/testing documents, as well as some statistics of the model, in a MAT file named './save/model.mat'.

The code also prints the top words for topics saved in './save/top_words.txt'.

# Notes

1. To reproduce the results of our [paper](http://proceedings.mlr.press/v77/zhao17a/zhao17a.pdf), use alpha_option="off" and beta_option="meta".  

2. For the Polya-Gamma sampler (```PolyaGamRnd_Gam.m```), I used one of the [DSBN](https://github.com/zhegan27/dsbn_aistats2015). If you want to use the sampler, please cite the [paper](http://proceedings.mlr.press/v38/gan15.html).
