import itertools
import re
from collections import defaultdict, OrderedDict, Counter
from itertools import groupby

from nltk.tokenize import word_tokenize
from tqdm import tqdm

class SentenceExtraction:

    def __init__(self):
        self.alignments_dictionaries = dict()
        self.src_lengths = list()
        self.trg_lengths = list()
        self.source_sentences_dictionaries = defaultdict(dict)
        self.source_tags = list()
        self.target_sentences_dictionaries = defaultdict(dict)
        self.target_sentences_lengths = list()
        self.target_tags = list()
        self.src_trg_mapping = defaultdict(list)
        self.src_word_trg_pos = defaultdict(list)
        self.changed_dictionaries = dict()
        self.testing_list_lengths = list()
        self.source_indexes_lengths = list()
        self.target_indexes_lengths = list()
        self.changed_dictionaries = dict()
        self.testing_list_lengths_2 = list()
        self.sorted_dict = dict()

    def extract_alignments(self, alignment_file = "train_en-zh_.src-mt.alignments"):
        """ Obtain the alignments from the alignment file.

        Args:
            alignment_file: text file with strings

        Returns
            dict: key = sentence_id; value = alignments for that particular sentence

        """
        with open(alignment_file, "r") as alignments:
            alignments = alignments.readlines()
            temporal_dictionary = defaultdict(dict)
            key_aligned = 0
            for alignment in alignments:
                if key_aligned not in temporal_dictionary.keys():
                    temporal_dictionary[key_aligned] = tuple(aligned.split("-") for aligned in alignment[0:-1].split(" "))
                    key_aligned += 1
                else:
                    temporal_dictionary[key_aligned] = tuple(aligned.split("-") for aligned in alignment[0:-1].split(" "))
            key = 0
            for key_aligned, value_alignment in temporal_dictionary.items():
                if key not in self.alignments_dictionaries.keys():
                    #tuple_to_dict = dict((int(digit_src), int(digit_trg)) for digit_src, digit_trg in value_alignment)
                    self.alignments_dictionaries[key] = value_alignment
                    key += 1
                else:
                    self.alignments_dictionaries[key] = value_alignment
            return self.alignments_dictionaries

    def extract_lengths(self):
        with open("train_en-zh_.src", "r", encoding="utf-8") as src_lengths, open("train_en-zh_.mt", "r", encoding="utf-8") as trg_lengths:
            src_sentences = src_lengths.readlines()
            trg_sentences = trg_lengths.readlines()

            for src_sentence in src_sentences:
                self.src_lengths.append(len(src_sentence.split()))
            print(sum(self.src_lengths)/len(self.src_lengths))
            for trg_sentence in trg_sentences:
                self.trg_lengths.append((len(trg_sentence.split())))
            print(sum(self.trg_lengths)/len(self.trg_lengths))

    def extract_source(self, source_file="task1_en-zh.train_src.conll"):
        """ Obtain the source words and their indexes from the source file.

        Args:
            source_file: text file where sentence words are divided into different lines

        Returns
            dict: key = sentence_id; value = src_word_index: (src_word, src_tag)

        """
        with open(source_file, "r", encoding="utf-8") as src_alignments:
            src_alignments = src_alignments.readlines()
            unwanted_words = ["sent_id", "newdoc", "newpar"]
            source_indexes = []
            source_sentences_lengths = list()
            i = 0
            for src_line in src_alignments:
                #print(re.findall("lex=[+\-\^,a-z]*", "".join(src_line.split()[9:10])))
                for word in src_line.split()[:1]:
                    if word.isdigit():
                        source_indexes.append(int(word))
                        self.source_tags += [(int(word), (source_word, ("en_tag=" + source_tag, "".join(re.findall("lex=[+\-\^,a-z]*", "".join(src_line.split()[9:10]))).replace("lex=", "|en_lex=")))) for source_word in src_line.split()[1:2] for source_tag in src_line.split()[3:4] if source_word not in unwanted_words]

            for digit_1, digit_2 in zip(source_indexes, source_indexes[1:]):
                if digit_1 > digit_2:
                    source_sentences_lengths.append(digit_1)

            iter_list = iter(self.source_tags)
            sliced = [list(itertools.islice(iter_list, 0, length)) for length in self.src_lengths]
            #self.source_tags
            print(len(sliced))
            for sliced_list in sliced:
                self.source_sentences_dictionaries[i] = dict(sliced_list)
                i += 1

    def extract_target(self, target_file="task1_en-zh.train_trg.conll"):
        """ Obtain the target words and their indexes from the target file.

        Args:
            target_file: text file where sentence words are divided into different lines

        Returns
            dict: key = sentence_id; value = trg_word_index: (trg_word, trg_tag)

        """
        with open(target_file, "r", encoding="utf-8") as trg_alignments:
            trg_alignments = trg_alignments.readlines()
            unwanted_words = ["sent_id", "newdoc", "newpar"]
            target_indexes = []
            i = 0
            for trg_line in trg_alignments:
                for word in trg_line.split()[:1]:
                    if word.isdigit():
                        target_indexes.append(int(word))
                        self.target_tags += [(int(word), (target_word, (target_tag, "".join(re.findall("lex=[+\-\^,a-z]*", "".join(trg_line.split()[9:10])))))) for target_word in trg_line.split()[1:2] for target_tag in trg_line.split()[3:4] if target_word not in unwanted_words]

            for digit_1, digit_2 in zip(target_indexes, target_indexes[1:]):
                if digit_1 > digit_2:
                    self.target_sentences_lengths.append(digit_1)

            iter_list = iter(self.target_tags)
            sliced = [list(itertools.islice(iter_list, 0, length)) for length in self.trg_lengths]
            #self.target_tags
            print(len(sliced))
            for sliced_list in sliced:
                self.target_sentences_dictionaries[i] = dict(sliced_list)
                i += 1

    def map_source_target(self):
        """ Map source words with their corresponding target words based on the self.alignment_dictionaries
        obtained in the extract_alignments function as well as the self.source_sentences_dictionaries and
        self.target_sentences.

        Args:
            __init__ args

        Returns
            dict: key = sentence_id; value = {(src_index, trg_index): (src_word, trg_word)}
            dict: key = sentence_id + trg_index + trg word; value = src_pos_tag

        """
        alignments_values = self.alignments_dictionaries.values()
        testing_list = list()
        source_indexes = defaultdict(list)
        target_indexes = defaultdict(list)

        i = 0
        alignment_id = 0

        source_sentences_values = list(self.source_sentences_dictionaries.values())
        target_sentences_values = list(self.target_sentences_dictionaries.values())
        for sentence_id, value_alignment in enumerate(alignments_values):
            for index_key, index_value in value_alignment:
                source_indexes[sentence_id].append(int(index_key) + 1)
                target_indexes[sentence_id].append(int(index_value) + 1)
                for index_src, (key_src, value_src) in enumerate(source_sentences_values[sentence_id].items()):
                    for index_trg, (key_trg, value_trg) in enumerate(target_sentences_values[sentence_id].items()):
                        if int(index_key) + 1 == int(key_src) and int(index_value) + 1 == int(key_trg):
                            self.src_trg_mapping[int(sentence_id)].append({(key_src, key_trg): (value_src, value_trg)})
                            ### sent_id = " + str(int(sentence_id) + 1) + "\n"
                            alignment_id += 1
                            #testing_list.append(str(int(key_trg)) + " " + value_trg[0] + " " + value_src[1])
                            self.src_word_trg_pos["{} {} {}".format(int(sentence_id) + 1, int(key_trg), value_trg[0])].append(value_src[1])
                            #self.src_word_trg_pos["# sent_id = " + str(int(sentence_id) + 1) + " " + str(int(key_trg)) + " " + value_trg[0]].append(value_src[1])

        for sentence_ids in self.src_word_trg_pos.keys():
            testing_list.append(int(" ".join(sentence_ids.split()[:1])))
        self.testing_list_lengths = [len(list(group)) for key, group in groupby(testing_list)]

        it = iter(list(self.src_word_trg_pos.items()))
        sliced_lists = [[next(it) for _ in range(0, size)] for size in self.testing_list_lengths]

        for sliced_list in sliced_lists:
            self.changed_dictionaries[i] = dict(sliced_list)
            i += 1

        for sentence_id, alignments in source_indexes.items():
            self.source_indexes_lengths.append(len(alignments))
            for key_src, value_src in source_sentences_values[sentence_id].items():
                if int(key_src) not in alignments:
                    self.src_trg_mapping[int(sentence_id)].append({(key_src, "UNK"):(value_src, "(UNK)")})

        for sentence_id, alignments in target_indexes.items():
            self.target_indexes_lengths.append(len(alignments))
            for key_trg, value_trg in target_sentences_values[sentence_id].items():
                if int(key_trg) not in alignments:
                    self.changed_dictionaries[int(sentence_id)]["{} {} {}".format(int(sentence_id) + 1, int(key_trg), value_trg[0])] = ["UNK"]
                    self.src_trg_mapping[int(sentence_id)].append({("UNK", key_trg):("(UNK)", value_trg)})

        self.testing_list_lengths_2 = [len(self.changed_dictionaries[sentence_id].values()) for sentence_id, value in self.changed_dictionaries.items()]
        #return self.map_unk_src_tokens(self.src_trg_mapping)

    def change_pos_tags_trg(self):
        sliced = list()
        i = 0
        for sentence, dictionary in self.changed_dictionaries.items(): #sorts the dictionary items by key
            self.sorted_dict[sentence] = dict(OrderedDict(sorted(dictionary.items(), key=lambda x: int("".join(x[0].split()[1])))).items())
            #self.changed_dictionaries[sentence].update(OrderedDict(sorted(dictionary.items(), key=lambda x: int("".join(x[0].split()[1])))))

        for (key, value), (key_2, value_2) in zip(self.target_sentences_dictionaries.items(), self.sorted_dict.items()):
            for (index, target), element in zip(value.items(), value_2.items()):
                if index == int(element[0].split()[1]):
                    target = (target[0], element[1])
                    sliced.append((element[0].split()[1], target))
            #     else:
            #         target = (target[0], element[1])
            #         print(target)
                #elif target[0] not in corpus[1][0]:
                    #     target = (target[0], corpus[0][1])
                    #     sliced.append((index, target))
                #for alignment in element:
                    #     target = (target[0], corpus[0][1])
                    # elif target[0] not in corpus[1][0]:
                    #     target = (target[0], corpus[0][1])
                    #     sliced.append((index, target))

        iter_list = iter(sliced)
        #iter_list = iter(self.sorted_dict)
        sliced_2 = [list(itertools.islice(iter_list, 0, length)) for length in self.testing_list_lengths_2]

        for sliced_list in sliced_2:
            self.changed_dictionaries[i] = dict(sliced_list)
            i += 1

    def counter_multialignments(self):
        counter = 0
        print(self.changed_dictionaries)
        for key, value in self.changed_dictionaries.items():
            for index, align in value.items():
                if len(align[1]) >= 2:
                    counter += 1
        return counter

    def extract_source_features(self):
        """ Extract keys and values from self.src_word_trg_pos.items() to look for the corresponding lines in the conll
        file, and the "en_tag = " at the end of the line.

        For instance, if we extract:

        Sentence_Id: 1
        Word_Id: 2
        Trg_Word: Teil

        The self.add_source_pos_tags function should look for the line containing these parameters to then add the
        Src_Tag, "|en_tag=NOUN", at the end (or at some index in the middle) of the corresponding line.
        """
        for sentence_id, alignment in zip(self.changed_dictionaries.keys(), self.changed_dictionaries.values()):
            self.add_source_pos_tags("# sent_id = " + str(int(sentence_id) + 1) + "\n", alignment)
        #for sentence_idx, src_tag in zip(self.src_word_trg_pos.keys(), self.src_word_trg_pos.values()):
            #self.add_source_pos_tags(sentence_idx, src_tag)
        #for sentence_idx, src_tag in self.src_trg_mapping.items():
            #for src in src_tag:
                #for key, value in src.items():
                    #print(sentence_idx)
                    #self.add_source_pos_tags(sentence_idx, key[1], value[1][0], value[1][1])

    #, word_id, pos_word, pos_tag
    def add_source_pos_tags(self, sentence_id, src_tag):
        """

        :param sentence_id: Sentence Index
        :param word_id: Word Index
        :param pos_word: Trg Word
        :param pos_tag: Src Tag

        :return: file containing target lines with the added src_tags.

        2	Teil	Teil	NOUN	NN	Number=Sing|Person=3	_	_	_	freq=p99|gap_tag=OK|lex=-lexambig,-posambig,^syncretic|tag=OK|en_tag=NOUN
        """
        with open("task1_en-zh.train_trg.conll", "r", encoding="utf-8") as test, open("en_zh_train_set_mixed_test_good.txt", "a", encoding="utf-8") as testing:
            conll = test.readlines()
            copy = False
            for line in conll:
                if line.startswith(sentence_id):
                    print(line)
                    testing.writelines("\n")
                    testing.writelines(line)
                    copy = True
                    continue
                if line.startswith(str(" ".join(sentence_id.split()[:3]) + " " + str(int(sentence_id.split()[3]) + 1))):
                    copy = False
                    continue
                elif copy:
                    for index, (src_word, src_tag_2) in src_tag.items():
                        global result
                        if src_tag_2 != ["UNK"]:
                            if len(src_tag_2) == 1:
                                result = ["".join(tup) for tup in src_tag_2]
                            elif len(src_tag_2) == 2:
                                multi_tags_two = src_tag_2[0][0].replace("en_tag=", "multi_tag=") + src_tag_2[1][0].replace("en_tag=", "-")
                                if len(re.findall("\+|\^", src_tag_2[0][1])) > len(re.findall("\+|\^", src_tag_2[1][1])):
                                    multi_lex_two = src_tag_2[0][1]
                                else:
                                    multi_lex_two = src_tag_2[1][1]
                                result = [multi_tags_two + multi_lex_two.replace("|en_lex=", "|multi_lex=")]
                            elif len(src_tag_2) == 3:
                                multi_tags_three = src_tag_2[0][0].replace("en_tag=", "multi_tag=") + src_tag_2[1][0].replace("en_tag=", "-") + src_tag_2[2][0].replace("en_tag=", "-")
                                if len(re.findall("\+|\^", src_tag_2[0][1])) > len(re.findall("\+|\^", src_tag_2[1][1])):
                                    multi_lex_three = src_tag_2[0][1]
                                elif len(re.findall("\+|\^", src_tag_2[1][1])) < len(re.findall("\+|\^", src_tag_2[2][1])):
                                    multi_lex_three = src_tag_2[2][1]
                                else:
                                    multi_lex_three = src_tag_2[1][1]
                                result = [multi_tags_three + multi_lex_three.replace("|en_lex=", "|multi_lex=")]
                            elif len(src_tag_2) == 4:
                                multi_tags_four = src_tag_2[0][0].replace("en_tag=", "multi_tag=") + src_tag_2[1][0].replace("en_tag=", "-") + src_tag_2[2][0].replace("en_tag=", "-") + src_tag_2[3][0].replace("en_tag=", "-")
                                if len(re.findall("\+|\^", src_tag_2[0][1])) > len(re.findall("\+|\^", src_tag_2[1][1])):
                                    multi_lex_four = src_tag_2[0][1]
                                elif len(re.findall("\+|\^", src_tag_2[1][1])) < len(re.findall("\+|\^", src_tag_2[2][1])):
                                    multi_lex_four = src_tag_2[2][1]
                                elif len(re.findall("\+|\^", src_tag_2[2][1])) < len(re.findall("\+|\^", src_tag_2[3][1])):
                                    multi_lex_four = src_tag_2[3][1]
                                else:
                                    multi_lex_four = src_tag_2[1][1]
                                result = [multi_tags_four + multi_lex_four.replace("|en_lex=", "|multi_lex=")]
                        #if src_tag_2 != ["UNK"]:
                            #print(src_tag_2)
                            #if len(src_tag_2) == 1:
                                #result = ["".join(tup) for tup in src_tag_2]
                            #elif len(src_tag_2) == 2:
                                #result = ["multi=" + src_tag_2[0][0].replace("en_tag=", "") + "-" + src_tag_2[1][0].replace("en_tag=", "") + "|en_lex=" + src_tag_2[0][1].replace("|en_lex=", "") + "," + src_tag_2[1][1].replace("|en_lex=", "")]
                            #elif len(src_tag_2) == 3:
                                #result = ["multi=" + src_tag_2[0][0].replace("en_tag=", "") + "-" + src_tag_2[1][0].replace("en_tag=", "") + "-" + src_tag_2[2][0].replace("en_tag=", "") + "|en_lex=" + src_tag_2[0][1].replace("|en_lex=", "") + "," + src_tag_2[1][1].replace("|en_lex=", "")]
                            #elif len(src_tag_2) == 4:
                                #result = ["multi=" + src_tag_2[0][0].replace("en_tag=", "") + "-" + src_tag_2[1][0].replace("en_tag=", "") + "-" + src_tag_2[2][0].replace("en_tag=", "") + "-" + src_tag_2[3][0].replace("en_tag=", "") + "|en_lex=" + src_tag_2[0][1].replace("|en_lex=", "")  + "," + src_tag_2[1][1].replace("|en_lex=", "")]
                        else:
                            result = ["en_tag=UNK"]

                        try:
                            if line.split()[0] == index and line.split()[1] == src_word:
                                #print(("	".join(line.split()[:9]) + "	" + "|".join(result) + "|" + line.split()[9] + "\n"))
                                testing.writelines(("	".join(line.split()[:9]) + "	" + "|".join(result) + "|" + line.split()[9] + "\n"))
                        except IndexError:
                           continue

                #         try:
                #             if line.split()[0] == src.split()[0] and line.split()[1] == src.split()[1]:
                #                 testing.writelines("	".join(line.split()[:9]) + "	" + "en_tag=" + src.split()[2] + "|" + line.split()[9] + "\n")
                #         except IndexError:
                #             continue
                #     if line.startswith(str(int(word_id) + 1) + "	" + pos_word):
                #         print(line)

if __name__ == "__main__":
    sentence_extractor = SentenceExtraction()
    sentence_extractor.extract_alignments()
    sentence_extractor.extract_lengths()
    sentence_extractor.extract_source()
    sentence_extractor.extract_target()
    sentence_extractor.map_source_target()
    #print(sentence_extractor.src_trg_mapping)
    sentence_extractor.change_pos_tags_trg()
    #print(sentence_extractor.counter_multialignments())
    sentence_extractor.extract_source_features()

    # for sentence_id, index in sentence_extractor.source_sentences_dictionaries.items():
    #     for word_tag in index.values():
    #         for element in word_tag[1]:
    #             if element == "|en_lex=+posambig,-lexambig,^syncretic":
    #                 print(element)