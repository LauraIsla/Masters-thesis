import re

class MultiTagging:

    def __init__(self, mixed_file, output_file):
        self.mixed_file = mixed_file
        self.output_file = output_file
        self.lines = list()
        self.multi_tags = dict()
        self.key = list()
        self.value = list()

    def count_multialignments(self):
        with open(self.mixed_file, "r", encoding="utf-8") as mixed_file:
            self.lines = mixed_file.readlines()
            find_multi_tags = list()
            for line in self.lines:
                if re.findall("multi_tag=\w*-\w*", line) != []:
                    find_multi_tags.append("".join(re.findall("multi_tag=\w*-\w*", line)))
                if re.findall("multi_tag=\w*-\w*-\w*", line) != []:
                    find_multi_tags.append("".join(re.findall("multi_tag=\w*-\w*-\w*", line)))
                if re.findall("multi_tag=\w*-\w*-\w*-\w*", line) != []:
                    find_multi_tags.append("".join(re.findall("multi_tag=\w*-\w*-\w*-\w*", line)))

            for multi_tag in find_multi_tags:
                if multi_tag not in self.multi_tags:
                    self.multi_tags[multi_tag] = 1
                else:
                    self.multi_tags[multi_tag] += 1

            print("Multi-tag ranking:\n" + str(dict(sorted(self.multi_tags.items(), key=lambda item: item[1]))))
            print("\n")
            print("Amount of multi-tagged instances: " + str(sum(self.multi_tags.values())))
            print("\n")

            for multi, count in self.multi_tags.items():
                if count > 200:
                    new_multi = "en_tag=" + multi[10:]
                    self.multi_tags[multi] = new_multi
                elif count <= 200:
                    new_multi = "en_tag=MULTI"
                    self.multi_tags[multi] = new_multi

        return self.change_multi_tags_file()

    def change_multi_tags_file(self):
        with open(self.output_file, "w", encoding="utf-8") as multi_tag_modification_file:
            sentences = list()
            i = 1
            copy = False
            for line in self.lines:
                if line.startswith("# sent_id = " + str(i)):
                    print("New multi-tagged file is being written...")
                    sentences.append(line)
                    copy = True
                    continue
                if line.startswith("# sent_id = " + str(i+1)):
                    copy = False
                    continue
                elif copy:
                    i += 1
                    if "	".join(line.split()[9:]).split("|")[0] in self.multi_tags:
                        new_replacement = "	".join(line.split()[9:]).split("|")[0].replace("	".join(line.split()[9:]).split("|")[0], self.multi_tags["	".join(line.split()[9:]).split("|")[0]])
                        sentences.append("	".join(line.split()[:9]) + "	" + new_replacement + "|" + "|".join(("	".join(line.split()[9:]).split("|")[1:])).replace("multi_lex=", "en_lex=") + "\n")
                    else:
                        sentences.append(line)
                    i += 1

            multi_tag_modification_file.writelines(sentence for sentence in sentences)

if __name__ == "__main__":
    multi_tagging = MultiTagging("test.txt", "test_multi-tag.txt")
    multi_tagging.count_multialignments()

