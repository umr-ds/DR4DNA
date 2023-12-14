import functools
import itertools
import string
import numpy as np
from repair_algorithms.FileSpecificRepair import FileSpecificRepair
from googletrans import Translator
import language_tool_python
from collections import Counter
from Levenshtein import distance as levenshtein_distance

from repair_algorithms.PluginManager import PluginManager

nonprintable = itertools.chain(range(0x00, 0x20), range(0x7f, 0xa0))

lang_to_LanguageTool = {"en": "en-US", "de": "de-DE", "fr": "fr-FR", "es": "es-ES", "it": "it-IT", "pt": "pt-PT", }


class LangaugeToolTextRepair(FileSpecificRepair):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replace_ae_oe_ue = True
        self.analyzed_row = None
        self.error_matrix = None
        self.no_inspect_chunks = self.gepp.b.shape[0]
        self.tool = None
        self.lang = None
        self.no_columns_to_repair = None

    def set_use_header(self, use_header):
        self.use_header_chunk = use_header

    def set_no_inspect_chunks(self, *args, **kwargs):
        try:
            self.no_inspect_chunks = int(kwargs["c_ctx"].triggered[0]["value"])
        except:
            print("Error: could not set number of chunks to inspect")
        return {"updates_b": False, "refresh_view": False}

    @staticmethod
    def filter_nonprintable(text):
        import itertools
        # Use characters of control category
        nonprintable = itertools.chain(range(0x00, 0x20), range(0x7f, 0xa0))
        # Use translate to remove all non - printable characters
        return text.translate({character: None for character in nonprintable})

    def detect_language(self, *args, **kwargs):
        translator = Translator()
        start_pos = 1 if self.use_header_chunk else 0
        x = translator.detect(self.filter_nonprintable(
            "".join([chr(x) for x in self.gepp.b[start_pos:, :].reshape(-1)[self.gepp.b.shape[1]:1000]])))
        self.lang = x.lang
        return {"info": f"Detected language: {x.lang}", "updates_b": False, "refresh_view": False}
        # ( https://github.com/chenterry85/Language-Detection )

    def find_error_region(self, language=None, *args, **kwargs):
        if language is None:
            language = self.lang
        return self.find_error_region_by_words(language, *args, **kwargs)

    def find_error_region_by_words(self, language=None, *args, **kwargs):
        if language is None:
            language = self.lang
        if language is None:
            raise ValueError("No language selected. Detect language first!")
        # use a language dictonary to find incorrect words per chunk and find the most likely position of the error
        # across multiple chunks (pin down the position of the error)

        # WARNING: this method is rather slow but will yield better results than the character based method
        # IF the words stored in the file are in the used dictionary
        # change language according to "language" parameter:
        if self.tool is None or self.tool.language.normalized_tag != lang_to_LanguageTool.get(language, "en-US"):
            self.tool = language_tool_python.LanguageTool(lang_to_LanguageTool.get(language, "en-US"))
        # , config={'disabledRuleIds': "DROP_DOWN,SOME_OF_THE,THE_SUPERLATIVE,UPPERCASE_SENTENCE_START,DOPPELPUNKT_GROSS,KOMMA_ZWISCHEN_HAUPT_UND_NEBENSATZ_2,VIELZAHL_PLUS_SINGULAR,EMPFOHLENE_ZUSAMMENSCHREIBUNG,SEMIKOLON_VOR_ANFUEHRUNGSZEICHEN,DURCHEINANDER"})
        start_pos = 1 if self.use_header_chunk else 0

        blob = "".join(
            [chr(x) for x in
             self.gepp.b[start_pos:, :].reshape(-1)[0:self.no_inspect_chunks * self.gepp.b.shape[1]]])
        chars_appended = 0
        if self.no_inspect_chunks < self.gepp.b.shape[0]:
            # this is a fix for if the last word of the last row inspected spans to the next row and inspecting would
            # produce an error
            tmp = "".join(
                [chr(x) for x in
                 self.gepp.b[start_pos:, :].reshape(-1)[(self.no_inspect_chunks * self.gepp.b.shape[1]):(
                             (self.no_inspect_chunks + 1) * self.gepp.b.shape[1])]]).split(" ")[0]
            chars_appended = len(tmp)
            blob += tmp
        pos_correct = np.zeros(len(blob) + (start_pos * self.gepp.b.shape[1]), dtype=np.float32)
        matches = self.tool.check(blob)
        for matching_rule in matches:
            offset = matching_rule.offset
            # TODO: we might want to extend error_length to include non-printable characters behind the error,
            #  then iterate over range(error_length, error_length + len(offset)) to find the correct word
            error_length = matching_rule.errorLength
            if matching_rule.category == "TYPOS" or matching_rule.ruleIssueType == "misspelling":
                if self.replace_ae_oe_ue:
                    candidates = [x.replace("oe", "ö").replace("ae", "ä").replace("ue", "ü") for x in
                                  matching_rule.replacements]
                    un_escaped = matching_rule.matchedText.replace("oe", "ö").replace("ae", "ä").replace("ue", "ü")
                else:
                    candidates = matching_rule.replacements
                    un_escaped = matching_rule.matchedText
                if un_escaped in candidates or un_escaped.replace("ss", "ß") in candidates:
                    continue
                possible_substitutions = [x for x in candidates if len(x) == error_length]
                token = blob[offset:offset + error_length]
                if len(possible_substitutions) == 0:
                    # no possible substitution found
                    pos_correct[offset:offset + error_length] = 0.5
                else:
                    sort_func = functools.partial(lambda x, tkn: levenshtein_distance(x, tkn), tkn=token)
                    correction = \
                        sorted(possible_substitutions, key=sort_func, reverse=False)[0]
                    try:
                        pos_correct[offset:offset + error_length] = np.array(
                            [(ord(a) ^ ord(b)) for a, b in zip(token, correction)])
                    except:
                        print(f"Error while processing {matching_rule}")
            elif matching_rule.category == "EN_UNPAIRED_BRACKETS":
                pos_correct[offset:offset + error_length] = 0.5
            else:
                pos_correct[offset:offset + error_length] = 0.1
                print(f"Skipping rule {matching_rule}")
                # TODO check if there are further edgecases that we might want to handle

        # punish non-printable characters not yet found by rules:
        for i in range(len(blob)):
            filter_func = lambda x: (x not in string.printable) if language == "en-US" else (x in nonprintable)
            if filter_func(blob[i]) and pos_correct[i + (start_pos * self.gepp.b.shape[1])] == 0:
                pos_correct[i + (start_pos * self.gepp.b.shape[1])] = 0.5

        pos_correct = np.array(pos_correct[:-chars_appended]).reshape(-1, self.gepp.b.shape[1])
        # vstack the first row of self.gepp.b and pos_correct if start_pos == 1
        if start_pos == 1:
            pos_correct = np.vstack((np.zeros((1, self.gepp.b.shape[1]), dtype=np.float32), pos_correct))
        print(pos_correct)

        return pos_correct

    def find_incorrect_rows(self, *args, **kwargs):
        if kwargs is None or kwargs.get("chunk_tag") is None:
            self.chunk_tag = np.zeros(self.gepp.b.shape[0], dtype=np.int32)
        else:
            self.chunk_tag = kwargs.get("chunk_tag")
        """
        max_count, max_col = 0, 0
        max_counter = None
        for i, counter in enumerate(counters):
            most_non_zero_column = np.argmin([x.get(0.0) for x in self.get_column_counter()])
            for diff, count in counter.most_common(4):  # 0.0, 0.1, 0.5 and the most common real error...
                if diff < 1.0:
                    continue
                else:
                    if count > max_count:
                        max_col = i
                        max_count = count
                        max_counter = counter
        """
        incorrect_columns = [x for x in self.find_incorrect_columns()]
        tmp = sorted(incorrect_columns, key=lambda x: x[2], reverse=True)
        column_tags = [x[2] for x in incorrect_columns]
        for i in range(len(self.error_matrix)):
            if tmp[0][1] != 0.0 and self.error_matrix[i, tmp[0][0]] == tmp[0][1]:
                self.chunk_tag[i] = 1
        return {"chunk_tag": self.chunk_tag, "updates_b": False, "refresh_view": True, "column_tag": column_tags}

    def find_correct_rows(self, *args, **kwargs):
        # all rows with no errors according to the spellchecker are treated as correct!
        if kwargs is None or kwargs.get("chunk_tag") is None:
            self.chunk_tag = np.zeros(self.gepp.b.shape[0], dtype=np.int32)
        else:
            self.chunk_tag = kwargs.get("chunk_tag")

        res = []
        if self.error_matrix is None:
            self.error_matrix = self.find_error_region()
        # for each row: count all entrys != 0
        for i in range(len(self.error_matrix)):
            res.append(np.sum(self.error_matrix[i, :]) == 0)
        for i in range(1 if self.use_header_chunk else 0, min(len(self.chunk_tag), len(res))):
            if res[i]:
                self.chunk_tag[i] = 2
        return {"chunk_tag": self.chunk_tag, "updates_b": False, "refresh_view": True}

    def find_incorrect_columns(self, *args, **kwargs):
        column_counters = self.get_column_counter()
        for i, counter in enumerate(column_counters):
            exists_gr_zero = False
            for diff, count in counter.most_common(4):
                if diff < 1.0:
                    continue
                else:
                    exists_gr_zero = True
                    yield i, diff, count, counter
                    break
            if not exists_gr_zero:
                yield i, 0.0, 0, counter

    def get_incorrect_columns(self, *args, **kwargs):
        incorrect_columns = self.find_incorrect_columns()
        column_tags = [x[2] for x in incorrect_columns]
        return {"column_tag": column_tags, "updates_b": False, "refresh_view": True}

    def get_column_counter(self, *args, **kwargs):
        if self.error_matrix is None or self.analyzed_row != self.no_inspect_chunks:
            self.error_matrix = self.find_error_region(*args, **kwargs)
            self.analyzed_row = self.no_inspect_chunks
        avg_errors = []
        row_counters = []
        for i in range(self.gepp.b.shape[1]):
            avg_errors.append(np.mean(self.error_matrix[:, i]))
        for i in range(self.gepp.b.shape[1]):
            # avg_error = np.mean(self.error_matrix[:, i])
            ctr = Counter(self.error_matrix[:, i])
            row_counters.append(ctr)
        #    for key, value in ctr.items():
        #        pass
        return row_counters

    def repair(self, *args, **kwargs):
        if self.chunk_tag is None or sum(self.chunk_tag) == 0:
            self.find_error_region(*args, **kwargs)
            self.find_incorrect_rows()
        # np.bitwise_xor(self.gepp.b[i,j], most_common_difference[i])
        tmp = [x for x in self.find_incorrect_columns()]
        if self.no_columns_to_repair is None or self.no_columns_to_repair == 0:
            incorrect_columns = sorted(tmp, key=lambda x: x[2], reverse=True)
        else:
            incorrect_columns = sorted(tmp, key=lambda x: x[2], reverse=True)[:min(len(tmp), self.no_columns_to_repair)]
        for i in range(len(self.error_matrix)):
            if self.error_matrix[i, incorrect_columns[0][0]] == incorrect_columns[0][1]:
                tmp_b_i = self.gepp.b[i].copy()
                tmp_b_i[incorrect_columns[0][0]] = np.bitwise_xor(tmp_b_i[incorrect_columns[0][0]],
                                                                  int(incorrect_columns[0][1]))
                return {"updates_b": True, "repair": {"corrected_row": i, "corrected_value": tmp_b_i},
                        "refresh_view": True, "chunk_tag": self.chunk_tag}

    def is_compatible(self, meta_info):
        # parse magic info string:
        return meta_info == "data" or "Unicode text" in meta_info

    def get_ui_elements(self):
        return {"btn-textfile-lt-detect-language": {"type": "button", "text": "Detect language",
                                                    "callback": self.detect_language},
                "txt-textfile-lt-analyze-row-count": {"type": "int", "text": "Number of rows to analyze",
                                                      "callback": self.set_no_inspect_chunks},
                "btn-textfile-lt-find-incorrect-rows": {"type": "button", "text": "Tag incorrect rows",
                                                        "callback": self.find_incorrect_rows,
                                                        "updates_b": True},
                "btn-textfile-lt-find-correct-rows": {"type": "button", "text": "Tag correct rows",
                                                      "callback": self.find_correct_rows,
                                                      "updates_b": True},
                "btn-textfile-lt-find-columns": {"type": "button", "text": "Tag (in)correct columns",
                                                 "callback": self.get_incorrect_columns, "updates_b": False},
                "txt-textfile-lt-repair-column-count": {"type": "int", "text": "Columns to repair",
                                                        "callback": self.set_no_columns_to_repair},
                "btn-textfile-lt-repair": {"type": "button", "text": "Repair", "callback": self.repair,
                                           "updates_b": True}}

    def set_no_columns_to_repair(self, *args, **kwargs):
        try:
            self.no_columns_to_repair = int(kwargs["c_ctx"].triggered[0]["value"])
        except:
            print("Error: could not set number of columns to repair")
        return {"updates_b": False, "refresh_view": False}

    def update_chunk_tag(self, chunk_tag):
        super().update_chunk_tag(chunk_tag)
        self.error_matrix = None  # this could be speed-up?!

    def update_gepp(self, gepp):
        # invalidate error matrix:
        self.error_matrix = None
        self.gepp = self.semi_automatic_solver.decoder.GEPP
        # trigger recalculating the error matrix:
        # self.find_error_region()


mgr = PluginManager()
mgr.register_plugin(LangaugeToolTextRepair)
