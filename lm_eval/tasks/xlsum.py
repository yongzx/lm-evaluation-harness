from lm_eval.api.task import PromptSourceTask
import typing 

class XLSumBase(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "csebuetnlp/xlsum"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def max_generation_length(self):
        return 64

class XLSumAz(XLSumBase):
    DATASET_NAME = "azerbaijani"

class XLSumAm(XLSumBase):
    DATASET_NAME = "amharic"

class XLSumAr(XLSumBase):
    DATASET_NAME = "arabic"

class XLSumBn(XLSumBase):
    DATASET_NAME = "begali"

class XLSumMy(XLSumBase):
    DATASET_NAME = "burmese"

class XLSumZh(XLSumBase):
    DATASET_NAME = "chinese_simplified"

class XLSumZhCht(XLSumBase):
    DATASET_NAME = "chinese_traditional"

class XLSumEn(XLSumBase):
    DATASET_NAME = "english"

class XLSumFr(XLSumBase):
    DATASET_NAME = "french"

class XLSumGu(XLSumBase):
    DATASET_NAME = "gujarati"

class XLSumHa(XLSumBase):
    DATASET_NAME = "hausa"

class XLSumHi(XLSumBase):
    DATASET_NAME = "hindi"

# igbo
# indonesian
# japanese
# kirundi
# korean
# kyrgyz
# marathi
# nepali
# oromo
# pashto
# persian
# pidgin
# portuguese
# punjabi
# russian
# scottish_gaelic
# serbian_cyrillic
# serbian_latin
# sinhala
# somali
# spanish
# swahili
# tamil
# telugu
# thai
# tigrinya
# turkish
# ukrainian
# urdu
# uzbek
# vietnamese
# welsh
# yoruba

WIKILINGUA_TASKS = [
    XLSumAz,
]


def construct_tasks() -> typing.Dict[str, XLSumBase]:
    """
    Returns a dictionary of tasks keyed by task name, for example:
        "GEM/wiki_lingua_ar"
    will dispatch to the GEM WikiLingua Arabic class.
    """
    tasks = {}
    for task_class in WIKILINGUA_TASKS:
        lang = task_class.DATASET_NAME
        tasks[f"xlsum_{lang}"] = task_class
    return tasks
