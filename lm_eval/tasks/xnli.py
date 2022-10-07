from lm_eval.api.task import PromptSourceTask

class XNLIBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "xnli"
    DATASET_NAME = None
    SPLIT = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

class XNLI_EN(XNLIBase):
    DATASET_NAME = "en"

class XNLI_AR(XNLIBase):
    DATASET_NAME = "ar"

class XNLI_BG(XNLIBase):
    DATASET_NAME = "bg"

class XNLI_DE(XNLIBase):
    DATASET_NAME = "de"

class XNLI_EL(XNLIBase):
    DATASET_NAME = "el"

class XNLI_ES(XNLIBase):
    DATASET_NAME = "es"

class XNLI_FR(XNLIBase):
    DATASET_NAME = "fr"

class XNLI_HI(XNLIBase):
    DATASET_NAME = "hi"

class XNLI_RU(XNLIBase):
    DATASET_NAME = "ru"

class XNLI_SW(XNLIBase):
    DATASET_NAME = "sw"

class XNLI_TH(XNLIBase):
    DATASET_NAME = "th"

class XNLI_TR(XNLIBase):
    DATASET_NAME = "tr"

class XNLI_UR(XNLIBase):
    DATASET_NAME = "ur"

class XNLI_VI(XNLIBase):
    DATASET_NAME = "vi"

class XNLI_ZH(XNLIBase):
    DATASET_NAME = "zh"



LANGS = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
    "ur", "vi", "zh"
]

LANG_CLASSES = [
    XNLI_AR, XNLI_BG, XNLI_DE, XNLI_EL, XNLI_EN, XNLI_ES, XNLI_FR, XNLI_HI,
    XNLI_RU, XNLI_SW, XNLI_TH, XNLI_TR, XNLI_UR, XNLI_VI, XNLI_ZH
]

def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"xnli_{lang}"] = lang_class
    return tasks
