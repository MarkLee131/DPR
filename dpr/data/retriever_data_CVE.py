import collections
import csv
import json
import logging
import pickle
from typing import Dict, List

import hydra
import jsonlines
import torch
from omegaconf import DictConfig

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
    get_dpr_files,
    read_nq_tables_jsonl,
    split_tables_to_chunks,
)

from dpr.utils.data_utils import normalize_question

logger = logging.getLogger(__name__)

TableChunk = collections.namedtuple("TableChunk", ["text", "title", "table_id"])

class QASample:
    def __init__(self, query: str, id, answers: List[str]):
        self.query = query
        self.id = id
        self.answers = answers


class RetrieverData(torch.utils.data.Dataset):
    def __init__(self, file: str):
        """
        :param file: - real file name or the resource name as they are defined in download_data.py
        """
        self.file = file
        self.data_files = []

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        assert (
            len(self.data_files) == 1
        ), "RetrieverData source currently works with single files only. Files specified: {}".format(self.data_files)
        self.file = self.data_files[0]



class QASrc(RetrieverData):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file)
        self.data = None
        self.selector = hydra.utils.instantiate(selector) if selector else None
        self.special_query_token = special_query_token
        self.query_special_suffix = query_special_suffix

    def __getitem__(self, index) -> QASample:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _process_question(self, question: str):
        # as of now, always normalize query
        question = normalize_question(question)
        if self.query_special_suffix and not question.endswith(self.query_special_suffix):
            question += self.query_special_suffix
        return question
    
class CsvQASrc(QASrc):
    def __init__(
        self,
        file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
        data_range_start: int = -1,
        data_size: int = -1,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_col = question_col
        self.answers_col = answers_col
        self.id_col = id_col
        self.data_range_start = data_range_start
        self.data_size = data_size

    def load_data(self):
        super().load_data()
        data = []
        start = self.data_range_start
        # size = self.data_size
        samples_count = 0
        # TODO: optimize
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[self.question_col]
                answers = eval(row[self.answers_col])
                id = None
                if self.id_col >= 0:
                    id = row[self.id_col]
                samples_count += 1
                # if start !=-1 and samples_count<=start:
                #    continue
                data.append(QASample(self._process_question(question), id, answers))

        if start != -1:
            end = start + self.data_size if self.data_size != -1 else -1
            logger.info("Selecting dataset range [%s,%s]", start, end)
            self.data = data[start:end] if end != -1 else data[start:]
        else:
            self.data = data
