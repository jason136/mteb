from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HumanMicrobiomeProjectClassificationTest(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HumanMicrobiomeProjectClassificationTest",
        description="A dataset for classification of human microbiome samples",
        dataset={
            "path": "jason136/hmp_test",
            "revision": "afd5fe5f61514378ea1d0e804e1dee1cd39b08b8",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["train"],
        eval_langs=["en"],
        main_score="accuracy",
        domains=["Medical", "Academic"],
        task_subtypes=["Topic classification"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"sequence": "text", "source": "label"})
