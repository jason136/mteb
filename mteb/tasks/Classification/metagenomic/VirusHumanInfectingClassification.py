from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class VirusHumanInfectingClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="VirusHumanInfectingClassification",
        description="A dataset for classification of human microbiome samples",
        dataset={
            "path": "jason136/virus_human_infecting",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2014-01-01", "2018-01-01"),
        eval_splits=["test"],
        eval_langs=["en"],
        main_score="accuracy",
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({"sequence": "text", "human_infecting": "label"})
        
        test_dataset = self.dataset['test']

        train_test = test_dataset.train_test_split(test_size=0.5, seed=42)

        train_test['train'] = train_test['train'].select(range(min(len(train_test['train']), 5000)))
        train_test['test'] = train_test['test'].select(range(min(len(train_test['test']), 5000)))

        self.dataset['train'] = train_test['train']
        self.dataset['test'] = train_test['test']
