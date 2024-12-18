from __future__ import annotations

import datasets
import random

from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.TaskMetadata import TaskMetadata


class HMPDemonstrationProjectsClustering(AbsTaskClustering):
    """Base class for HMP clustering tasks"""
    metadata = TaskMetadata(
        name="HMPDemonstrationProjectsClustering",
        description="A dataset for classification of human microbiome samples",
        reference="https://www.ncbi.nlm.nih.gov/bioproject/46305",
        dataset={
            "path": "jason136/hmp_demonstration_projects",
            "revision": "359c35411c400a6350f2fa49f45010f5639aeb11",
        },
        type="Clustering",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=[],
        main_score="v_measure",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
    )

    def dataset_transform(self):
        ds_split = self.dataset["test"]

        documents = list(ds_split["sequence"])
        labels = [f"{disease}_{host_sex}_{isolation_source}_{analyte_type}" for disease, host_sex, isolation_source, analyte_type in zip(ds_split["disease"], ds_split["host_sex"], ds_split["isolation_source"], ds_split["analyte_type"])]

        unique_labels = list(set(labels))
        label_indices = [unique_labels.index(label) for label in labels]

        assert len(documents) == len(label_indices)

        rng = random.Random(42)  # local only seed
        pairs = list(zip(documents, label_indices))
        rng.shuffle(pairs)

        pairs = pairs[:5000]

        documents, label_indices = [list(collection) for collection in zip(*pairs)]

        batch_size = 128
        batched_documents = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        batched_labels = [label_indices[i:i + batch_size] for i in range(0, len(label_indices), batch_size)]

        ds = {
            "test": datasets.Dataset.from_dict(
            {
                "sentences": batched_documents,
                "labels": batched_labels,
            })
        }

        self.dataset = datasets.DatasetDict(ds)
