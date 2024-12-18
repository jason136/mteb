from __future__ import annotations

import datasets
import random
from collections import Counter

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HMPReferenceGenomesClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HMPReferenceGenomesClassification",
        description="Classification of metagenomic sequences based on multiple attributes",
        reference="https://www.ncbi.nlm.nih.gov/bioproject/28331",
        dataset={
            "path": "jason136/hmp_reference_genomes",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=[],
        main_score="accuracy",
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
        
        train_documents = []
        train_labels = []
        test_documents = []
        test_labels = []
        
        random.seed(42)
        
        # Process each sequence
        for sequence, organism in zip(ds_split["sequence"], ds_split["organism"]):
            chunks = []
            target_size = 200
            target_overlap = 50
            
            # Calculate variance (20% of target size)
            size_std = target_size * 0.2
            overlap_std = target_overlap * 0.2
            
            current_pos = 0
            while current_pos < len(sequence):
                chunk_size = int(random.gauss(target_size, size_std))
                
                if current_pos + chunk_size <= len(sequence):
                    chunk = sequence[current_pos:current_pos + chunk_size]
                    chunks.append(chunk)
                
                overlap = int(random.gauss(target_overlap, overlap_std))
                current_pos += chunk_size - overlap
            
            random.shuffle(chunks)
            split_idx = int(len(chunks) * 0.8)
            
            train_documents.extend(chunks[:split_idx])
            train_labels.extend([organism] * len(chunks[:split_idx]))
            test_documents.extend(chunks[split_idx:])
            test_labels.extend([organism] * len(chunks[split_idx:]))
        
        ds = {
            "train": datasets.Dataset.from_dict({
                "text": train_documents[:2000],
                "label": train_labels[:2000],
            }),
            "test": datasets.Dataset.from_dict({
                "text": test_documents[:2000],
                "label": test_labels[:2000],
            }),
        }
        
        self.dataset = datasets.DatasetDict(ds)
