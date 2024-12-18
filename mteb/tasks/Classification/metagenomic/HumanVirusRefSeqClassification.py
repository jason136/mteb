from __future__ import annotations

import datasets
import random

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class HumanVirusRefSeqClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="HumanVirusRefSeqClassification",
        description="Classification of metagenomic sequences based on multiple attributes",
        reference="https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?SeqType_s=Nucleotide&VirusLineage_ss=taxid:10239&SourceDB_s=RefSeq&GenomeCompleteness_s=complete&Completeness_s=complete&HostLineage_ss=Homo%20sapiens%20(human),%20taxid:9606",
        dataset={
            "path": "jason136/human_virus_refseq",
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

    def _split_sequence(self, sequence: str, target_length: int = 200) -> list[str]:
        """Split a DNA sequence into chunks of approximately target_length base pairs.
        
        Args:
            sequence: The DNA sequence to split
            target_length: Approximate target length for each chunk
            
        Returns:
            List of sequence chunks
        """
        if len(sequence) <= target_length:
            return [sequence]
            
        chunks = []
        pos = 0
        while pos < len(sequence):
            chunk_length = int(target_length * random.uniform(0.8, 1.2))
            chunk_length = min(chunk_length, len(sequence) - pos)
            chunks.append(sequence[pos:pos + chunk_length])
            pos += chunk_length
        return chunks

    def dataset_transform(self):
        ds_split = self.dataset["test"]
        
        documents = []
        labels = []
        for seq, label in zip(ds_split["sequence"], ds_split["virus_name"]):
            seq_chunks = self._split_sequence(seq)
            documents.extend(seq_chunks)
            labels.extend([label] * len(seq_chunks))
        
        assert len(documents) == len(labels)

        rng = random.Random(42)  # local only seed
        pairs = list(zip(documents, labels))
        rng.shuffle(pairs)

        pairs = pairs[:5000]  # Limit to 5000 samples total

        documents, labels = [list(collection) for collection in zip(*pairs)]

        # Split into train (80%) and test (20%) sets
        split_idx = int(len(documents) * 0.8)
        
        ds = {
            "train": datasets.Dataset.from_dict({
                "text": documents[:split_idx],
                "label": labels[:split_idx],
            }),
            "test": datasets.Dataset.from_dict({
                "text": documents[split_idx:],
                "label": labels[split_idx:],
            })
        }

        self.dataset = datasets.DatasetDict(ds)
