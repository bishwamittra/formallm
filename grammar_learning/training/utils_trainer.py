from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from transformers import Trainer


class NoShuffleTrainer(Trainer):
    """Trainer variant that disables shuffle in the training DataLoader.

    Useful when the training order is managed externally (e.g. memorization-aware
    curriculum that updates dataset indices after each epoch).
    """

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.args.world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=False,
            )
        else:
            sampler = SequentialSampler(self.train_dataset)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
