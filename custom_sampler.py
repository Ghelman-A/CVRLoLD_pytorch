from torch.utils.data import BatchSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Iterator, List, Optional, TypeVar

T_co = TypeVar('T_co', covariant=True)


class CustomBatchSampler(BatchSampler):
    """
        This custom sampler is used to implement the idea of extracting multiple pairs of clips from each video
        in each epoch, instead of simply extracting one pair. The reason for doing this is that in the latter case, 
        given the small number of videos in the training set and the collected dataset in general, we are only
        using a few samples per epoch. Which is not enough for training a 3D ResNet network.
        
        By using this sampler, in each epoch we can use multiple extracted clip-pairs from each video for training,
        which is specified by the comb_count parameter refering to the number of combinations used.

    Args:
        BatchSampler (_type_): _description_
    """
    def __init__(self, comb_count: int, sampler: DistributedSampler[int], batch_size: int, drop_last: bool) -> None:
        super().__init__(sampler, batch_size, drop_last)        
        self.comb_count = comb_count

    def __len__(self) -> int:
        """
            This custom batch sampler operates over a distributed sampler, and length returned by the distributed
            sampler is the number of indices assigned to each process (GPU). For example, if the length of the
            dataset is 725 and we use 2 GPUs, we get len(self.sampler) = 363 (drop_last is false in distributed
            sampler as well.)

        Returns:
            int: _description_
        """
        new_ds_len = len(self.sampler) * self.comb_count
        
        if self.drop_last:
            return new_ds_len // self.batch_size  # type: ignore[arg-type]
        else:
            return (new_ds_len + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
    
    def __iter__(self) -> Iterator[List[int]]:
        """
            This implementation is mostly taken from the PyTorch source code and so far only the portion
            for drop_last=True has been implemented. Also it should be noted that the returned generator
            is for batch indexes for the entire epoch. For example if we have 56 steps in each epoch with
            the batch size of 64, the generator will create indices for this case.

        Yields:
            Iterator[List[int]]: The generator for batch indexes in each epoch
        """
        new_ds_iter = iter(list(iter(self.sampler)) * self.comb_count)
        while True:
            try:
                batch = [next(new_ds_iter) for _ in range(self.batch_size)]
                yield batch
            except StopIteration:
                break


class CustomTestSampler(DistributedSampler):
    """
        This sampler is used to divide the test data between the GPUs without data appending
        which is the default behavior of PyTorch. As a result, the test results are accurate
        even when done on multiple GPUs.   

    Args:
        DistributedSampler (_type_): _description_
    """
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None, 
                 shuffle: bool = False, seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        
        # self.num_samples = len(dataset)
        self.total_size = len(dataset)

    def __iter__(self) -> Iterator[T_co]:

        # No drop_last and no shuffling in test mode
        indices = list(range(len(self.dataset)))[self.rank::self.num_replicas]
        return iter(indices)
