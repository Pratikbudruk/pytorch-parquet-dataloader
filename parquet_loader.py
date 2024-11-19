import os
import torch.multiprocessing as mp
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset
import argparse

from queue import Empty

mp.set_start_method('spawn', force=True)


class ParquetIterableDataset(IterableDataset):
    def __init__(self, parquet_file, columns=None):
        """
        Args:
            parquet_file (str): Path to the Parquet file.
            columns (list): Columns to extract from the Parquet file.
        """
        self.parquet_file = parquet_file
        self.columns = columns

    # def __iter__(self):
    #     worker_info = torch.utils.data.get_worker_info()
    #
    #     if worker_info is None:
    #         workers = 1
    #         worker_id = 0
    #     else:
    #         workers = worker_info.num_workers
    #         worker_id = worker_info.id
    #
    #     parquet_reader = pq.ParquetFile(self.parquet_file)
    #     num_row_groups = parquet_reader.num_row_groups
    #
    #     # Assign row groups to workers in a round-robin fashion
    #     row_groups = list(range(num_row_groups))
    #     worker_row_groups = row_groups[worker_id::workers]
    #
    #     rows_processed = 0

    #     #TODO Does not handle out-of-core batching if Parquet files exceed memory per worker
    #
    #     for row_group_idx in worker_row_groups:
    #         batch = parquet_reader.read_row_group(row_group_idx).to_pandas()
    #
    #         # Validate and filter columns
    #         if self.columns:
    #             invalid_columns = [col for col in self.columns if col not in batch.columns]
    #             if invalid_columns:
    #                 raise ValueError(f"Invalid columns: {', '.join(invalid_columns)} not found in the Parquet file.")
    #             batch = batch[self.columns]
    #
    #         batch = batch.dropna()
    #
    #
    #         for _, row in batch.iterrows():
    #             yield row.to_dict()
    #             rows_processed += 1
    #
    #     # Final message when all row groups are processed
    #     yield f"Worker {worker_id} finished processing its assigned row groups."

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            workers = 1
            worker_id = 0
        else:
            workers = worker_info.num_workers
            worker_id = worker_info.id

        # Open the Parquet file as a PyArrow Dataset
        dataset = pq.ParquetDataset(self.parquet_file)
        fragments = list(dataset.fragments)
        assigned_fragments = fragments[worker_id::workers]

        for fragment in assigned_fragments:
            # Stream rows from the fragment
            for batch in fragment.to_table(columns=self.columns).to_batches():
                batch_df = batch.to_pandas()
                batch_df = batch_df.dropna()

                # Yield rows one by one
                for _, row in batch_df.iterrows():
                    # TODO Tokenization (if you are working with text data)
                    # TODO Apply data filtration logic
                    yield row.to_dict()


def collate_fn(batch):
    """
    Collate function to combine individual row dictionaries into a batch of lists.
    """
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [item[key] for item in batch]
    return batch_dict


def print_worker_output(queue):
    """Function to safely print worker output from a queue."""
    while True:
        try:
            msg = queue.get(timeout=10)  # Wait for a message, timeout after 10 seconds
            print(msg)
        except Empty:
            break  # Exit when no more messages are available


def create_mock_parquet(file_path, num_rows=1000, num_columns=5):
    """
    Create a mock Parquet file for testing.
    """
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        f"col_{i}": np.random.rand(num_rows) for i in range(num_columns)
    })
    df.to_parquet(file_path, engine='pyarrow')
    print(f"Mock Parquet file created at: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Process a Parquet file using PyTorch DataLoader.")
    parser.add_argument('parquet_file', type=str, nargs='?', help="Path to the Parquet file.")
    parser.add_argument('--columns', type=str, nargs='*', help="Columns to extract from the Parquet file.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for DataLoader.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument('--mock', action='store_true', help="Create and test with a mock Parquet file.")
    args = parser.parse_args()

    if args.mock:
        # Create a mock Parquet file for testing
        mock_file = "mock.parquet"
        create_mock_parquet(mock_file)
        args.parquet_file = mock_file

    if not os.path.exists(args.parquet_file) and not args.mock:
        raise FileNotFoundError(f"Parquet file not found: {args.parquet_file}")

    # Create dataset and dataloader
    dataset = ParquetIterableDataset(args.parquet_file, columns=args.columns)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    # Iterate through the DataLoader and print batches
    # queue = mp.Queue()

    # print_process = mp.Process(target=print_worker_output, args=(queue,))
    # print_process.start(

    try:
        for i, batch in enumerate(dataloader):
            print(f"Batch {i + 1}: {batch}")
            if i >= 5:  # Stop after 5 batches for demonstration
                break
    except ValueError as e:
        print(f"Error occurred: {e}")

    # print_process.join()


if __name__ == "__main__":
    main()
