import json
from copy import deepcopy
from datetime import datetime
from math import ceil, log2
from os.path import join
from typing import Dict, List, Tuple, Type, TypeVar

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from utils.common import print_banner

# TypeVar of CICIDS2017_PerPacketDataset class
T_CICIDS2017_PerPacket = TypeVar("T_CICIDS2017_PerPacket", bound="CICIDS2017_PerPacket")


class CICIDS2017_PerPacket(torch.utils.data.Dataset):
    """
    Dataset class to load CICIDS2017 per-packet level dataset.

    Chooses from a subset of features and returns train/test sets.
    """

    def __init__(
        self,
        file_path: str,
        test_file_path: str = "",
        is_split: bool = False,
    ):
        """
        Class constructor.

        Args:

            file_path (str): Path to CSV file of dataset. This can be path to
            the train set CSV file if the dataset has already been pre-split
            into train/test sets. In that case, an optional test_file_path
            argument must also be passed.

            test_file_path (str): If is_split is set to True, then that
            signifies that the dataset has already been split into train/test
            sets. This argument must then be supplied to specify the path to
            the test file CSV. Default = ""

            is_split (bool): Boolean flag to specify if the dataset being
            loaded in is already split into train/test sets. Default = False.

        Returns: None
        """

        print("Loading CIC-IDS2017 per-packet-level dataset")

        # initialize defaults
        self._init_bitwidth_dict()
        self.seed = -1

        # store args
        self.file_path = file_path
        self.test_file_path = test_file_path
        self.is_split = is_split

        if is_split:
            self.train_df = pd.read_csv(file_path).reset_index().drop(columns=["index"])
            self.test_df = pd.read_csv(test_file_path).reset_index().drop(columns=["index"])

            print(f"Loaded training set of length = {len(self.train_df)}")
            print(f"Training set statistics: {self._get_stats_pd(self.train_df)}")
            print(f"Loaded test set of length = {len(self.test_df)}")
            print(f"Test set statistics: {self._get_stats_pd(self.test_df)}")

        else:
            self.df = pd.read_csv(file_path).reset_index().drop(columns=["index"])

            print(f"Loaded dataset of length = {len(self.df)}")
            print(f"Dataset statistics: {self._get_stats_pd(self.df)}")

    def set_seed(self, seed: int):
        """
        Sets a seed for reproducibility purposes.

        Args:

            seed (int): Integer seed value.

        Returns: None
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def split_into_train_and_test(self, train_split: float = 0.8):
        """
        Splits dataset into train and test sets. Note that at this point the
        train/test sets are stored in memory, and not saved to disk. So any
        split that is created here amy be lost in the next run. Create one-off
        splits statically and store to disk (see save_split_df_to_disk() below)
        for better reproducibility.

        Args:

            train_split (float): Specifies the percentage of dataset that will
            be split into a train set. The split is stratified to the labels
            column in the Pandas dataframe. Default = 0.8.

        Returns: None
        """
        self.train_df, self.test_df = train_test_split(
            self.df, test_size=1.0 - train_split, stratify=self.df[["label"]]
        )
        self.is_split = True
        self.train_split = train_split

    @classmethod
    def load_from_split(
        cls: Type[T_CICIDS2017_PerPacket],
        path_to_train_csv: str,
        path_to_test_csv: str,
    ) -> T_CICIDS2017_PerPacket:
        """
        Classmethod to load dataset from an already split dataset files saved
        on disk.

        Args:

            cls (T_CICIDS2017_PerPacket): This class.

            path_to_train_csv (str): Path to CSV of train set.

            path_to_test_csv (str): Path to CSV of test set.

        Returns:

            T_CICIDS2017_PerPacket: Instance of this class.
        """
        return cls(file_path=path_to_train_csv, test_file_path=path_to_test_csv, is_split=True)

    def sample(self, n: int = -1, seed: int = -1):
        """
        Print a few samples from the dataset to log (which could be stdout).

        Args:

            n (int): Number of samples to show, default = -1.

            seed (int): Set runtime seed for reproducibility. If set to -1,
            then no seed is set. Default = -1.

        Returns: None
        """
        if self.is_split:
            self.sample_from_splits(n, seed)
            return

        if n == -1:
            num_samples = len(self.df)

        print(print_banner(heading="Dataset Samples", print_len=40, return_str=True))
        if seed != -1:
            print(f"{self.df.sample(num_samples, random_state=seed)}")
        else:
            print(f"{self.df.sample(num_samples)}")

    def sample_from_splits(self, n: int = -1, seed: int = -1):
        """
        Print a few samples from the split train/test datasets to log (which
        could be stdout).

        Args:

            n (int): Number of samples to show, default = -1.

            seed (int): Set runtime seed for reproducibility. If set to -1,
            then no seed is set. Default = -1.

        Returns: None
        """
        if n == -1:
            n_train = len(self.train_df)
            n_test = len(self.test_df)
        else:
            n_train = n
            n_test = n

        print(print_banner(heading="Train Set Samples", print_len=40, return_str=True))
        if seed != -1:
            print(f"{self.train_df.sample(n_train, random_state=seed)}")
        else:
            print(f"{self.train_df.sample(n_train)}")

        print(print_banner(heading="Test Set Samples", print_len=40, return_str=True))
        if seed != -1:
            print(f"{self.test_df.sample(n_test, random_state=seed)}")
        else:
            print(f"{self.test_df.sample(n_test)}")

    def save_split_df_to_disk(
        self,
        out_name_prefix: str,
        out_dir_path: str,
    ):
        """
        Saves train/test split dataset to disk. This should be done as a
        one-time step that allows us to improve reproducibility and runtime.
        The outputs are stored in a serialized binary format; two output
        dataset files are created: out_dir_path/out_name_prefix.train and
        out_dir_path/out_name_prefix.test. These dataset files contain both
        samples and labels, and relevant dataset loaders are provided as
        classmethods in this class. All feature columns are preserved from the
        original dataset into the newer split datasets. A third output file is
        also created to store metadata about the split datasets, which contains
        information on when/how the split dataset was created. This is useful
        for data provenance and reporudcibility purposes. The metadata file is
        a text file stored as out_dir_path/out_name_prefix.metadata.

        Args:

            in_file_path (str): Path to input entire dataset file.

            out_name_prefix (str): Name to give output files.

            out_dir_path (str): Path to directory where output files will be
            written to.

        Returns: None
        """
        ts = datetime.now()
        hr_ts = ts.strftime("%d-%m-%Y, %H:%M:%S")

        train_file = f"{join(out_dir_path, out_name_prefix)}.train.csv"
        test_file = f"{join(out_dir_path, out_name_prefix)}.test.csv"
        metadata_file = f"{join(out_dir_path, out_name_prefix)}.metadata"

        self.train_df.to_csv(train_file, index=False)
        self.test_df.to_csv(test_file, index=False)

        # create metadata
        metadata = f"Dataset created on: {hr_ts} ({ts.timestamp()})\n"
        metadata += "Dataset split created with the following command:\n"
        metadata += "\tpoetry run data dataset-split \\\n"
        metadata += "\t\t--name cicids2017 \\\n"
        metadata += f"\t\t--fpath {self.file_path} \\\n"
        metadata += f"\t\t--out-name {out_name_prefix} \\\n"
        metadata += f"\t\t--out-dir {out_dir_path} \\\n"
        if self.seed != -1:
            metadata += f"\t\t--seed {self.seed} \\\n"
        metadata += f"\t\t--train-split {self.train_split}\n\n"
        metadata += f"train_stats = {self._get_stats_pd(self.train_df)}\n"
        metadata += f"test_stats = {self._get_stats_pd(self.test_df)}\n"

        # write metadata to file
        with open(metadata_file, "w") as fp:
            fp.write(metadata)

        print(f"Saved split train set to {train_file}")
        print(f"Saved split test set to {test_file}")
        print(f"Saved train/test set metadata to {metadata_file}")

    def binarize_features_and_labels(
        self, df: pd.DataFrame, columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Extracts relevant features and labels from a dataframe, and return them
        as numpy arrays.

        Args:

            df (pd.DataFrame): Pandas dataframe to evaluate.

            columns (List[str]): Name of columns (i.e. features) to be extracted.

        Returns:

            np.ndarray: Numpy array of features.

            np.ndarray: Numpy array of labels.

            Dict: Metadata of created dataset, primarily for RTL building
            purposes
        """

        def char_split(s):
            """
            Splits a string into list of characters.
            """
            return np.array([ch for ch in s])

        columns_to_extract = deepcopy(columns)
        if "label" not in columns_to_extract:
            columns_to_extract.append("label")

        columns_to_drop = [col for col in df.columns if col not in columns_to_extract]

        # convert decimal to binary
        quantized_df = df.copy().drop(columns=columns_to_drop)
        for column in quantized_df.columns:
            column_data = df[column]
            quantized_df[column] = (
                self._dec2bin(column_data, self.bitwidths[column], left_msb=False)
                .reshape((-1, 1))
                .flatten()
            )

        for column in quantized_df.columns:
            if column == "label":
                continue
            quantized_df[column] = quantized_df[column].apply(char_split).values

        # get labels from dataframe
        all_labels = quantized_df["label"].values.tolist()
        all_labels = np.array([int(a) for a in all_labels]).astype(np.float32)

        # drop label column from dataframe
        all_features = quantized_df.drop(columns=["label"])

        # rearrange columns to order specified in original list
        all_features = all_features[columns]

        # collect metadata about dataset that has been created (primarily for
        # RTL purposes)
        metadata = {"total_in_bitwidth": 0, "ordering": []}
        for column in all_features.columns:
            metadata["total_in_bitwidth"] += self.bitwidths[column]  # type: ignore
            metadata["ordering"].append((column, self.bitwidths[column]))  # type: ignore

        all_features = (
            pd.DataFrame(np.column_stack(all_features.values.T.tolist()))
            .to_numpy()
            .astype(np.float32)
        )

        # get num output neurons required
        out_bitwidth = int(ceil(log2(len(set(np.reshape(all_labels, -1).tolist())))))
        metadata["total_out_bitwidth"] = out_bitwidth

        return all_features, all_labels, metadata

    def get_binarized_train_test_sets(
        self,
        columns: List[str],
    ) -> Tuple[TensorDataset, TensorDataset, Dict]:
        """
        Return train and test datasets that have been quantized and binarized
        for training purposes. Only return subset of columns that are required
        for training; i.e. not all features may be needed for prediction.

        Args:

            columns (List[str]): Name of columns (i.e. features) to be extracted.

        Returns:

            TensorDataset: Train dataset.

            TensorDataset: Test dataset.

            Dict: Metadata that describes the dataset created, ordering of the
            features, etc.
        """
        print("Binarizing train and test sets...")
        train_set, metadata = self.make_binarized_set(self.train_df, columns)
        test_set, _ = self.make_binarized_set(self.test_df, columns)

        print(f"Dataset metadata: {json.dumps(metadata, indent=4)}")
        return train_set, test_set, metadata

    def get_binarized_dataset(
        self,
        columns: List[str],
    ) -> Tuple[TensorDataset, Dict]:
        """
        Return dataset that has been quantized and binarized for training
        purposes. Only return subset of columns that are required for training;
        i.e. not all features may be needed for prediction.

        Args:

            columns (List[str]): Name of columns (i.e. features) to be extracted.

        Returns:

            TensorDataset: Binarized dataset.

            Dict: Metadata that describes the dataset created, ordering of the
            features, etc.
        """
        dataset, metadata = self.make_binarized_set(self.df, columns)
        print(f"Dataset metadata: {json.dumps(metadata, indent=4)}")

        return dataset, metadata

    def make_binarized_set(
        self,
        df: pd.DataFrame,
        columns: List[str],
    ) -> Tuple[TensorDataset, Dict]:
        """
        Converts a pandas dataframe dataset into its binarized version, and
        returns it as a TensorDataset record. Metadata on how the binarized
        dataset was created is also captured and returned as a python
        dictionary.

        Args:

            columns (List[str]): Name of columns (i.e. features) to be extracted.

        Returns:

            TensorDataset: Tensor dataset in binarized format.

            Dict: Metadata that describes the dataset created, ordering of the
            features, etc.
        """
        x, y, metadata = self.binarize_features_and_labels(df, columns)
        dset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

        return dset, metadata

    def _get_stats_pd(self, df: pd.DataFrame) -> str:
        """
        Computes label distribution stats of a set of training/test samples
        from a Pandas DataFrame.

        Args:

            df (pd.DataFrame): Pandas DataFrame instance of data samples.

        Returns:

            str: Computed statistics in pretty-print string format.
        """
        num_true = len(df[df["label"] == 1])
        return f"{num_true}/{len(df)} ({num_true*100.0/len(df):.2f}% TRUE labels)"

    def _get_stats(self, ds: TensorDataset) -> str:
        """
        Computes label distribution stats of a set of training/test samples.

        Args:

            ds (TensorDataset): TensorDataset instance of data samples.

        Returns:

            str: Computed statistics in pretty-print string format.
        """
        num_true = len([y for x, y in ds if y == 1])  # type: ignore
        return f"{num_true}/{len(ds)} ({num_true*100.0/len(ds):.2f}% TRUE labels)"

    def _dec2bin(self, column: pd.Series, number_of_bits: int, left_msb: bool = True) -> pd.Series:
        """
        Convert a decimal pd.Series to binary pd.Series with numbers in their
        base-2 equivalents. The output is a numpy nd array.

        Adapted from: https://stackoverflow.com/q/51471097/1520469

        Args:

            column (pd.Series): Series with all decimal numbers that will be
            cast to binary

            number_of_bits (int): The desired number of bits for the binary
            number. If bigger than what is needed, then those bits will be 0.
            The number_of_bits should be >= than what is needed to express the
            largest decimal input

            left_msb (bool): Specify that the most significant digit is the
            leftmost element. If this is False, it will be the rightmost
            element. (Default = True)

        Returns:

            numpy.ndarray: Numpy array with all elements in binary
            representation of the input.
        """

        def my_binary_repr(number, nbits):
            return np.binary_repr(number, nbits)[::-1]

        func = my_binary_repr if left_msb else np.binary_repr

        return np.vectorize(func)(column.values, number_of_bits)  # type: ignore

    def _init_bitwidth_dict(self):
        """
        Initialises bitwidth dictionary which contains bitwidth of all features
        in the dataset. Note that this is currently manually keyed in.

        Args: None, Returns: None
        """
        # Note that "flow_id" is not a feature, it is unlikely to be used for
        # training, but is kept in the split datasets for diagnostics purposes.
        # It is given a default bitwidth of 32b.
        self.bitwidths = {
            "flow_id": 32,
            "total_pkts": 16,  # 16b accumulation => 65k packets max per flow
            "total_bytes": 32,
            "duration_usec": 64,  # 64b timestamp field
            "total_ttl": 16,  # 16b accumulation, TTL is an 8b field
            "max_ttl": 8,
            "min_ttl": 8,
            "total_iat_usec": 64,  # 64b timestamp
            "max_iat_usec": 64,
            "min_iat_usec": 64,
            "iat_usec": 64,
            "total_syn": 16,
            "total_ack": 16,
            "total_fin": 16,
            "total_rst": 16,
            "total_psh": 16,
            "total_urg": 16,
            "total_cwr": 16,
            "total_ece": 16,
            "init_window_bytes": 16,
            "num_active_flows": 16,  # might need to change?
            "label": 1,
        }
