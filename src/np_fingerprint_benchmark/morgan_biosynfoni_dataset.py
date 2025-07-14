
import os
import random
from typing import List, Union
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from deepmol.datasets.datasets import SmilesDataset

import torch
from rdkit.Chem import Mol
from deepmol.tokenizers import SmilesTokenizer
from transformers import BertTokenizer

import re
from tqdm import tqdm
from deepmol.datasets.datasets import Dataset
from deepmol.tokenizers.tokenizer import Tokenizer
from transformers import BertTokenizer

from rdkit import Chem
from rdkit.Chem import AllChem
from biosynfoni import Biosynfoni
from deepmol.utils.utils import canonicalize_mol_object

from biosynfoni.subkeys import defaultVersion
from biosynfoni.rdkfnx import BiosynfoniVersion

from deepmol.loaders import CSVLoader

class MorganBiosinfonyTokenizer(Tokenizer):
    """Run regex tokenization"""

    def __init__(self, n_jobs=-1) -> None:
        """Constructs a RegexTokenizer.
        Args:
            regex_pattern: regex pattern used for tokenization.
            suffix: optional suffix for the tokens. Defaults to "".
        """
        super().__init__(n_jobs)
        self._vocabulary = None
        self._max_length = None

    def generate_sentence(self, smiles):# max_length = 2032
        
        tokens = BiosynfoniVersion(defaultVersion).subs_names
        try:
            mol = Chem.MolFromSmiles(smiles) # conver smiles to mol
            mol = canonicalize_mol_object(mol)
        except:
            return None
        if mol is None:
            return None
        numAtoms = mol.GetNumAtoms() # so luong atoms
        atom_info = {}
        info = {}
        _ = AllChem.GetMorganFingerprint(mol, 1, bitInfo=info, useChirality=True, useFeatures=True)
        for key, tup in info.items():
            for t in tup:
                atom_index = t[0]
                if atom_index not in atom_info:
                    atom_info[atom_index] = []
                # r = t[1]
                k = str(key)
                atom_info[atom_index].append(k)

        fp = Biosynfoni(mol)._detect_substructures()
        for list_of_list_of_atoms, token in zip(fp, tokens):
            
            for list_of_atoms in list_of_list_of_atoms:
                for atom in list_of_atoms:
                    if token not in atom_info[atom]:
                        atom_info[atom].append(token)

        sentence = []
        for atom in sorted(atom_info):
            sentence.extend(atom_info[atom]) 

        return sentence

    def _tokenize(self, text: str):
        """Regex tokenization.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens separated by spaces.
        """
        tokens = self.generate_sentence(text)
        return tokens
    
    @classmethod
    def from_file(cls, file_path: str):

        with open(file_path, mode="r") as f:
            lines = f.readlines()
            vocabulary = list(set([token.strip() for token in lines]))

        new_tokenizer = cls()
        new_tokenizer._vocabulary = vocabulary
        new_tokenizer._is_fitted = True
        return new_tokenizer
        
    
    def _fit(self, dataset: Dataset) -> 'MorganBiosinfonyTokenizer':
        """
        Fits the tokenizer to the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the tokenizer to.

        Returns
        -------
        self: AtomLevelSmilesTokenizer
            The fitted tokenizer.
        """
        self._is_fitted = True
        tokens = self.tokenize(dataset)
        self._vocabulary = list(set([token for sublist in tokens for token in sublist]))
        self._max_length = max([len(tokens) for tokens in tokens])
        return self
    
    @property
    def max_length(self) -> int:
        """
        Returns the maximum length of the SMILES strings.

        Returns
        -------
        max_length: int
            The maximum length of the SMILES strings.
        """
        return self._max_length
    
    @property
    def vocabulary(self) -> list:
        """
        Returns the vocabulary of the tokenizer.

        Returns
        -------
        vocabulary: list
            The vocabulary of the tokenizer.
        """
        return self._vocabulary

class BERTMorganBiosinfonyTokenizer(BertTokenizer):
    """
    Constructs a SmilesTokenizer.
    Adapted from https://github.com/huggingface/transformers
    and https://github.com/rxn4chemistry/rxnfp.

    Args:
        vocabulary_file: path to a token per line vocabulary file.
    """

    def __init__(
        self,
        vocab_file: str,
        do_lower_case=False,
        **kwargs,
    ) -> None:
        """Constructs an SmilesTokenizer.
        Args:
            vocabulary_file: vocabulary file containing tokens.
            unk_token: unknown token. Defaults to "[UNK]".
            sep_token: separator token. Defaults to "[SEP]".
            pad_token: pad token. Defaults to "[PAD]".
            cls_token: cls token. Defaults to "[CLS]".
            mask_token: mask token. Defaults to "[MASK]".
        """
        unk_token: str = "[UNK]"
        sep_token: str = "[SEP]"
        pad_token: str = "[PAD]"
        cls_token: str = "[CLS]"
        mask_token: str = "[MASK]"
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )
        # define tokenization utilities
        self.tokenizer = MorganBiosinfonyTokenizer.from_file(vocab_file)
        self.unique_tokens = [pad_token, unk_token, sep_token, cls_token, mask_token]

    @property
    def vocab_list(self):
        """List vocabulary tokens.
        Returns:
            a list of vocabulary tokens.
        """
        return list(self.vocab.keys())

    def _tokenize(self, text: str):
        """Tokenize a text representing a SMILES
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens.
        """
        return self.tokenizer._tokenize(text)

    @staticmethod
    def export_vocab(dataset, output_path, export_tokenizer=False):
        unique_tokens = set()

        tokenizer = MorganBiosinfonyTokenizer()
        tokenizer.fit(dataset=dataset)

        unique_tokens = tokenizer.vocabulary
        unk_token: str = "[UNK]"
        sep_token: str = "[SEP]"
        pad_token: str = "[PAD]"
        cls_token: str = "[CLS]"
        mask_token: str = "[MASK]"

        unique_tokens = [pad_token, unk_token, sep_token, mask_token, cls_token] + unique_tokens
        with open(output_path, "w") as f:
            for token in unique_tokens:
                f.write(token + "\n")

        if export_tokenizer:
            tokenizer.to_pickle("tokenizer.pkl")

    def get_max_size(self, smiles_list):
        
        max_size = 0
        for smile in tqdm(smiles_list, total=len(smiles_list)):
            tokens = self.tokenizer._tokenize(smile)
            if len(tokens) > max_size:
                max_size = len(tokens)
        
        return max_size
    
    def get_all_sizes(self, smiles_list):

        lengths = []
        for smile in tqdm(smiles_list, total=len(smiles_list)):
            tokens = self.tokenizer._tokenize(smile)
            lengths.append(len(tokens))

        return lengths



class MorganBiosinfonyDataset(TorchDataset, SmilesDataset):

    def __init__(self,
                 smiles: Union[np.ndarray, List[str]],
                 mols: Union[np.ndarray, List[Mol]] = None,
                 ids: Union[List, np.ndarray] = None,
                 X: Union[List, np.ndarray] = None,
                 feature_names: Union[List, np.ndarray] = None,
                 y: Union[List, np.ndarray] = None,
                 label_names: Union[List, np.ndarray] = None,
                 mode: Union[str, List[str]] = 'auto',
                 max_length: int = 256, 
                 masking_probability: float = 0.15,
                 vocabulary_path: str = None,
                 mask: bool = True):
        """

        Parameters
        ----------
        tokenizer : BertTokenizer
            BertTokenizer
        smiles : Union[np.ndarray, List[str]]
            SMILES
        mols : Union[np.ndarray, List[Mol]], optional
            mols, by default None
        ids : Union[List, np.ndarray], optional
            Identifiers, by default None
        X : Union[List, np.ndarray], optional
            features, by default None
        feature_names : Union[List, np.ndarray], optional
            feature names, by default None
        y : Union[List, np.ndarray], optional
            label values, by default None
        label_names : Union[List, np.ndarray], optional
            label names, by default None
        mode : Union[str, List[str]], optional
            mode, by default 'auto'
        max_length : int, optional
            max length, by default 256
        masking_probability : float, optional
            Probability of masking item, by default 0.15
        mask : bool, optional
            Whether the dataset will be used for masked learning. If False, it assumes it will be used for fine-tuning
        """

        super().__init__(smiles, mols, ids, X, feature_names, y, label_names, mode)
        
        if vocabulary_path is None:
            dir_path = os.getcwd()
            BERTMorganBiosinfonyTokenizer.export_vocab(self, os.path.join(dir_path, "vocab.txt"))
            self.tokenizer = BERTMorganBiosinfonyTokenizer(os.path.join(dir_path, "vocab.txt"))
        else:
            self.tokenizer = BERTMorganBiosinfonyTokenizer(vocabulary_path)
            
        self.masking_probability = masking_probability
        self.max_length = max_length
        self.mask = mask

    def __len__(self):
        
        return len(self._smiles)

    def __getitem__(self, idx):
        smiles = self._smiles[idx]
        tokens = self.tokenizer(smiles, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokens.input_ids.squeeze()
        attention_mask = tokens.attention_mask.squeeze()
        mask_indices = None

        if self.mask:
            torch.manual_seed(42)
            random.seed(42)
            labels = input_ids.clone()
            
            # Identify special tokens
            special_tokens = {self.tokenizer.pad_token_id, self.tokenizer.unk_token_id, 
                            self.tokenizer.sep_token_id, self.tokenizer.mask_token_id, 
                            self.tokenizer.cls_token_id}
            
            # Create masking probability matrix
            probability_matrix = torch.full(labels.shape, self.masking_probability)
            mask_indices = torch.bernoulli(probability_matrix).bool()
            
            # Avoid masking special tokens
            for token_id in special_tokens:
                mask_indices &= input_ids != token_id
            
            # Ensure masked tokens are valid (select other tokens if special)
            for idx in torch.where(mask_indices)[0]:
                if input_ids[idx].item() in special_tokens:
                    available_tokens = [t for t in range(self.tokenizer.vocab_size) if t not in special_tokens]
                    input_ids[idx] = torch.tensor(random.choice(available_tokens), dtype=torch.long)
            
            input_ids[mask_indices] = self.tokenizer.mask_token_id
            return input_ids, attention_mask, labels, mask_indices
        
        else:
            if len(self.y.shape) == 1:
                labels = torch.tensor(self.y[idx], dtype=torch.long)
            else:
                labels = torch.tensor(self.y[idx, :], dtype=torch.float)

            return input_ids, attention_mask, labels

class CSVLoaderMorganBiosinfony(CSVLoader):

    def __init__(self,
                 dataset_path: str,
                 smiles_field: str,
                 id_field: str = None,
                 labels_fields: List[str] = None,
                 features_fields: List[str] = None,
                 shard_size: int = None,
                 mode: Union[str, List[str]] = 'auto',
                 vocabulary_path: str = None, 
                 masking_probability: str = 0.15, 
                 max_length=256) -> None:
        """
        Initialize the CSVLoader.

        Parameters
        ----------
        dataset_path: str
            path to the dataset file
        smiles_field: str
            field containing the molecules'
        id_field: str
            field containing the ids
        labels_fields: List[str]
            field containing the labels
        features_fields: List[str]
            field containing the features
        shard_size: int
            size of the shard to load
        mode: Union[str, List[str]]
            The mode of the dataset.
            If 'auto', the mode is inferred from the labels. If 'classification', the dataset is treated as a
            classification dataset. If 'regression', the dataset is treated as a regression dataset. If a list of
            modes is passed, the dataset is treated as a multi-task dataset.
        """
        self.dataset_path = dataset_path
        self.mols_field = smiles_field
        self.id_field = id_field
        self.labels_fields = labels_fields
        self.features_fields = features_fields
        self.shard_size = shard_size
        fields2keep = [smiles_field]
        self.max_length = max_length

        if id_field is not None:
            fields2keep.append(id_field)

        if labels_fields is not None:
            fields2keep.extend(labels_fields)

        if features_fields is not None:
            fields2keep.extend(features_fields)

        self.fields2keep = fields2keep
        self.mode = mode
        self.vocabulary_path = vocabulary_path
        self.masking_probability = masking_probability

    def create_dataset(self, **kwargs) -> SmilesDataset:
        """
        Creates a dataset from the CSV file.

        Parameters
        ----------
        kwargs:
            Keyword arguments to pass to pandas.read_csv.

        Returns
        -------
        SmilesDataset
            Dataset with the data.
        """
        dataset = self._get_dataset(self.dataset_path, fields=self.fields2keep, chunk_size=self.shard_size, **kwargs)

        mols = dataset[self.mols_field].to_numpy()

        if self.features_fields is not None:
            if len(self.features_fields) == 1:
                X = dataset[self.features_fields[0]].to_numpy()
            else:
                X = dataset[self.features_fields].to_numpy()
        else:
            X = None

        if self.labels_fields is not None:
            if len(self.labels_fields) == 1:
                y = dataset[self.labels_fields[0]].to_numpy()
            else:
                y = dataset[self.labels_fields].to_numpy()
        else:
            y = None

        if self.id_field is not None:
            ids = dataset[self.id_field].to_numpy()
        else:
            ids = None

        if self.masking_probability == None or self.masking_probability == 0:
            mask = False
        else:
            mask = True

        return MorganBiosinfonyDataset(smiles=mols,
                             X=X,
                             y=y,
                             ids=ids,
                             feature_names=self.features_fields,
                             label_names=self.labels_fields,
                             mode=self.mode, 
                             vocabulary_path = self.vocabulary_path,
                             masking_probability = self.masking_probability,
                             mask = mask, 
                             max_length=self.max_length)
