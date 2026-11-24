import torch
import torch.nn.functional as F
import torch.utils.data as d

# Label value ignored by torch.nn.CrossEntropyLoss (and therefore by Hugging Face causal LM loss).
IGNORE_INDEX = -100

SYSTEM_PROMPT = 'You are a translator. Please translate English to Deutsch.'


def build_prompt(source: str) -> str:
    """
    Builds a Llama-3 chat-formatted prompt that ends right before the assistant's answer.

    Args:
        source (str): English source text.
    Returns:
        str: The prompt text (the tokenizer adds <|begin_of_text|> itself).
    """
    return (
        f'<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>'
        f'<|start_header_id|>user<|end_header_id|>\n\n{source}<|eot_id|>'
        f'<|start_header_id|>assistant<|end_header_id|>\n\n'
    )


class BaseDataset(d.Dataset):
    """
    Base class for datasets used in the training and evaluation of machine translation models.

    Attributes:
        data_x (list): List of source texts.
        data_y (list): List of target texts.
        tokenizer (Tokenizer): Tokenizer used to process the text data.
    """
    def __init__(self, data_pairs, tokenizer):
        """
        Args:
            data_pairs (list): List of dictionaries containing 'source' and 'label' keys.
            tokenizer (Tokenizer): Tokenizer used to process the text data.
        """
        super().__init__()
        pairs = [(data['source'], data['label']) for data in data_pairs if len(data['label']) != 0]
        self.data_x = [x for x, _ in pairs]
        self.data_y = [y for _, y in pairs]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_x)

    def _tokenize(self, index):
        """
        Tokenizes the prompt and the target separately.
        The target is tokenized without special tokens so that no extra <|begin_of_text|>
        is inserted between the prompt and the answer.
        """
        prompt_ids = self.tokenizer(
            text=build_prompt(self.data_x[index]), return_tensors='pt', truncation=True,
        )['input_ids'].squeeze(0)
        target_ids = self.tokenizer(
            text=f'{self.data_y[index]}<|eot_id|>', return_tensors='pt', truncation=True,
            add_special_tokens=False,
        )['input_ids'].squeeze(0)
        return prompt_ids, target_ids


class TrainDataset(BaseDataset):
    """
    Dataset for training: input_ids = prompt + target, labels = target only (prompt masked).
    """
    def __getitem__(self, index):
        """
        Returns:
            dict:
                'input_ids': Concatenated prompt and target tokens.
                'labels': Same as input_ids, but the prompt positions are set to IGNORE_INDEX
                          so that the loss is computed only on the target translation.
        """
        prompt_ids, target_ids = self._tokenize(index)

        input_ids = torch.cat([prompt_ids, target_ids])
        labels = input_ids.clone()
        labels[:len(prompt_ids)] = IGNORE_INDEX

        return {'input_ids': input_ids, 'labels': labels}


class EvaluateDataset(BaseDataset):
    """
    Dataset for evaluation: input_ids = prompt only, labels = reference translation tokens.
    """
    def __getitem__(self, index):
        """
        Returns:
            dict:
                'input_ids': Prompt tokens used for generation.
                'labels': Reference translation tokens (decoded and used as BLEU reference).
        """
        prompt_ids, target_ids = self._tokenize(index)

        return {'input_ids': prompt_ids, 'labels': target_ids}


class Collator():
    """
    Pads a batch and builds the attention mask.

    Attributes:
        tokenizer (Tokenizer): Tokenizer whose pad_token_id is used to pad input_ids.
        padding_side (str): 'right' for training, 'left' for generation (decoder-only models).
        label_pad_token_id (int): Value used to pad labels.
            IGNORE_INDEX for training (padding is excluded from the loss),
            tokenizer.pad_token_id for evaluation (labels are only decoded, and -100 cannot be decoded).
    """
    def __init__(self, tokenizer, padding_side='right', label_pad_token_id=IGNORE_INDEX):
        assert padding_side in ('left', 'right')
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.label_pad_token_id = label_pad_token_id

    def _pad(self, tokens, length, value):
        pad = (length - len(tokens), 0) if self.padding_side == 'left' else (0, length - len(tokens))
        return F.pad(tokens, pad, value=value)

    def __call__(self, batch):
        """
        Args:
            batch (list): List of dictionaries containing 'input_ids' and 'labels'.
        Returns:
            dict: 'input_ids', 'labels' and 'attention_mask' (1 for real tokens, 0 for padding).
        """
        xs = [b['input_ids'] for b in batch]
        ys = [b['labels'] for b in batch]

        max_x = max(len(x) for x in xs)
        max_y = max(len(y) for y in ys)

        input_ids = torch.stack([self._pad(x, max_x, self.tokenizer.pad_token_id) for x in xs])
        attention_mask = torch.stack([self._pad(torch.ones_like(x), max_x, 0) for x in xs])
        labels = torch.stack([self._pad(y, max_y, self.label_pad_token_id) for y in ys])

        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
