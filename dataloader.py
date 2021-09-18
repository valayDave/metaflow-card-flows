from torch.utils.data.dataset import Dataset,TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch import LongTensor
import torch


class ProductTokenizer:
    def __init__(self,session_lists) -> None:
        self.unique_products = list(set([prod for prod_list in session_lists for prod in prod_list]))
        special_tokens=['[MASK]','[PAD]']
        self.unique_products.extend(special_tokens)
        self.token_dict = {product:idx+1 for idx,product in enumerate(self.unique_products)}
    
    def __len__(self):
        return len(self.token_dict)

    @property
    def pad_token_id(self):
        return self.token_dict['[PAD]']

    @property
    def mask_token(self):
        return self.token_dict['[MASK]']
    
    def convert_tokens_to_ids(self,token):
        # todo :optimize
        return self.decode([token])[0]

    def encode_batch(self,product_lists,max_len = 50):
        encoded_products = []
        masks = []
        for product_list in product_lists:
            tensor = self.encode(product_list,to_tensor=True)
            padded_tensor, mask = self.create_padding_and_mask(tensor,max_len)
            encoded_products.append(padded_tensor)
        
        return TensorDataset(torch.stack(encoded_products),torch.stack(mask))
    
    @staticmethod
    def create_padding_and_mask(tensr, max_len, pad_value=0):
        # tensr : expected 1d tensor shape 
        seq_len = tensr.shape[0]
        padding_len = max_len - seq_len
        if padding_len > 0:
            padded_ten = torch.cat(
                [tensr, torch.empty(padding_len).fill_(pad_value)])
            mask = torch.cat(
                [torch.empty(seq_len).fill_(1), torch.zeros(padding_len)])
            return padded_ten, mask
        elif padding_len == 0:
            return tensr, torch.empty(max_len).fill_(1)
        else: # This will truncate the tensor if the length is larger than the max_len
            return tensr[:padding_len], torch.empty(max_len).fill_(1)

    def encode(self,product_list,to_tensor=False):
        tokens = [self.token_dict[product] for product in product_list]
        if to_tensor:
            return LongTensor(tokens)
        return tokens

    def decode(self,token_id_list):
        return [self.unique_products[token_id] for token_id in token_id_list]


# Collated Function For Mask Language Modeling. 
class ProductMaskCollateFn:

    def __init__(self,\
            tokenizer:ProductTokenizer,
            mlm_probability = 0.15):
        self.tokenizer = tokenizer
        self.mlm_probability =mlm_probability

    @staticmethod
    def mask_tokens(inputs, tokenizer:ProductTokenizer, mlm_probability=0.15):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 20% original. """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [list(map(lambda x: 1 if x == tokenizer.pad_token_id else 0, val)) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # The rest of the time (20% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def __call__(self,batch):
        products = []
        mask = []
        labels= []
        for b in batch:
            product_tensor,mask_tensor = b
            masked_tokens, label_tensor = self.mask_tokens(product_tensor,self.tokenizer,mlm_probability=self.mlm_probability)
            products.append(masked_tokens)
            mask.append(mask_tensor)
            labels.append(label_tensor)
        return (
            torch.stack(products),
            torch.stack(mask),
            torch.stack(labels)
        )
# Training Pipeline : 
class ProductDataset(Dataset):
    def __init__(self,session_list,max_len=50) -> None:
        super().__init__()
        self._tokenizer = ProductTokenizer(session_list)
        self._session_list = session_list
        self._tokenized_dataset = self._tokenizer.encode_batch(self._session_list,max_len=max_len)

    def __len__(self):
        return len(self._tokenized_dataset)
    
    def __getitem__(self, index):
        return self._tokenized_dataset[index]

    def collate_fn(self):
        return ProductMaskCollateFn(self._tokenizer)


def get_dataloader(sessions,batch_size=40,max_seq_len=50):
    dataset = ProductDataset(
        sessions,max_len=max_seq_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn()
    )