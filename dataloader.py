from torch.utils.data.dataset import Dataset,TensorDataset
from torch.utils.data.dataloader import DataLoader
from torch import LongTensor
import torch

MASK_TOKEN = '[MASK]'
PAD_TOKEN = '[PAD]'
class ProductTokenizer:
    # Tokenization happens prior to dataset creation
    # Takes unique tokens and creates a token dictionary for the object 
    def __init__(self,unique_tokens) -> None:
        self.unique_products = unique_tokens
        special_tokens=[MASK_TOKEN,PAD_TOKEN]
        self.unique_products.extend(special_tokens)
        self.token_dict = {product:idx+1 for idx,product in enumerate(self.unique_products)}
    
    def __len__(self):
        return len(self.token_dict)

    @property
    def pad_token_id(self):
        return self.token_dict[PAD_TOKEN]

    @property
    def mask_token(self):
        return MASK_TOKEN
    
    def convert_tokens_to_ids(self,token):
        # todo :optimize
        return self.encode([token])[0]

    def encode_batch(self,product_lists,max_len = 50):
        encoded_products = []
        masks = []
        for product_list in product_lists:
            tensor = self.encode(product_list,to_tensor=True)
            padded_tensor, mask = self.create_padding_and_mask(tensor,max_len)
            masks.append(mask)
            encoded_products.append(padded_tensor)
        
        return TensorDataset(torch.stack(encoded_products),torch.stack(masks))
    
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
        tokens = []
        for product in product_list:
            if product not in self.token_dict:
                raise Exception("Unknow Token")
            tokens.append(self.token_dict[product])
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
        return inputs.long(), labels.long()

    def __call__(self,batch):
        products = []
        mask = []
        masked_tokens = []
        for b in batch:
            product_tensor,mask_tensor = b
            masked_tokens.append(product_tensor)
            mask.append(mask_tensor)

        products, labels = self.mask_tokens(torch.stack(masked_tokens),self.tokenizer,mlm_probability=self.mlm_probability)
        return (
            products,
            torch.stack(mask),
            labels
        )

class ProductDataset(Dataset):
    def __init__(self,session_list,max_len=50,tokenizer:ProductTokenizer=None) -> None:
        super().__init__()
        assert tokenizer is not None
        self._tokenizer = tokenizer
        self._session_list = session_list
        self._tokenized_dataset = self._tokenizer.encode_batch(self._session_list,max_len=max_len)
    
    @property
    def tokenizer(self):
        return self._tokenizer

    def __len__(self):
        return len(self._tokenized_dataset)
    
    def __getitem__(self, index):
        return self._tokenized_dataset[index]

    def collate_fn(self):
        return ProductMaskCollateFn(self._tokenizer)



def get_dataloader(sessions,batch_size=40,max_seq_len=50,tokenizer=None):
    dataset = ProductDataset(
        sessions,max_len=max_seq_len,tokenizer=tokenizer
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn()
    )