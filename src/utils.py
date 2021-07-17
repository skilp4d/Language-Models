import random
import torch
import nltk
nltk.download('punkt')
from nltk import sent_tokenize


class DataCollatorForLanguageModeling:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.special_token_ids = []
        self.vocab_size = tokenizer.get_vocab_size()
        for i in range(len(tokenizer.get_vocab())):
            tok = tokenizer.id_to_token(i)
            if tok[0] == '[' and tok[-1] == ']':
                self.special_token_ids.append(i)
            else:
                break
        
    def __call__(self, batch):
        shape = (len(batch), len(batch[0]))
        labels = torch.full(shape, -100)
        mask = torch.full(shape, True)
#         input_ids
        for i in range(len(batch)):
            for j in range(len(batch[0])):
                tok = batch[i][j]
                if tok == 0:
                    break
                else:
                    r1 = random.random()
                    r2 = random.random()
                    if r1 < self.mlm_probability and tok not in self.special_token_ids:
                        if r2 < 0.8:
                            replacement = self.tokenizer.token_to_id('[MASK]')
                        elif r2 < 0.9:
                            replacement = random.randint(100, self.vocab_size-1)
                        else:
                            replacement = tok
                        
                        batch[i][j] = replacement
                        labels[i][j] = tok
                    else:
                        mask[i][j] = False
        
        return {
            'input_ids': torch.tensor(batch),
            'mask': mask,
            'labels': labels
        }
    



class LineByLineDataGenerator:
    def __init__(self, tokenizer, data_collator):
        self.tokenizer = tokenizer
        self.data_collator = data_collator
    
    def generate(self, jds, batch_size=32, max_length=128):
        examples = []
        ex = ''
        current_token_count = 0
        while True:
            for jd in jds:
                if len(ex) > 0:
                    ex += ' [SEP] '
                sents = sent_tokenize(jd)
                for sent in sents:
                    sent_len = len(sent.split())
                    if current_token_count + sent_len >= max_length:
                        examples.append(ex)
                        ex = ''
                        current_token_count = 0
                        if len(examples) == batch_size:
                            batch = self.get_mlm_batch(examples)
                            yield batch
                            examples = []

                    ex += sent
                    current_token_count += sent_len

    
    def get_mlm_batch(self, sentences):
        sentences = [f'[CLS] {s} [SEP]' for s in sentences]
        toks = self.tokenizer.encode_batch(
            sentences,
            add_special_tokens=True
        )
        toks = [t.ids for t in toks]

        return self.data_collator(toks)
    