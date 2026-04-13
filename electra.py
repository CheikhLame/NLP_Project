import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, tokenizer):
        # Filter out the last file if it's incomplete
        self.paths = paths[:len(paths)-1]
        self.tokenizer = tokenizer
        self.data = self.read_file(self.paths[0])
        self.current_file = 1
        self.remaining = len(self.data)
        self.encodings = self.get_encodings(self.data)

    def __len__(self):
        return 10000 * len(self.paths)
    
    def read_file(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            # BabyLM files are often one sentence/paragraph per line
            lines = f.read().split('\n')
        return [l for l in lines if len(l) > 0] # Remove empty lines

    def get_encodings(self, lines_all):
        # Tokenize using ELECTRA's tokenizer
        batch = self.tokenizer(lines_all, max_length=512, padding='max_length', truncation=True)

        labels = torch.tensor(batch['input_ids'])
        mask = torch.tensor(batch['attention_mask'])
        input_ids = labels.detach().clone()
        
        # ELECTRA theoretical note: 
        # If you are training the DISCRIMINATOR, you don't usually manually mask.
        # But if you are training the GENERATOR (MLM), we use the 15% rule:
        rand = torch.rand(input_ids.shape)
        
        # Masking logic: 0=PAD, 101=CLS, 102=SEP (standard for BERT/ELECTRA uncased)
        # Note: Check tokenizer.mask_token_id to be sure it's 103
        mask_token_id = self.tokenizer.mask_token_id 
        
        mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 101) * (input_ids != 102)
        
        input_ids[mask_arr] = 103
        
        return {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

    def __getitem__(self, i):
        if self.remaining == 0:
  
            self.data = self.read_file(self.paths[self.current_file])
            self.current_file += 1
            self.remaining = len(self.data)
            self.encodings = self.get_encodings(self.data)

        if self.current_file >= len(self.paths):
            self.current_file = 0
        
        self.remaining -= 1    
        # Use modulo to ensure we don't index out of bounds of the current file's encoding batch
        idx = i % len(self.encodings['input_ids'])
        return {key: tensor[idx] for key, tensor in self.encodings.items()}

def test_model(model, optim, test_ds_loader, device):
        """
        This function tests whether the parameters of the model that are frozen change, the ones that are not frozen do change,
        and whether any parameters become NaN or Inf
        :param model: model to be tested
        :param optim: optimiser used for training
        :param test_ds_loader: dataset to perform the forward pass on
        :param device: current device
        :raises Exception: if any of the above conditions are not met
        """
        ## Check if non-frozen parameters changed and frozen ones did not
    
        # get initial parameters to check against
        params = [ np for np in model.named_parameters() if np[1].requires_grad ]
        initial_params = [ (name, p.clone()) for (name, p) in params ]
    
        params_frozen = [ np for np in model.named_parameters() if not np[1].requires_grad ]
        initial_params_frozen = [ (name, p.clone()) for (name, p) in params_frozen ]
    
        optim.zero_grad()
    
        # get data
        batch = next(iter(test_ds_loader))
    
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
    
        # forward pass and backpropagation
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()
    
        # check if variables have changed
        for (_, p0), (name, p1) in zip(initial_params, params):
            # check different than initial
            try:
                assert not torch.equal(p0.to(device), p1.to(device))
            except AssertionError:
                raise Exception(
                "{var_name} {msg}".format(
                    var_name=name, 
                    msg='did not change!'
                    )
                )
            # check not NaN
            try:
                assert not torch.isnan(p1).byte().any()
            except AssertionError:
                raise Exception(
                "{var_name} {msg}".format(
                    var_name=name, 
                    msg='is NaN!'
                    )
                )
            # check finite
            try:
                assert torch.isfinite(p1).byte().all()
            except AssertionError:
                raise Exception(
                "{var_name} {msg}".format(
                    var_name=name, 
                    msg='is Inf!'
                    )
                )
            
        # check that frozen weights have not changed
        for (_, p0), (name, p1) in zip(initial_params_frozen, params_frozen):
            # should be the same
            try:
                assert torch.equal(p0.to(device), p1.to(device))
            except AssertionError:
                raise Exception(
                "{var_name} {msg}".format(
                    var_name=name, 
                    msg='changed!' 
                    )
                )
            # check not NaN
            try:
                assert not torch.isnan(p1).byte().any()
            except AssertionError:
                raise Exception(
                "{var_name} {msg}".format(
                    var_name=name, 
                    msg='is NaN!'
                    )
                )
                
            # check finite numbers
            try:
                assert torch.isfinite(p1).byte().all()
            except AssertionError:
                raise Exception(
                "{var_name} {msg}".format(
                    var_name=name, 
                    msg='is Inf!'
                    )
                )
        print("Passed")