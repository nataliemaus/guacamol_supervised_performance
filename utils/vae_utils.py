from selfies_vae.model_positional_unbounded import SELFIESDataset, InfoTransformerVAE
import torch 
from selfies_vae.data import collate_fn

def vae_seqs_to_zs(vae, dataobj, selfies_list, vae_latent_dim=256):
    ''' Input: 
            a list selfies strings 
        Output: 
            z: tensor of resultant latent space codes 
                obtained by passing the xs through the encoder
    '''
    # assumes xs_batch is a batch of smiles strings 
    X_list = []
    # TODO: batch this ? 
    for selfie_str in selfies_list:
        tokenized_selfie = dataobj.tokenize_selfies([selfie_str])[0]
        encoded_selfie = dataobj.encode(tokenized_selfie).unsqueeze(0)
        X_list.append(encoded_selfie)
    X = collate_fn(X_list)
    dict1 = vae(X.cuda())
    z = dict1['z']
    z = z.reshape(-1,vae_latent_dim) # self.dim 

    return z

def initialize_vae(path_to_vae_statedict="selfies_vae/SELFIES-VAE-state-dict.pt"):
    ''' Sets self.vae to the desired pretrained vae and 
        sets self.dataobj to the corresponding data class 
        used to tokenize inputs, etc. '''
    dataobj = SELFIESDataset()
    vae = InfoTransformerVAE(dataset=dataobj)
    # load in state dict of trained model:
    state_dict = torch.load(path_to_vae_statedict) 
    vae.load_state_dict(state_dict, strict=True) 
    vae = vae.cuda()
    vae = vae.eval()
    return vae, dataobj 