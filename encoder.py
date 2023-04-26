import torch
import torch.nn as nn
import git
import sys

class Encoder(nn.Module):
    """
    ViT encoder based on TinyViT pre-trained model
    Please see https://github.com/microsoft/Cream/tree/main/TinyViT for details
    """
    def __init__(self):
        super(Encoder, self).__init__()
        home_dir = git.Repo('.', search_parent_directories=True).working_tree_dir
        sys.path.append(home_dir + '/TinyViT')
        self.model = torch.load(home_dir + '/TinyViT/tinyvit_21M.pt')

        # Freeze patch_embed and layer[0] (since they extract high-level feature representations)
        for param in self.model.patch_embed.parameters():
            param.requires_grad = False
        
        for param in self.model.layers[0].parameters():
            param.requires_grad = False


    def forward(self, X):
        out = self.model(X)
        return out


    def unfreeze(self):
        for param in self.model.patch_embed.parameters():
            param.requires_grad = True
        
        for param in self.model.layers[0].parameters():
            param.requires_grad = True



if __name__ == "__main__":
    model = Encoder()
    print(f"{model=}")
