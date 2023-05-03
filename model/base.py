from torch import nn

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    name: str

    def set_name(self, name: str):
        self.name = name