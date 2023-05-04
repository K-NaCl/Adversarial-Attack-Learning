from torch import nn

class BaseModel(nn.Module):
    def __init__(self) -> None:
        '''
        自定义nn.Module类，可自定义模型名称
        '''
        super().__init__()
         
    name: str 
    '模型名称'

    is_load: bool = False
    '是否为已训练模型'

    def set_name(self, name: str):
        self.name = name