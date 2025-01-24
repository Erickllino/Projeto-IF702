# criando o modelo da rede CNN
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader




# Criar uma classe que herda o modulo nn.Module

class CNN_Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # transformar a entrada em um único tensor unidimensional

        self.Fl = nn.Flatten()
        
        # a primeira camada de convolução
        # o parâmetro in_channels é para identificar as dimensões que a representação da sua imagem possuí, ex RGB = 3, gray_scale = 1, ou os mapas de características dado por camadas anteriores
        # e o out_channels é para a saída gerada pela unidade computacional, maiores valores gerarão mais mapas de características
        # o kernel e o stride podem assumir valores diferentes, mas por fins de simplificação iremos apenas passar inteiros

        self.C1 = nn.Conv2d(in_channels = 1, out_channels = 5, kernel_size = 4, stride = 1 )

        # após a operação de convolução, virá a operação de pooling
        # existem diferentes formas pela biblioteca para fazer a operação de pooling, mas utilizaremos o Max

        self.P1 = nn.MaxPool2d( kernel_size = 2)
        
        #segunda camada de convolução + pooling

        self.C2 = nn.Conv2d(in_channels = 5, out_channels = 10, kernel_size = 4, stride = 1)
        self.P2 = nn.MaxPool2d(kernel_size = 4)
        

        # Após n operações de convolução e pooling a imagem será achatada pela operação Flatten
        # e será usado em uma rede MLP
        # o pytorch não oferece maneiras dinâmicas de fazer alteração nos parâmetros da rede após a criação
        # então o valor das entradas e saídas após cada camada devem ser calculadas: output_size=((input_size−kernel_size+2×padding)/stride) + 1
        # para a primeira camada após a convolução temos output_size = 25
        # para a primeira camada após o pooling temos output_size = 13
        # para a segunda camada após a convolução temos output_size = 10
        # para a segunda camada após o pooling temos output_size = 2
        # então para a rede MLP teremos uma quantidade de entradas 10*2*2

        #primeira camada terá 100 unidades computacionais
        self.L1 = nn.Linear(10*16*16, 100)

        #segunda camada para 50 unidades computacionais
        self.L2 = nn.Linear(100, 50)

        #terceira camada para 10
        self.L3 = nn.Linear(50, 10)


    def forward(self, x):
        
        
        # primeira camada de convolução + função relu
        x = nn.relu(self.C1(x))
        
        # primeira camada de pooling
        x = self.P1(x)
        
        # segunda camada de convolução + função relu

        x = nn.relu(self.C2(x))

        # segunda camada de pooling

        x = self.P2(x)

        # a operação de achatamento

        x = self.Fl(x)

        # as 3 camadas da rede MLP + função relu
        
        x = nn.relu(self.L1(x))
        
        x = nn.relu(self.L2(x))

        x = self.L3(x)
        
        return x       
