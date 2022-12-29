<h1>Trabalho 3 da Disciplina de Visão Computacional</h1>
<h2>Alunos: Paulo Victor e João Davi<h2/>

Para rodar o projeto, é necessário ter instalado o python3 e o pip3. Após isso, basta rodar os comandos abaixo para instalar as dependências do projeto:

```
pip3 install scitkit-learn
pip3 install opencv-python
pip3 install numpy
pip3 install matplotlib

```

Para treinar a rede com nossa proposta de arquitetura, basta executar todas as células do arquivo `train.ipynb`.

Para treinar a rede feita com transfer-learning do VGG16, basta executar todas as células do arquivo `transfer-learning/main.ipynb`.

Para testar a rede, basta executar todas as células do arquivo `detectFaces.ipynb`, colocando qual rede gostaria de utilizar.

É possível alterar parâmetros no momento do treino, os utilizados foram:
Loss: binary_crossentropy
Optimizer: Adam
Learning Rate: 0.0001
Batch Size: 64
Epochs: 120

Para pré-processamento, foram realizadas as técnicas:

- Clean data and normalize
- Data augmentation
