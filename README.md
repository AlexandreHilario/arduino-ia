# Torre Anti-Zumbi com IA

Implementa o fluxo do PDF para uma maquete com acionamento simbolico:
HC-SR04 detecta algo a menos de 100 cm → Arduino envia `ANALISAR` → notebook
classifica a webcam → responde `Z` ou `N` → somente `Z` aciona servo.

A webcam permanece aberta; a inferencia ocorre quando o Arduino pede.
O modelo antigo incluido usa Zumbi e Pessoa (normal); o codigo tambem aceita Vazio
quando presente em uma nova exportacao. O limite padrao e 80%:
a classe vencedora deve ser Zumbi com pontuacao >= 0.80, senao a resposta e N.
A pontuacao nao representa acuracia medida. O processamento ocorre no notebook.

## Montagem da maquete

Desconecte a alimentacao antes de montar.

| Componente | Arduino Uno |
| --- | --- |
| HC-SR04 VCC / GND | 5V / GND |
| HC-SR04 TRIG / ECHO | 11 / 9 |
| Sinal do servo de acionamento | 7 |
| Sinal do servo de varredura | 13 |
| Alimentacao do servo | Fonte regulada adequada ao servo |
| GND da fonte do servo | GND comum com Arduino |

Nao alimente o servo por um pino de sinal. Consulte a tensao do seu servo.
O servo deve mover apenas o indicador ou mecanismo inofensivo da maquete.
No sketch, ajuste REPOUSO e ACIONADO conforme a montagem (padrao 0 e 60 graus).
O pulso dura 1.000 ms; pedidos ficam separados por pelo menos 1,5 s.
Sem resposta por 1,5 s, ou ao afastar o objeto, servos voltam ao repouso.
Uma resposta Z so e aceita com pedido pendente e objeto ainda perto.

Abra `arduino/detector_zumbi/detector_zumbi.ino` na Arduino IDE, instale a biblioteca
Servo se necessario, selecione Uno e carregue. Feche o Monitor Serial antes do Python.
Posicione sensor e webcam para observar o mesmo local; mantenha o objeto no centro.
O modelo classifica o recorte central da imagem, sem localizar objetos com caixas.

## Instalar

Use Python 3.11 de 64 bits. No Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No Linux, crie o ambiente com `python3 -m venv .venv` e use
`.venv/bin/python` nos comandos abaixo no lugar de `.\.venv\Scripts\python.exe`.

## Treinamento do modelo do zero — explicação para apresentação

### Objetivo e significado de “do zero”

O objetivo é classificar a imagem da câmera em **Zumbi** ou **Pessoa**. No código,
Pessoa recebe o nome `normal`. É um problema de classificação supervisionada:
cada foto tem uma resposta conhecida, indicada pela pasta em que foi colocada.
O modelo aprende a relacionar os pixels com esses rótulos.

O treinamento foi implementado em [`treinar.py`](treinar.py), usando
TensorFlow/Keras. A arquitetura foi definida no código e seus pesos treináveis
foram inicializados aleatoriamente. Não foram carregados pesos de MobileNet,
ImageNet ou de outro modelo pré-treinado; não houve transferência de aprendizado.
O modelo antigo da pasta `converted_tflite/` não participou desse treinamento.
O Teachable Machine também não foi usado para treinar este novo modelo.

“Do zero” se refere à inicialização e ao aprendizado da rede. As operações de
convolução, cálculo de gradientes e otimização são fornecidas pelo TensorFlow;
não implementamos uma biblioteca de redes neurais manualmente.

### 1. Organização das imagens

As fotos foram copiadas de `Downloads/zumbi` e `Downloads/pessoa`, preservando
os originais. A pasta Pessoa foi mapeada para `dados/normal` para manter a
compatibilidade com o programa existente.

```text
dados/
├── zumbi/    # 613 imagens, rótulo 0
└── normal/   # 604 imagens de pessoa, rótulo 1
```

Na importação, as imagens foram verificadas com Pillow e comparadas por hash
SHA-256 dentro de cada classe para evitar cópias de arquivos idênticos. Não foram
encontradas duplicatas exatas nessa comparação. Isso não identifica fotos
visualmente parecidas nem quadros consecutivos de um vídeo.

O conjunto utilizado contém **1.217 imagens** e duas classes. A classe `vazio`
não participou deste treinamento.

### 2. Separação entre treino e validação

Antes de qualquer transformação aleatória das imagens, o script embaralha os
arquivos de cada classe e reserva aproximadamente 15% para validação. Os outros
85% são usados no treino. O arredondamento é feito separadamente por classe.

| Classe | Treinamento | Validação | Total |
| --- | ---: | ---: | ---: |
| Zumbi | 521 | 92 | 613 |
| Pessoa (`normal`) | 513 | 91 | 604 |
| **Total** | **1.034** | **183** | **1.217** |

- **Treino:** imagens usadas para calcular os gradientes e ajustar os pesos.
- **Validação:** imagens usadas para acompanhar o desempenho e escolher os
  melhores pesos, sem atualizar os pesos diretamente com elas.
- **Teste independente:** novas imagens, fora dos dois conjuntos anteriores,
  para avaliar o modelo final. Esse teste ainda não foi executado neste fluxo.

A semente aleatória utilizada foi **42**, aplicada ao TensorFlow/Keras e ao
embaralhamento do NumPy. Com os mesmos arquivos e ambiente, isso ajuda a repetir
o experimento. Resultados numéricos podem variar conforme as versões e o hardware.
Os caminhos das imagens de cada divisão estão registrados em `treinamento.json`.

### 3. Preparação das imagens

Cada imagem passa pelas seguintes etapas:

1. Conversão para **RGB**, com três canais de cor.
2. Recorte central para proporção quadrada e redimensionamento para **96 × 96
   pixels**, usando `ImageOps.fit` e interpolação Lanczos.
3. Conversão para números `float32`.
4. Normalização dos pixels pela fórmula:

```text
pixel_normalizado = pixel_original / 127,5 - 1
```

Assim, o valor 0 vira −1, o valor 255 vira 1 e os valores intermediários ficam
nessa faixa. A entrada de uma imagem tem formato `96 × 96 × 3`.
O programa de reconhecimento aplica o mesmo preparo antes de executar o modelo.
O recorte central pode descartar partes das bordas: o objeto deve ficar centralizado.

Durante o treinamento, a rede também aplica **aumento de dados**: espelhamento
horizontal aleatório e rotação aleatória de até aproximadamente ±18°
(`RandomRotation(0.05)`, ou 5% de uma volta em cada sentido). Essas transformações
criam variações durante o processamento, sem salvar novas fotos nas pastas.
Elas ficam desligadas na validação e no uso da câmera.

### 4. Arquitetura da rede neural

Foi criada uma **CNN — rede neural convolucional**. As convoluções aprendem
filtros que respondem a padrões visuais úteis para separar as classes. Não foram
definidas manualmente regras como “olho de zumbi” ou “cor de pessoa”, e não foi
feita uma análise dos filtros para afirmar exatamente quais características
cada um aprendeu.

| Etapa | Configuração | Função |
| --- | --- | --- |
| Entrada | 96 × 96 × 3 | Receber a imagem RGB normalizada |
| Aumento de dados | Espelhamento e rotação | Variar os exemplos durante o treino |
| Convolução 1 | 16 filtros de 3 × 3, ReLU, padding `same` | Aprender padrões visuais |
| Max pooling 1 | Janela 2 × 2 | Reduzir a resolução para 48 × 48 |
| Convolução 2 | 32 filtros de 3 × 3, ReLU, padding `same` | Combinar os padrões aprendidos |
| Max pooling 2 | Janela 2 × 2 | Reduzir a resolução para 24 × 24 |
| Convolução 3 | 64 filtros de 3 × 3, ReLU, padding `same` | Produzir mapas de características |
| Global average pooling | Média espacial de cada mapa | Gerar um vetor com 64 valores |
| Dropout | Taxa de 0,3 | Desativar aleatoriamente 30% das entradas dessa camada no treino |
| Saída densa | 2 unidades, softmax | Produzir as pontuações de Zumbi e Pessoa |

A rede possui **23.714 parâmetros treináveis**, incluindo pesos e vieses.
ReLU aplica `max(0, x)` e permite que a rede represente relações não lineares.
O dropout ajuda a reduzir a dependência excessiva de algumas características,
mas não garante que o modelo generalize para qualquer cenário.
A softmax gera dois valores entre 0 e 1 cuja soma é aproximadamente 1.
Esses valores são pontuações do modelo, não garantias de acerto.

### 5. Como o aprendizado aconteceu

O comando usado para iniciar o treinamento foi:

```bash
TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=2 .venv/bin/python -u treinar.py --epocas 50 > treinamento.log 2>&1
```

Ele foi executado no processador (CPU). As variáveis de ambiente limitam o
paralelismo do TensorFlow. A saída do processo foi registrada em `treinamento.log`.
Uma execução comum, mostrando o progresso no terminal, pode ser iniciada com:

```bash
.venv/bin/python treinar.py --epocas 50
```

No Windows, substitua `.venv/bin/python` por `.\venv\Scripts\python.exe`.
Executar novamente inicia outro treinamento do zero e grava os resultados na
pasta de saída. Para preservar a execução anterior, use, por exemplo,
`--saida modelo_nova_execucao`.

| Parâmetro | Valor usado |
| --- | --- |
| Otimizador | Adam, com os padrões do Keras usado na execução |
| Função de perda | `sparse_categorical_crossentropy` |
| Métrica acompanhada | Acurácia (`accuracy`) |
| Tamanho do lote | 32 imagens |
| Máximo de épocas | 50 |
| Critério de parada | Perda de validação (`val_loss`) |
| Paciência | 8 épocas sem melhorar |
| Restauração dos melhores pesos | Ativada |
| Semente | 42 |

Para cada lote, a rede faz uma previsão, compara a saída com os rótulos corretos
e calcula a **perda**, um número que expressa o erro da previsão. A entropia
cruzada considera a pontuação atribuída à classe correta, e não apenas se a
classe vencedora está certa. Por isso, duas épocas podem ter a mesma acurácia e
perdas diferentes.

A retropropagação calcula como cada parâmetro contribuiu para a perda. O Adam
usa esses gradientes para atualizar os parâmetros. Esse processo se repete ao
longo dos lotes. Uma **época** corresponde a uma passagem por todo o conjunto de
treino; nesta execução, foram 33 lotes por época, com o último lote menor.
Após cada época, o modelo é avaliado nas imagens de validação.

### 6. Parada e escolha do modelo final

O treinamento completou **48 épocas**, encerrando antes do limite de 50.
A menor perda de validação ocorreu na **época 40**. Como as oito épocas
seguintes não melhoraram essa perda, o `EarlyStopping` encerrou o processo e
restaurou os pesos da época 40.

Portanto, o modelo exportado corresponde aos melhores pesos segundo a perda de
validação, e não simplesmente aos pesos da última época executada.

### 7. Resultados reais da execução

Os números abaixo foram conferidos no arquivo
[`modelo_do_zero/treinamento.json`](modelo_do_zero/treinamento.json).

| Indicador | Resultado |
| --- | ---: |
| Épocas executadas | 48 |
| Época dos pesos selecionados | 40 |
| Acurácia de treino registrada na época 40 | Aproximadamente 99,90% |
| Perda de treino registrada na época 40 | Aproximadamente 0,01020 |
| Acurácia de validação do modelo restaurado | **100% — 183 acertos em 183 imagens** |
| Perda de validação do modelo restaurado | Aproximadamente 0,00510 |

A métrica de treino é a registrada durante a época, com aumento de dados e
dropout ativos; não é uma nova avaliação do modelo final em todas as fotos de
treino. A validação é feita com essas operações aleatórias desligadas.

A matriz de confusão tem as **classes reais nas linhas** e as **previstas nas
colunas**, na ordem Zumbi e Pessoa:

| Classe real | Previsão: Zumbi | Previsão: Pessoa |
| --- | ---: | ---: |
| Zumbi | 92 | 0 |
| Pessoa | 0 | 91 |

A acurácia foi calculada como `(92 + 91) / 183 = 1`, equivalente a 100%.
Essa matriz considera a classe vencedora. Ela não mede diretamente a decisão de
acionar o servo com o limiar de 80%, que é uma etapa posterior.

### 8. Como interpretar os 100% com cuidado

O resultado demonstra que o modelo separou corretamente as imagens da validação
usada neste experimento. **Não demonstra 100% de acerto em qualquer situação.**
A validação também foi usada para selecionar os pesos, portanto não equivale a
um teste final independente.

A separação atual é aleatória por imagem, sem agrupar por vídeo, pessoa, boneco
ou sessão de captura. Se houver fotos muito semelhantes ou quadros próximos de
um mesmo vídeo nos dois conjuntos, o resultado pode ser otimista. A verificação
de arquivos idênticos não elimina esse problema.

Para avaliar a generalização, é necessário usar novas sessões de captura, com
variações de iluminação, posição, fundo e objetos, mantendo essas imagens fora
do treinamento e da seleção dos pesos. Em um experimento futuro com vídeos,
o ideal é separar sessões ou vídeos inteiros entre os conjuntos.

O modelo atual não tem uma classe de cenário vazio e pode atribuir Zumbi ou
Pessoa mesmo a uma imagem que não contenha nenhum dos dois. Ele classifica o
recorte central; não localiza objetos com caixas nem conta quantas pessoas ou
zumbis aparecem.

### 9. Arquivos gerados e exportação

```text
modelo_do_zero/
├── modelo.keras
├── model_unquant.tflite
├── labels.txt
└── treinamento.json
```

| Arquivo | Conteúdo e uso |
| --- | --- |
| `modelo.keras` | Rede treinada no formato Keras, para reabrir e inspecionar |
| `model_unquant.tflite` | Rede exportada em ponto flutuante (`float32`), usada pelo programa da câmera |
| `labels.txt` | Ordem das classes de saída do modelo |
| `treinamento.json` | Semente, classes, caminhos das imagens de cada divisão, histórico e métricas de validação |
| `treinamento.log` (na raiz) | Registro da saída do treinamento e da conversão |

O arquivo de rótulos contém:

```text
0 zumbi
1 normal
```

A exportação para TensorFlow Lite converte a rede já treinada; não realiza outro
treinamento. A inferência exportada recebe uma imagem por vez, com formato
`1 × 96 × 96 × 3`, e devolve duas pontuações. Não foi aplicada quantização.
O espelhamento, a rotação aleatória e o dropout ficam desligados na inferência.

Antes de gravar o `.tflite`, o script compara sua saída com a saída do modelo
Keras para uma imagem de validação, com tolerância numérica. Também foi
verificado que a classe `Modelo` de `projeto.py` consegue carregar a exportação
e executar uma previsão. Essa verificação de compatibilidade não substitui
uma avaliação completa do modelo convertido com imagens novas.

### 10. Ligação entre a IA, a câmera e o Arduino

```text
HC-SR04 detecta aproximação a menos de 100 cm
    → Arduino envia ANALISAR pela serial
    → Python prepara a imagem da câmera e executa o modelo
    → Zumbi com pontuação >= 0,80: envia Z
    → Qualquer outra decisão: envia N
    → Arduino recebe a resposta e controla o servo
```

O treinamento e a classificação acontecem no computador. O arquivo `.tflite`
não foi instalado no Arduino. O Arduino mede a distância e controla o mecanismo
simbólico da maquete conforme a mensagem recebida.

O **limiar de 80%** é uma regra de decisão sobre a pontuação de uma imagem.
A **acurácia de 100% na validação** é a proporção de imagens classificadas
corretamente naquele conjunto. São medidas diferentes.

O programa ainda aponta por padrão para o modelo antigo em `converted_tflite/`.
Por isso, os comandos abaixo indicam explicitamente o novo modelo e seus rótulos.

Teste somente com a câmera do computador (Q encerra):

```bash
.venv/bin/python projeto.py executar --sem-arduino --camera 0 --modelo modelo_do_zero/model_unquant.tflite --rotulos modelo_do_zero/labels.txt
```

Com Arduino, ajuste a porta conforme a montagem:

```bash
.venv/bin/python projeto.py executar --porta /dev/ttyUSB0 --camera 0 --modelo modelo_do_zero/model_unquant.tflite --rotulos modelo_do_zero/labels.txt
```

O sketch atual está com `TESTE_MANUAL = false`, preparado para a integração
automática. Carregue-o na placa e feche o Monitor Serial antes de abrir o Python.
O limite atual de aproximação é 100 cm (`DISTANCIA_CM`); o segundo servo faz
varredura de 20° a 160°. Esses parâmetros pertencem ao Arduino e não ao treino.
A execução física completa depende da montagem e deve ser verificada nela.

### 11. Avaliação independente que ainda falta executar

Separe pelo menos 20 imagens novas por classe, capturadas em outra sessão e
não usadas no treino ou na validação:

```text
testes/
├── zumbi/
└── normal/
```

Execute:

```bash
.venv/bin/python projeto.py avaliar --dados testes --modelo modelo_do_zero/model_unquant.tflite --rotulos modelo_do_zero/labels.txt
```

O comando gera `avaliacao.json`, com acurácia, matriz de confusão das classes e
matriz das decisões N/Z considerando o limiar. Assim é possível verificar tanto
a classificação quanto os acionamentos indevidos. Não há resultado desse teste
independente a apresentar antes de executá-lo com as novas imagens.

### 12. Roteiro curto para explicar ao professor

> “Organizamos 1.217 imagens em duas classes: zumbi e pessoa. Criamos uma CNN em
> TensorFlow/Keras com 23.714 parâmetros inicializados aleatoriamente, sem usar
> pesos pré-treinados. Separamos 1.034 imagens para treino e 183 para validação.
> As imagens foram recortadas e redimensionadas para 96 por 96 pixels, com
> normalização e aumento de dados durante o treino. Usamos Adam e entropia
> cruzada para ajustar os pesos. O treinamento encerrou na época 48 e recuperou
> os melhores pesos, da época 40. O modelo acertou as 183 imagens de validação,
> mas ainda precisa ser avaliado em uma nova sessão de captura. Exportamos a
> rede para TensorFlow Lite; ela roda no computador, e o Python envia ao Arduino
> a decisão usada para controlar o servo da maquete.”

Para demonstrar o trabalho, abra `treinar.py`, a divisão e o histórico em
`treinamento.json`, os rótulos em `labels.txt` e a execução da câmera com o
caminho explícito para o modelo novo.

## Executar

Teste primeiro sem Arduino:

```powershell
.\.venv\Scripts\python.exe projeto.py executar --sem-arduino
```

Depois conecte o Arduino:

```powershell
.\.venv\Scripts\python.exe projeto.py executar --porta COM3
```

No Linux, a porta pode ser `/dev/ttyACM0`. Q encerra e envia N.
Opcoes: `--camera 1`, `--confianca 0.85`, `--modelo caminho.tflite`,
`--rotulos caminho/labels.txt`. No teste sem Arduino, a inferencia e continua.
A classificacao nao exige tres confirmacoes: segue a decisao por imagem do PDF.

## Usar a camera do celular por USB (Arch Linux / CachyOS)

Execute os comandos do projeto na pasta `~/Projetos/arduino-ia`.
Instale as ferramentas, caso ainda nao estejam instaladas:

```bash
sudo pacman -S scrcpy android-tools
```

Ative a depuracao USB no Android, conecte o celular e aceite a autorizacao
na tela. Confira com `adb devices`: o estado deve ser `device`.

No primeiro terminal, prepare a webcam virtual e inicie a camera traseira:

```bash
sudo modprobe v4l2loopback video_nr=10 card_label="Celular" exclusive_caps=1
sudo setfacl -m "u:$USER:rw" /dev/video10
scrcpy --video-source=camera --camera-facing=back --camera-size=1280x720 --camera-fps=30 --no-audio --v4l2-sink=/dev/video10 --no-playback
```

O modulo `v4l2loopback` precisa estar instalado para o kernel em uso.
A permissao com `setfacl` precisa ser reaplicada se o dispositivo for recriado.
Mantenha esse terminal rodando; se o scrcpy encerrar ou apresentar erro, resolva
antes de abrir o Python. Com `exclusive_caps=1`, a camera fica disponivel para
captura quando o scrcpy comeca a transmitir.

Em outro terminal, teste a imagem sem Arduino:

```bash
cd ~/Projetos/arduino-ia
.venv/bin/python projeto.py executar --sem-arduino --camera /dev/video10
```

Pressione Q para fechar o teste. Para testar tudo, carregue o sketch com
`TESTE_MANUAL = false`, feche o Monitor Serial da Arduino IDE e execute:

```bash
.venv/bin/python projeto.py executar --porta /dev/ttyUSB0 --camera /dev/video10
```

Esses comandos usam o modelo padrao em `converted_tflite/`. Para usar o modelo
treinado do zero, execute:

```bash
.venv/bin/python projeto.py executar --porta /dev/ttyUSB0 --camera /dev/video10 --modelo modelo_do_zero/model_unquant.tflite --rotulos modelo_do_zero/labels.txt
```

Use o caminho `/dev/video10` explicitamente para selecionar o celular.
Se nao abrir, confira `ls -l /dev/video10`, as permissoes e a saida do scrcpy.
Q encerra o projeto; Ctrl+C no primeiro terminal encerra a transmissao do celular.

## Avaliar

Separe pelo menos 20 imagens novas por classe do modelo, de objetos/cenas nao usados
no treino, em `testes/zumbi` e `testes/normal` (Pessoa). Se exportar um modelo com
Vazio, adicione tambem `testes/vazio`. Para captura direta nesse conjunto,
copie para essas pastas apenas fotos novas e mantenha-as fora de `dados/`.

```powershell
.\.venv\Scripts\python.exe projeto.py avaliar --modelo modelo_do_zero/model_unquant.tflite --rotulos modelo_do_zero/labels.txt
```

Gera `avaliacao.json` com totais, acuracia e matriz de confusao (linhas reais,
colunas previstas, na ordem registrada em `classes` no JSON). Tambem gera uma matriz da decisao
N/Z com o limite aplicado, permitindo contar acionamentos indevidos.
Esse teste independente complementa a validacao do treinamento; nao a substitui.
Nenhuma acuracia e informada antes de executar a avaliacao com imagens reais.

O código efetivamente utilizado no treinamento está em `treinar.py`; os resultados
da execução estão em `modelo_do_zero/treinamento.json`.

## Executar no Arch Linux / CachyOS

Use o ambiente Python 3.11 em `.venv` (o Python do sistema pode ter outra versao).
Nao e necessario ativar o ambiente se usar o caminho completo:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python projeto.py executar --sem-arduino --camera 0
.venv/bin/python projeto.py executar --porta /dev/ttyUSB0 --camera 0
```

A porta `/dev/ttyUSB0` foi encontrada nesta maquina; confira novamente apos reconectar.
Na Arduino IDE, selecione a placa e essa porta, carregue o sketch e feche o Monitor
Serial antes de executar Python. Se a placa for um clone Nano, selecione a placa
real em vez de Uno. A porta USB sozinha nao identifica o modelo da placa.

Se houver erro de permissao na serial, confira `id` e `ls -l /dev/ttyUSB0`.
No Arch, normalmente a serial pertence a `uucp`; se seu usuario nao pertencer ao
grupo, use `sudo usermod -aG uucp "$USER"` e encerre/inicie a sessao.
Nao execute o aplicativo Python com sudo.

Com o fluxo completo aberto, centralize o objeto e aproxime a menos de 100 cm.
Pessoa deve gerar N; Zumbi com pontuacao >=80% deve gerar Z e um pulso de servo.
Afaste o objeto e confira o repouso. Q encerra. Sem a classe Vazio, teste tambem o
cenario vazio para identificar falsos positivos. Estes testes exigem a montagem real.
