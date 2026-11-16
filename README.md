# Torre Anti-Zumbi com IA

Implementa o fluxo do PDF para uma maquete com acionamento simbolico:
HC-SR04 detecta algo a menos de 200 cm → Arduino envia `ANALISAR` → notebook
classifica a webcam → responde `Z` ou `N` → somente `Z` aciona servo e LED.

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
| HC-SR04 TRIG / ECHO | 9 / 10 |
| LED com resistor de 220 ohms | 7 / GND |
| Sinal do servo | 6 |
| Alimentacao do servo | Fonte regulada adequada ao servo |
| GND da fonte do servo | GND comum com Arduino |

Nao alimente o servo por um pino de sinal. Consulte a tensao do seu servo.
O servo deve mover apenas o indicador ou mecanismo inofensivo da maquete.
No sketch, ajuste REPOUSO e ACIONADO conforme a montagem (padrao 0 e 60 graus).
O pulso dura 350 ms; pedidos ficam separados por pelo menos 1,5 s.
Sem resposta por 1,5 s, ou ao afastar o objeto, servo e LEDs voltam ao repouso.
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

## Criar e treinar o modelo do zero

O treinamento local em `treinar.py` cria uma CNN com pesos aleatorios, sem
MobileNet ou outro modelo pre-treinado. O modelo antigo em `converted_tflite/`
nao e usado no treinamento. A IA continua rodando no notebook.

As fotos de Downloads foram copiadas para `dados/zumbi` e `dados/normal`
(Pessoa). Os originais foram preservados. O treino padrao usa essas duas classes.

Para novas capturas, tire cerca de 200 fotos ou mais por classe na propria maquete. ESPACO salva;
Q sai. Varie luz, distancia, angulo e fundo em todas as classes.

```bash
.venv/bin/python projeto.py capturar zumbi
.venv/bin/python projeto.py capturar normal
.venv/bin/python treinar.py --epocas 50
```

No Windows, substitua `.venv/bin/python` por `.\venv\Scripts\python.exe`.
As fotos ficam em `dados/<classe>/`. O script separa aproximadamente 85% para
treino e 15% para validacao por classe, com semente fixa. Nao coloque fotos
quase identicas nas duas partes: para uma avaliacao independente, reserve outra
sessao de captura em `testes/`, fora de `dados/`. O treino pode encerrar antes
se a perda de validacao deixar de melhorar; recupera os melhores pesos.

A CNN recebe imagens RGB de 96 x 96, recortadas no centro e normalizadas para
[-1, 1], como na execucao. Tem tres convolucoes, pooling, dropout e uma saida
softmax para Zumbi e Normal (Pessoa). Para incluir uma terceira classe, capture
fotos em `dados/vazio` e treine com `--incluir-vazio`. Todas as camadas aprendem do zero.
Treinar sem pesos pre-treinados pode exigir mais fotos para generalizar bem.

A pasta `modelo_do_zero/` recebe:

- `modelo.keras`: rede treinada para reabrir no TensorFlow/Keras.
- `model_unquant.tflite`: exportacao float32 para o programa existente.
- `labels.txt`: classes na ordem das saidas.
- `treinamento.json`: fotos de cada divisao, historico, acuracia e matriz de
  confusao de validacao. Esses numeros nao substituem o teste independente.

Use explicitamente os novos arquivos para testar e avaliar:

```bash
.venv/bin/python projeto.py executar --sem-arduino --modelo modelo_do_zero/model_unquant.tflite --rotulos modelo_do_zero/labels.txt
.venv/bin/python projeto.py avaliar --modelo modelo_do_zero/model_unquant.tflite --rotulos modelo_do_zero/labels.txt
```

Para a entrega, inclua `treinar.py`, os arquivos gerados e os resultados reais
do teste. Nenhum resultado de acuracia e garantido antes de treinar e avaliar.

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

## Avaliar

Separe pelo menos 20 imagens novas por classe do modelo, de objetos/cenas nao usados
no treino, em `testes/zumbi` e `testes/normal` (Pessoa). Se exportar um modelo com
Vazio, adicione tambem `testes/vazio`. Para captura direta nesse conjunto,
copie para essas pastas apenas fotos novas e mantenha-as fora do Teachable Machine.

```powershell
.\.venv\Scripts\python.exe projeto.py avaliar
```

Gera `avaliacao.json` com totais, acuracia e matriz de confusao (linhas reais,
colunas previstas, na ordem registrada em `classes` no JSON). Tambem gera uma matriz da decisao
N/Z com o limite aplicado, permitindo contar acionamentos indevidos.
Esse teste independente complementa a validacao do treinamento; nao a substitui.
Nenhuma acuracia e informada antes de executar a avaliacao com imagens reais.

Referencias: [exportacao Teachable Machine](https://github.com/googlecreativelab/teachablemachine-community/tree/master/snippets)
e [interpretador TensorFlow Lite](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter).

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

Com o fluxo completo aberto, centralize o objeto e aproxime a menos de 200 cm.
Pessoa deve gerar N; Zumbi com pontuacao >=80% deve gerar Z e um pulso de servo/LED.
Afaste o objeto e confira o repouso. Q encerra. Sem a classe Vazio, teste tambem o
cenario vazio para identificar falsos positivos. Estes testes exigem a montagem real.
