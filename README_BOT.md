# inemaVOX Bot

Bot do Telegram com todas as funcionalidades do inemaVOX para processamento de video e audio com IA.

## Funcionalidades

- **Download de Video**: Baixa videos do YouTube, TikTok, Instagram, Facebook e +1000 sites
- **Transcricao**: Gera legendas SRT/TXT com Whisper (tiny ate large-v3)
- **Dublagem**: Traduz e dubla videos para qualquer idioma com voces Edge TTS
- **Corte**: Extrai clips por timestamps
- **TTS**: Texto para Fala com voces realistas
- **Clonagem de Voz**: Clona voz a partir de audio de referencia

## Instalacao

1. Instale as dependencias do bot:
```bash
pip install -r requirements_bot.txt
```

2. Configure o token do bot:
```bash
# Windows
set TELEGRAM_BOT_TOKEN=seu_token_aqui

# Linux/Mac
export TELEGRAM_BOT_TOKEN=seu_token_aqui
```

3. Execute o bot:
```bash
python bot_telegram.py
```

## Como Obter um Token

1. Fale com @BotFather no Telegram
2. Use o comando /newbot
3. Siga as instrucoes e obtenha o token
4. Use /start no seu novo bot para ativa-lo

## Comandos

- `/start` - Iniciar o bot e ver o menu
- `/help` - Ver ajuda
- `/status` - Ver status do sistema

## Uso

1. Execute `/start` para ver o menu
2. Selecione a operacao desejada
3. Siga as instrucoes na tela

### Download
- Selecione "Download"
- Envie a URL do video

### Transcrever
- Selecione "Transcrever"
- Escolha o modelo (tiny=rapido, large-v3=preciso)
- Envie o arquivo de video ou URL

### Dublar
- Selecione "Dublar"
- Escolha o idioma original
- Escolha o idioma de destino
- Escolha a voz
- Envie o arquivo de video ou URL

### TTS
- Selecione "TTS"
- Escolha o idioma
- Escolha a voz
- Envie o texto

### Clonar Voz
- Selecione "Clonar Voz"
- Envie um audio de referencia (5-30 segundos)
- Envie o texto que deseja gerar

## Requisitos do Sistema

- Python 3.8+
- Todas as dependencias do inemaVOX (requirements.txt)
- yt-dlp instalado
- FFMPEG instalado

## Notas

- O bot processa videos localmente, entao pode levar alguns minutos
- Videos grandes podem requerer mais tempo
- Asegure-se de ter espaco em disco suficiente

## Executando com Docker

### Opcao 1: docker-compose

1. Crie um arquivo `.env` com seu token:
```
TELEGRAM_BOT_TOKEN=seu_token_aqui
```

2. Execute:
```bash
docker-compose -f docker-compose.bot.yml up -d
```

### Opcao 2: Docker manual

1. Build:
```bash
docker build -f Dockerfile.bot -t inemavox-bot .
```

2. Run:
```bash
docker run -d --name inemavox-bot -e TELEGRAM_BOT_TOKEN=seu_token inemavox-bot
```

### Com GPU NVIDIA

Para usar com GPU NVIDIA (recomendado para dublagem):
```bash
docker run -d --name inemavox-bot \
  --gpus all \
  -e TELEGRAM_BOT_TOKEN=seu_token \
  -v $(pwd)/jobs:/app/jobs \
  inemavox-bot
```

### Volumes

- `./jobs:/app/jobs` - Persiste os arquivos processados
- `./temp:/app/temp` - Persiste arquivos temporarios
