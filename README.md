# 🎬 Dublar - Sistema de Dublagem Automática de Vídeos

Sistema completo de dublagem automática de vídeos usando IA para transcrição, tradução e síntese de voz.

## 🌟 Características

- **Transcrição automática** com Whisper (faster-whisper)
- **Tradução** com facebook/m2m100_418M (suporta múltiplos idiomas)
- **Síntese de voz** com Bark ou Coqui TTS
- **Sincronização inteligente** com 4 modos: none, fit, pad, smart
- **Suporte GPU** NVIDIA CUDA (opcional, funciona em CPU)
- **Preservação de termos técnicos** em vídeos de programação
- **Processamento em lote** de múltiplos vídeos

## 🚀 Início Rápido

### Windows:
```bash
git clone https://github.com/inematds/dublar.git
cd dublar
instalar.bat
dublar.bat video.mp4
```

### Linux:
```bash
git clone https://github.com/inematds/dublar.git
cd dublar
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python3 dublar.py video.mp4 --src_lang en --tgt_lang pt --tts bark --sync smart
```

**📖 [Guia completo de instalação Linux](INSTALL_LINUX.md)**

## 📋 Requisitos

- **Python**: 3.10 ou superior
- **FFmpeg**: Obrigatório (processamento de vídeo/áudio)
- **RAM**: 8GB mínimo, 16GB+ recomendado
- **Disco**: 10GB+ para modelos de IA
- **GPU** (opcional): NVIDIA com CUDA 11.8+ para processamento mais rápido

## 🎯 Uso Básico

### Sintaxe:
```bash
python dublar.py VIDEO.mp4 [opções]
```

### Exemplos:

**Inglês → Português (padrão):**
```bash
python dublar.py tutorial.mp4
```

**Espanhol → Inglês:**
```bash
python dublar.py video.mp4 --src_lang es --tgt_lang en
```

**Vídeo técnico com preservação de termos:**
```bash
python dublar_tech_v2.py codigo.mp4 --src_lang en --tgt_lang pt
```

**Usar voz específica:**
```bash
python dublar.py video.mp4 --voice v2/pt_speaker_5
```

### Parâmetros Principais:

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--src_lang` | `en` | Idioma de origem (en, es, fr, pt, etc.) |
| `--tgt_lang` | `pt` | Idioma de destino |
| `--tts` | `bark` | Engine de voz (`bark` ou `coqui`) |
| `--sync` | `smart` | Modo de sincronização (`none`, `fit`, `pad`, `smart`) |
| `--voice` | - | Voz específica (ex: `v2/pt_speaker_5`) |
| `--rate` | `22050` | Taxa de amostragem do áudio final |

### Modos de Sincronização:

- **`none`**: Sem ajuste, pode desincronizar
- **`fit`**: Comprime/expande áudio para caber no tempo
- **`pad`**: Adiciona silêncio se necessário
- **`smart`**: Automático (pad se curto, fit se longo) - **Recomendado**

## 📁 Estrutura de Arquivos

```
dublar/
├── dublar.py              # Script principal
├── dublar2.py             # Versão com melhorias de sync
├── dublar3.py             # Versão com output em pasta separada
├── dublar31.py            # Versão com preservação de gaps
├── dublar_tech_v2.py      # Otimizado para vídeos técnicos
├── dublar_sync_v2.py      # Versão com sync avançado
│
├── requirements.txt       # Dependências Python
│
├── instalar.bat           # Instalador Windows
├── dublar.bat             # Launcher Windows
├── ativar_gpu.bat         # Configurar GPU no Windows
│
├── test_*.py              # Scripts de teste
│
├── INSTALL_LINUX.md       # 📖 Guia de instalação Linux
├── README_TECH.md         # Documentação versão técnica
├── README_SYNC_V2.md      # Documentação sync v2
├── README_BAT.md          # Documentação scripts Windows
├── MAPA_ARQUIVOS.md       # Estrutura detalhada do projeto
│
├── venv/                  # Ambiente virtual (você cria)
├── dub_work/              # Temporários (criado automaticamente)
└── dublado/               # Vídeos finais (criado automaticamente)
```

## 🔧 Versões Disponíveis

### Scripts Principais:

1. **`dublar.py`** - Versão base estável
   - Funcionalidades essenciais
   - Melhor para começar

2. **`dublar_tech_v2.py`** - Para vídeos técnicos
   - Glossário de 100+ termos técnicos
   - Preserva nomes de tecnologias
   - Otimizado para tutoriais de programação
   - [📖 Documentação](README_TECH.md)

3. **`dublar_sync_v2.py`** - Sincronização avançada
   - Melhor alinhamento de áudio
   - Controle fino de timing
   - [📖 Documentação](README_SYNC_V2.md)

4. **`dublar31.py`** - Com preservação de pausas
   - Mantém silêncios entre frases
   - Mais natural para palestras

## 🧪 Testar Instalação

```bash
# Ativar ambiente (se ainda não ativou)
source venv/bin/activate  # Linux
# venv\Scripts\activate   # Windows

# Teste rápido
python test_quick.py

# Testar GPU
python test_gpu.py

# Testar Whisper
python test_whisper_gpu.py
```

## 🌍 Idiomas Suportados

### Transcrição (Whisper):
Mais de 90 idiomas incluindo: en, pt, es, fr, de, it, ja, ko, zh, ru, ar, hi, etc.

### Tradução (M2M100):
100 idiomas incluindo todos os principais.

### Síntese de Voz:
- **Bark**: Multilíngue (en, pt, es, fr, de, it, pl, zh, ja, hi, etc.)
- **Coqui TTS**: Varia por modelo

## 📊 Fluxo de Processamento

```
1. Extração de Áudio (FFmpeg)
   video.mp4 → audio_src.wav

2. Transcrição (Whisper)
   audio_src.wav → asr.srt + asr.json

3. Tradução (M2M100)
   asr.json → asr_trad.json + asr_trad.srt

4. Síntese de Voz (Bark/Coqui)
   asr_trad.json → seg_0001.wav, seg_0002.wav, ...

5. Sincronização (smart/fit/pad/none)
   seg_*.wav → seg_*_fit.wav

6. Concatenação + Pós-processamento
   seg_*_fit.wav → dub_final.wav

7. Mixagem Final (FFmpeg)
   video.mp4 + dub_final.wav → video_dublado.mp4
```

## 🎓 Documentação Adicional

- **[INSTALL_LINUX.md](INSTALL_LINUX.md)** - Guia completo de instalação no Linux
- **[README_TECH.md](README_TECH.md)** - Versão otimizada para vídeos técnicos
- **[README_SYNC_V2.md](README_SYNC_V2.md)** - Sincronização avançada
- **[README_BAT.md](README_BAT.md)** - Scripts Windows (.bat)
- **[MAPA_ARQUIVOS.md](MAPA_ARQUIVOS.md)** - Estrutura completa do projeto
- **[FIX_CUDA.md](FIX_CUDA.md)** - Solução de problemas CUDA
- **[INSTALACAO_GPU.md](INSTALACAO_GPU.md)** - Configurar GPU NVIDIA

## ⚠️ Notas Importantes

1. **Diretórios automáticos**: `dub_work/` e `dublado/` são criados automaticamente - não crie manualmente
2. **FFmpeg obrigatório**: Sem FFmpeg nada funciona
3. **GPU opcional**: Funciona em CPU, apenas mais lento
4. **Primeira execução**: Download de modelos pode levar tempo (5-10GB)
5. **Memória**: Vídeos longos (+30min) podem precisar de 16GB+ RAM

## 🐛 Solução de Problemas

### FFmpeg não encontrado:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Baixe de https://ffmpeg.org e adicione ao PATH
```

### Erro de CUDA/GPU:
```bash
# Forçar CPU (mais estável)
# Edite o script e mude device="cuda" para device="cpu"
```

### Falta de memória:
```bash
# Use vídeos menores ou processe em partes
# Reduza batch_size no código
```

### Modelos não baixam:
```bash
# Verifique conexão com internet
# Modelos são baixados automaticamente do HuggingFace
```

## 📝 Logs e Debug

Cada execução gera:
- `dub_work/logs.json` - Log completo do processo
- `dub_work/asr.srt` - Transcrição original
- `dub_work/asr_trad.srt` - Tradução
- `dub_work/*.wav` - Arquivos de áudio intermediários

## 🤝 Contribuindo

Pull requests são bem-vindos! Para mudanças maiores, abra uma issue primeiro.

## 📄 Licença

Este projeto é open source.

## 🔗 Links Úteis

- **FFmpeg**: https://ffmpeg.org
- **Whisper**: https://github.com/openai/whisper
- **Bark**: https://github.com/suno-ai/bark
- **M2M100**: https://huggingface.co/facebook/m2m100_418M

---

**Desenvolvido com ❤️ usando IA de ponta para dublagem de vídeos**
