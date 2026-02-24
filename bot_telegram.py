#!/usr/bin/env python3
"""
inemaVOX Bot - Telegram Bot com todas as funcionalidades
Download, Transcricao, Dublagem, Corte, TTS e Voice Clone
"""

import os
import sys
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

# Configuracao de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# IMPORTS DO PROJETO
# ============================================
from pathlib import Path
import subprocess
import shutil

# Configuracao de caminhos
PROJECT_DIR = Path(__file__).parent
JOBS_DIR = PROJECT_DIR / "jobs"
TEMP_DIR = PROJECT_DIR / "temp"

# Criar diretorios necessarios
JOBS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ============================================
# TELEGRAM BOT SETUP
# ============================================
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, ConversationHandler, filters

# Token do bot - configure via variavel de ambiente
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Modo LITE - desabilita funcionalidades pesadas (dublagem, transcricao, clonagem)
BOT_LITE_MODE = os.environ.get("BOT_LITE_MODE", "0") == "1"
if BOT_LITE_MODE:
    logger.info("Modo LITE ativado - dublagem, transcricao e clonagem desabilitados")

# Estados da conversa
(
    ESCOLHER_OPERACAO,
    ENVIAR_URL,
    ENVIAR_ARQUIVO,
    SELECIONAR_IDIOMA,
    SELECIONAR_VOZ,
    SELECIONAR_MODELO,
    CONFIRMAR_PROCESSAMENTO,
    USAR_VOZ_CLONADA,
    ENVIAR_AUDIO_CLONE,
    SELECIONAR_TIPO_CORTE,
    ENVIAR_TIMESTAMPS,
) = range(12)

# ============================================
# CONFIGURACOES
# ============================================

# Idiomas suportados
IDIOMAS = {
    "pt": "Portugues",
    "en": "Ingles",
    "es": "Espanhol",
    "fr": "Frances",
    "de": "Alemao",
    "it": "Italiano",
    "ja": "Japones",
    "ko": "Coreano",
    "zh": "Chines",
    "ru": "Russo",
    "ar": "Arabe",
    "hi": "Hindi",
}

# Vozes Edge TTS disponiveis
VOZES_EDGE = {
    "pt-BR": ["Antoni", "Francisca", "Antonio"],
    "en-US": ["Aria", "Guy", "Jenny"],
    "es-ES": ["Elvira", "Dario"],
    "fr-FR": ["Denise", "Henri"],
    "de-DE": ["Katja", "Conrad"],
    "it-IT": ["Elsa", "Diego"],
    "ja-JP": ["Nanami", "Kei"],
    "ko-KR": ["SunHi", "InJoon"],
    "zh-CN": ["Xiaoxiao", "Yunxi"],
}

# Modelos de transcricao
MODELOS_WHISPER = {
    "tiny": "Mais rapido, menos preciso",
    "base": "Rapido, precisao media",
    "small": "Balanco bom",
    "medium": "Bom equilibrio",
    "large": "Mais lento, mais preciso",
    "large-v3": "Melhor precisao",
}

# Tipos de operacao
OPERACOES = {
    "download": "Download de Video",
    "transcrever": "Transcrever Video",
    "dublar": "Dublar Video",
    "cortar": "Cortar Video",
    "tts": "Texto para Fala (TTS)",
    "voice_clone": "Clonar Voz",
    "status": "Status do Sistema",
}

# ============================================
# CLASSES DE PROCESSAMENTO
# ============================================

class Processador:
    """Classe principal para processar todas as operacoes"""
    
    def __init__(self):
        self.temp_dir = TEMP_DIR
        self.jobs_dir = JOBS_DIR
        
    async def download_video(self, url: str, chat_id: int) -> str:
        """Download de video usando yt-dlp"""
        job_id = f"download_{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = self.jobs_dir / job_id
        output_dir.mkdir(exist_ok=True)
        
        output_template = str(output_dir / "%(title)s.%(ext)s")
        
        cmd = [
            "python", "baixar_v1.py",
            "--url", url,
            "--output", str(output_dir)
        ]
        
        logger.info(f"Baixando video: {url}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_DIR)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Encontrar o arquivo baixado
                arquivos = list(output_dir.glob("*"))
                if arquivos:
                    return str(arquivos[0])
                return f"Download concluido! Arquivos em: {output_dir}"
            else:
                logger.error(f"Erro no download: {stderr.decode()}")
                return f"Erro no download: {stderr.decode()}"
                
        except Exception as e:
            logger.error(f"Excecao no download: {e}")
            return f"Erro: {str(e)}"
    
    async def transcrever_video(self, arquivo: str, modelo: str, chat_id: int) -> str:
        """Transcrever video usando Whisper"""
        job_id = f"transcribe_{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = self.jobs_dir / job_id
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            "python", "transcrever_v1.py",
            "--input", arquivo,
            "--model", modelo,
            "--output", str(output_dir)
        ]
        
        logger.info(f"Transcrevendo video: {arquivo}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_DIR)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Verificar arquivos de saida
                srt_file = output_dir / "legenda.srt"
                txt_file = output_dir / "legenda.txt"
                
                if srt_file.exists():
                    return str(srt_file)
                elif txt_file.exists():
                    return str(txt_file)
                return f"Transcricao concluida! Arquivos em: {output_dir}"
            else:
                return f"Erro na transcricao: {stderr.decode()}"
                
        except Exception as e:
            return f"Erro: {str(e)}"
    
    async def dublar_video(self, arquivo: str, idioma_origem: str, idioma_destino: str, voz: str, chat_id: int) -> str:
        """Dubla o video para o idioma desejado"""
        job_id = f"dublar_{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = self.jobs_dir / job_id
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            "python", "dublar_pro.py",
            "--input", arquivo,
            "--source", idioma_origem,
            "--target", idioma_destino,
            "--voice", voz,
            "--output", str(output_dir)
        ]
        
        logger.info(f"Dublando video: {arquivo}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_DIR)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Encontrar video dublado
                arquivos = list(output_dir.glob("*_dublado.*"))
                if arquivos:
                    return str(arquivos[0])
                return f"Dublagem concluida! Arquivos em: {output_dir}"
            else:
                return f"Erro na dublagem: {stderr.decode()}"
                
        except Exception as e:
            return f"Erro: {str(e)}"
    
    async def cortar_video(self, arquivo: str, inicio: str, fim: str, chat_id: int) -> str:
        """Corta o video usando timestamps"""
        job_id = f"cortar_{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = self.jobs_dir / job_id
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            "python", "clipar_v1.py",
            "--input", arquivo,
            "--start", inicio,
            "--end", fim,
            "--output", str(output_dir)
        ]
        
        logger.info(f"Cortando video: {arquivo}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_DIR)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                arquivos = list(output_dir.glob("*.mp4"))
                if arquivos:
                    return str(arquivos[0])
                return f"Corte concluido! Arquivos em: {output_dir}"
            else:
                return f"Erro no corte: {stderr.decode()}"
                
        except Exception as e:
            return f"Erro: {str(e)}"
    
    async def gerar_tts(self, texto: str, voz: str, idioma: str, chat_id: int) -> str:
        """Gera audio TTS a partir de texto"""
        job_id = f"tts_{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = self.jobs_dir / job_id
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "tts_audio.mp3"
        
        cmd = [
            "python", "tts_direct.py",
            "--text", texto,
            "--voice", voz,
            "--lang", idioma,
            "--output", str(output_file)
        ]
        
        logger.info(f"Gerando TTS: {texto[:50]}...")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_DIR)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and output_file.exists():
                return str(output_file)
            else:
                return f"Erro no TTS: {stderr.decode()}"
                
        except Exception as e:
            return f"Erro: {str(e)}"
    
    async def clonar_voz(self, audio_ref: str, texto: str, chat_id: int) -> str:
        """Clona voz e gera audio com o texto"""
        job_id = f"clone_{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = self.jobs_dir / job_id
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "voz_clonada.mp3"
        
        # Usar Chatterbox TTS para clonagem
        cmd = [
            "python", "-c",
            f"""
import asyncio
from chatterbox_tts import ChatterboxTTS

async def main():
    tts = ChatterboxTTS()
    await tts.generate(
        text="{texto}",
        audio_ref_path="{audio_ref}",
        output_path="{output_file}"
    )

asyncio.run(main())
"""
        ]
        
        logger.info(f"Clonando voz com referencia: {audio_ref}")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_DIR)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 and output_file.exists():
                return str(output_file)
            else:
                return f"Erro na clonagem: {stderr.decode()}"
                
        except Exception as e:
            return f"Erro: {str(e)}"
    
    def get_status(self) -> str:
        """Retorna status do sistema"""
        try:
            import psutil
            
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = f"""
Status do Sistema inemaVOX

CPU: {cpu}%
Memoria: {memory.percent}%
Disco: {disk.percent}%

Jobs: {len(list(self.jobs_dir.glob('*')))}
Temp: {len(list(self.temp_dir.glob('*')))}
"""
            return status
        except:
            return "Status nao disponivel"


# Instancia global do processador
processador = Processador()

# Armazenamento de dados do usuario
dados_usuario = {}


# ============================================
# HANDLERS DO BOT
# ============================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler do comando /start"""
    modo_texto = "\n(Modo LITE - dublagem desabilitada)" if BOT_LITE_MODE else ""
    await update.message.reply_text(
        f"inemaVOX Bot{modo_texto}\n\n"
        "Bem-vindo ao bot de dublagem e processamento de video com IA!\n\n"
        "Selecione uma operacao:",
        reply_markup=menu_principal()
    )
    return ESCOLHER_OPERACAO


def menu_principal():
    """Cria o menu principal de operacoes"""
    keyboard = [
        [InlineKeyboardButton("Download", callback_data="op_download")],
        [InlineKeyboardButton("Cortar", callback_data="op_cortar")],
        [InlineKeyboardButton("TTS", callback_data="op_tts")],
    ]
    
    # Funcionalidades pesadas - apenas se NAO estiver em modo LITE
    if not BOT_LITE_MODE:
        keyboard.insert(1, [InlineKeyboardButton("Transcrever", callback_data="op_transcrever")])
        keyboard.insert(2, [InlineKeyboardButton("Dublar", callback_data="op_dublar")])
        keyboard.append([InlineKeyboardButton("Clonar Voz", callback_data="op_voice_clone")])
    
    keyboard.append([InlineKeyboardButton("Status", callback_data="op_status")])
    return InlineKeyboardMarkup(keyboard)


async def menu_operacoes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra o menu de operacoes"""
    query = update.callback_query
    await query.answer()
    
    await query.edit_message_text(
        "Selecione a Operacao\n\n"
        "Escolha o que deseja fazer:",
        reply_markup=menu_principal()
    )
    return ESCOLHER_OPERACAO


# ============================================
# HANDLERS DE CALLBACK
# ============================================

async def callback_download(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para operacao de download"""
    query = update.callback_query
    await query.answer()
    
    dados_usuario[query.from_user.id] = {"operacao": "download"}
    
    await query.edit_message_text(
        "Download de Video\n\n"
        "Envie a URL do video (YouTube, TikTok, Instagram, etc.):"
    )
    return ENVIAR_URL


async def callback_transcrever(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para transcricao"""
    query = update.callback_query
    await query.answer()
    
    # Verificar modo LITE
    if BOT_LITE_MODE:
        await query.edit_message_text(
            "Transcrever Video\n\n"
            "Desculpe, esta funcionalidade esta desabilitada no modo LITE.\n"
            "Motivo: requer modelos de IA pesados (Whisper, PyTorch).\n\n"
            "Use o modo completo para transcrever videos."
        )
        return ConversationHandler.END
    
    dados_usuario[query.from_user.id] = {"operacao": "transcrever"}
    
    # Menu de modelos
    keyboard = []
    for modelo, desc in MODELOS_WHISPER.items():
        keyboard.append([InlineKeyboardButton(f"{modelo.upper()} - {desc}", callback_data=f"model_{modelo}")])
    
    await query.edit_message_text(
        "Transcrever Video\n\n"
        "Selecione o modelo de transcricao:\n"
        "- tiny/base: Mais rapido\n"
        "- small/medium: Equilibrio\n"
        "- large/large-v3: Mais preciso",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return SELECIONAR_MODELO


async def callback_dublar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para dublagem"""
    query = update.callback_query
    await query.answer()
    
    # Verificar modo LITE
    if BOT_LITE_MODE:
        await query.edit_message_text(
            "Dublar Video\n\n"
            "Desculpe, esta funcionalidade esta desabilitada no modo LITE.\n"
            "Motivo: requer modelos de IA pesados (Whisper, M2M100, PyTorch).\n\n"
            "Use o modo completo para dublar videos."
        )
        return ConversationHandler.END
    
    dados_usuario[query.from_user.id] = {"operacao": "dublar"}
    
    keyboard = []
    for code, nome in IDIOMAS.items():
        keyboard.append([InlineKeyboardButton(nome, callback_data=f"source_{code}")])
    
    await query.edit_message_text(
        "Dublar Video - Passo 1/3\n\n"
        "Selecione o idioma do video original:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return SELECIONAR_IDIOMA


async def callback_cortar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para corte de video"""
    query = update.callback_query
    await query.answer()
    
    dados_usuario[query.from_user.id] = {"operacao": "cortar"}
    
    keyboard = [
        [InlineKeyboardButton("Por Timestamps", callback_data="corte_timestamps")],
        [InlineKeyboardButton("Detectar Momentos Virais", callback_data="corte_auto")],
    ]
    
    await query.edit_message_text(
        "Cortar Video\n\n"
        "Como deseja cortar o video?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return SELECIONAR_TIPO_CORTE


async def callback_tts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para TTS"""
    query = update.callback_query
    await query.answer()
    
    dados_usuario[query.from_user.id] = {"operacao": "tts"}
    
    keyboard = []
    for code, nome in IDIOMAS.items():
        keyboard.append([InlineKeyboardButton(nome, callback_data=f"tts_lang_{code}")])
    
    await query.edit_message_text(
        "Texto para Fala (TTS)\n\n"
        "Selecione o idioma:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return SELECIONAR_IDIOMA


async def callback_voice_clone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para clonagem de voz"""
    query = update.callback_query
    await query.answer()
    
    # Verificar modo LITE
    if BOT_LITE_MODE:
        await query.edit_message_text(
            "Clonar Voz\n\n"
            "Desculpe, esta funcionalidade esta desabilitada no modo LITE.\n"
            "Motivo: requer modelos de IA pesados (Bark, PyTorch).\n\n"
            "Use o modo completo para clonar vozes."
        )
        return ConversationHandler.END
    
    dados_usuario[query.from_user.id] = {"operacao": "voice_clone"}
    
    await query.edit_message_text(
        "Clonar Voz\n\n"
        "1. Primeiro, envie um audio de referencia (5-30 segundos)\n"
        "2. Depois, envie o texto que deseja gerar com essa voz\n\n"
        "Envie o audio de referencia agora:"
    )
    return ENVIAR_AUDIO_CLONE


async def callback_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para status do sistema"""
    query = update.callback_query
    await query.answer()
    
    status = processador.get_status()
    keyboard = [[InlineKeyboardButton("Atualizar", callback_data="op_status")]]
    
    await query.edit_message_text(
        status,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


# ============================================
# HANDLERS DE SELECAO
# ============================================

async def selecionar_modelo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa selecao de modelo"""
    query = update.callback_query
    await query.answer()
    
    modelo = query.data.replace("model_", "")
    user_id = query.from_user.id
    
    if user_id in dados_usuario:
        dados_usuario[user_id]["modelo"] = modelo
    
    await query.edit_message_text(
        "Transcrever Video\n\n"
        "Agora envie o arquivo de video ou URL:"
    )
    return ENVIAR_ARQUIVO


async def selecionar_idioma_origem(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa selecao de idioma de origem"""
    query = update.callback_query
    await query.answer()
    
    idioma = query.data.replace("source_", "")
    user_id = query.from_user.id
    
    if user_id in dados_usuario:
        dados_usuario[user_id]["idioma_origem"] = idioma
    
    # Mostrar idiomas de destino
    keyboard = []
    for code, nome in IDIOMAS.items():
        if code != idioma:
            keyboard.append([InlineKeyboardButton(nome, callback_data=f"target_{code}")])
    
    await query.edit_message_text(
        f"Dublar Video - Passo 2/3\n\n"
        f"Idioma original: {IDIOMAS.get(idioma, idioma)}\n\n"
        "Selecione o idioma de destino:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return SELECIONAR_IDIOMA


async def selecionar_idioma_destino(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa selecao de idioma de destino"""
    query = update.callback_query
    await query.answer()
    
    idioma = query.data.replace("target_", "")
    user_id = query.from_user.id
    
    if user_id in dados_usuario:
        dados_usuario[user_id]["idioma_destino"] = idioma
        idioma_origem = dados_usuario[user_id].get("idioma_origem", "en")
        
        # Mostrar vozes disponiveis
        lang_code = idioma
        vozes = VOZES_EDGE.get(lang_code, VOZES_EDGE["en-US"])
        
        keyboard = []
        for voz in vozes:
            keyboard.append([InlineKeyboardButton(voz, callback_data=f"voice_{voz}")])
        
        await query.edit_message_text(
            f"Dublar Video - Passo 3/3\n\n"
            f"Idioma destino: {IDIOMAS.get(idioma, idioma)}\n\n"
            "Selecione a voz:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    return SELECIONAR_VOZ


async def selecionar_voz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa selecao de voz"""
    query = update.callback_query
    await query.answer()
    
    voz = query.data.replace("voice_", "")
    user_id = query.from_user.id
    
    if user_id in dados_usuario:
        dados_usuario[user_id]["voz"] = voz
    
    await query.edit_message_text(
        "Dublar Video\n\n"
        "Agora envie o arquivo de video ou URL para dublar:"
    )
    return ENVIAR_ARQUIVO


async def selecionar_idioma_tts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa selecao de idioma para TTS"""
    query = update.callback_query
    await query.answer()
    
    idioma = query.data.replace("tts_lang_", "")
    user_id = query.from_user.id
    
    if user_id in dados_usuario:
        dados_usuario[user_id]["idioma"] = idioma
        
        # Mostrar vozes disponiveis
        vozes = VOZES_EDGE.get(idioma, VOZES_EDGE["en-US"])
        
        keyboard = []
        for voz in vozes:
            keyboard.append([InlineKeyboardButton(voz, callback_data=f"tts_voice_{voz}")])
        
        await query.edit_message_text(
            f"TTS - {IDIOMAS.get(idioma, idioma)}\n\n"
            "Selecione a voz:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    return SELECIONAR_VOZ


async def selecionar_voz_tts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa selecao de voz para TTS"""
    query = update.callback_query
    await query.answer()
    
    voz = query.data.replace("tts_voice_", "")
    user_id = query.from_user.id
    
    if user_id in dados_usuario:
        dados_usuario[user_id]["voz_tts"] = voz
    
    await query.edit_message_text(
        "Texto para Fala\n\n"
        "Agora envie o texto que deseja converter em audio:"
    )
    return ENVIAR_ARQUIVO


# ============================================
# HANDLERS DE MENSAGENS
# ============================================

async def processar_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa URL enviada pelo usuario"""
    user_id = update.message.from_user.id
    url = update.message.text
    
    if user_id not in dados_usuario:
        await update.message.reply_text("Use /start para comecar")
        return ConversationHandler.END
    
    operacao = dados_usuario[user_id].get("operacao")
    
    await update.message.reply_text("Processando... Isso pode levar alguns minutos.")
    
    try:
        if operacao == "download":
            resultado = await processador.download_video(url, user_id)
            
            if resultado and resultado.endswith(".mp4"):
                await update.message.reply_video(video=open(resultado, "rb"))
            else:
                await update.message.reply_text(f"OK: {resultado}")
        
        elif operacao == "transcrever":
            # Primeiro faz download
            arquivo_video = await processador.download_video(url, user_id)
            modelo = dados_usuario[user_id].get("modelo", "base")
            
            resultado = await processador.transcrever_video(arquivo_video, modelo, user_id)
            
            if resultado and resultado.endswith(".srt"):
                await update.message.reply_document(document=open(resultado, "rb"), caption="Legenda SRT")
            elif resultado and resultado.endswith(".txt"):
                await update.message.reply_document(document=open(resultado, "rb"), caption="Transcricao")
            else:
                await update.message.reply_text(f"OK: {resultado}")
        
        elif operacao == "dublar":
            arquivo_video = await processador.download_video(url, user_id)
            idioma_origem = dados_usuario[user_id].get("idioma_origem", "en")
            idioma_destino = dados_usuario[user_id].get("idioma_destino", "pt")
            voz = dados_usuario[user_id].get("voz", "Antoni")
            
            resultado = await processador.dublar_video(
                arquivo_video, idioma_origem, idioma_destino, voz, user_id
            )
            
            if resultado and resultado.endswith(".mp4"):
                await update.message.reply_video(video=open(resultado, "rb"))
            else:
                await update.message.reply_text(f"OK: {resultado}")
        
        else:
            await update.message.reply_text("Operacao nao reconhecida. Use /start")
    
    except Exception as e:
        logger.error(f"Erro ao processar URL: {e}")
        await update.message.reply_text(f"Erro: {str(e)}")
    
    return ConversationHandler.END


async def processar_arquivo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa arquivo enviado pelo usuario"""
    user_id = update.message.from_user.id
    
    if user_id not in dados_usuario:
        await update.message.reply_text("Use /start para comecar")
        return ConversationHandler.END
    
    operacao = dados_usuario[user_id].get("operacao")
    
    # Baixar arquivo
    await update.message.reply_text("Recebendo arquivo...")
    
    arquivo = await context.bot.get_file(update.message.document.file_id)
    extensao = update.message.document.file_name.split('.')[-1]
    
    temp_file = TEMP_DIR / f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extensao}"
    await arquivo.download_to_drive(temp_file)
    
    await update.message.reply_text("Processando... Isso pode levar alguns minutos.")
    
    try:
        if operacao == "transcrever":
            modelo = dados_usuario[user_id].get("modelo", "base")
            resultado = await processador.transcrever_video(str(temp_file), modelo, user_id)
            
            if resultado and resultado.endswith(".srt"):
                await update.message.reply_document(document=open(resultado, "rb"), caption="Legenda SRT")
            elif resultado and resultado.endswith(".txt"):
                await update.message.reply_document(document=open(resultado, "rb"), caption="Transcricao")
            else:
                await update.message.reply_text(f"OK: {resultado}")
        
        elif operacao == "dublar":
            idioma_origem = dados_usuario[user_id].get("idioma_origem", "en")
            idioma_destino = dados_usuario[user_id].get("idioma_destino", "pt")
            voz = dados_usuario[user_id].get("voz", "Antoni")
            
            resultado = await processador.dublar_video(
                str(temp_file), idioma_origem, idioma_destino, voz, user_id
            )
            
            if resultado and resultado.endswith(".mp4"):
                await update.message.reply_video(video=open(resultado, "rb"))
            else:
                await update.message.reply_text(f"OK: {resultado}")
        
        elif operacao == "tts":
            # TTS com texto enviado
            pass  # tratado em processar_texto
        
        elif operacao == "voice_clone":
            # Armazenar referencia de audio
            dados_usuario[user_id]["audio_ref"] = str(temp_file)
            await update.message.reply_text(
                "Audio de referencia recebido!\n\n"
                "Agora envie o texto que deseja gerar com essa voz:"
            )
            return ENVIAR_ARQUIVO
        
        else:
            await update.message.reply_text("Operacao nao reconhecida")
    
    except Exception as e:
        logger.error(f"Erro ao processar arquivo: {e}")
        await update.message.reply_text(f"Erro: {str(e)}")
    
    return ConversationHandler.END


async def processar_texto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa texto para TTS"""
    user_id = update.message.from_user.id
    texto = update.message.text
    
    if user_id not in dados_usuario:
        await update.message.reply_text("Use /start para comecar")
        return ConversationHandler.END
    
    operacao = dados_usuario[user_id].get("operacao")
    
    await update.message.reply_text("Gerando audio... Isso pode levar alguns minutos.")
    
    try:
        if operacao == "tts":
            voz = dados_usuario[user_id].get("voz_tts", "Antoni")
            idioma = dados_usuario[user_id].get("idioma", "pt-BR")
            
            resultado = await processador.gerar_tts(texto, voz, idioma, user_id)
            
            if resultado and resultado.endswith(".mp3"):
                await update.message.reply_audio(audio=open(resultado, "rb"))
            else:
                await update.message.reply_text(f"OK: {resultado}")
        
        elif operacao == "voice_clone":
            audio_ref = dados_usuario[user_id].get("audio_ref")
            if not audio_ref:
                await update.message.reply_text("Erro: audio de referencia nao encontrado")
                return ConversationHandler.END
            
            resultado = await processador.clonar_voz(audio_ref, texto, user_id)
            
            if resultado and resultado.endswith(".mp3"):
                await update.message.reply_audio(audio=open(resultado, "rb"))
            else:
                await update.message.reply_text(f"OK: {resultado}")
        
        else:
            await update.message.reply_text("Operacao nao reconhecida")
    
    except Exception as e:
        logger.error(f"Erro ao processar texto: {e}")
        await update.message.reply_text(f"Erro: {str(e)}")
    
    return ConversationHandler.END


async def processar_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processa audio para clonagem de voz"""
    user_id = update.message.from_user.id
    
    if user_id not in dados_usuario:
        await update.message.reply_text("Use /start para comecar")
        return ConversationHandler.END
    
    operacao = dados_usuario[user_id].get("operacao")
    
    if operacao != "voice_clone":
        await update.message.reply_text("Envie um audio para clonagem de voz")
        return ConversationHandler.END
    
    # Baixar audio
    await update.message.reply_text("Recebendo audio de referencia...")
    
    arquivo = await context.bot.get_file(update.message.audio.file_id)
    extensao = "mp3" if update.message.audio.mime_type == "audio/mpeg" else "wav"
    
    temp_file = TEMP_DIR / f"ref_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extensao}"
    await arquivo.download_to_drive(temp_file)
    
    dados_usuario[user_id]["audio_ref"] = str(temp_file)
    
    await update.message.reply_text(
        "Audio de referencia recebido!\n\n"
        "Agora envie o texto que deseja gerar com essa voz clonada:"
    )
    
    return ENVIAR_ARQUIVO


# ============================================
# MAIN
# ============================================

def main():
    """Inicia o bot"""
    if not BOT_TOKEN:
        logger.error("ERRO: Defina a variavel de ambiente TELEGRAM_BOT_TOKEN")
        print("=" * 50)
        print("ERRO: Configure o token do bot!")
        print("Execute: set TELEGRAM_BOT_TOKEN=seu_token_aqui")
        print("=" * 50)
        sys.exit(1)
    
    logger.info("Iniciando inemaVOX Bot...")
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Conversation handler para o menu principal
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            ESCOLHER_OPERACAO: [
                CallbackQueryHandler(callback_download, pattern="^op_download$"),
                CallbackQueryHandler(callback_transcrever, pattern="^op_transcrever$"),
                CallbackQueryHandler(callback_dublar, pattern="^op_dublar$"),
                CallbackQueryHandler(callback_cortar, pattern="^op_cortar$"),
                CallbackQueryHandler(callback_tts, pattern="^op_tts$"),
                CallbackQueryHandler(callback_voice_clone, pattern="^op_voice_clone$"),
                CallbackQueryHandler(callback_status, pattern="^op_status$"),
            ],
            ENVIAR_URL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, processar_url),
            ],
            ENVIAR_ARQUIVO: [
                MessageHandler(filters.Document.ALL, processar_arquivo),
                MessageHandler(filters.AUDIO, processar_audio),
                MessageHandler(filters.TEXT & ~filters.COMMAND, processar_texto),
            ],
            SELECIONAR_IDIOMA: [
                CallbackQueryHandler(selecionar_idioma_origem, pattern="^source_"),
                CallbackQueryHandler(selecionar_idioma_destino, pattern="^target_"),
                CallbackQueryHandler(selecionar_idioma_tts, pattern="^tts_lang_"),
            ],
            SELECIONAR_VOZ: [
                CallbackQueryHandler(selecionar_voz, pattern="^voice_"),
                CallbackQueryHandler(selecionar_voz_tts, pattern="^tts_voice_"),
            ],
            SELECIONAR_MODELO: [
                CallbackQueryHandler(selecionar_modelo, pattern="^model_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, processar_url),
            ],
            SELECIONAR_TIPO_CORTE: [
                CallbackQueryHandler(callback_cortar, pattern="^corte_"),
            ],
        },
        fallbacks=[CommandHandler("start", start_command)],
    )
    
    app.add_handler(conv_handler)
    
    # Comandos simples
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("status", lambda u, c: u.message.reply_text(processador.get_status())))
    
    logger.info("Bot iniciado! Pressione Ctrl+C para parar.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
