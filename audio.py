import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

def transcribe_audio(client: OpenAI, audio_file: str):
    logger.info(f"Starting transcription of file: {audio_file}")
    try:
        with open(audio_file, "rb") as file:
            logger.info("File opened successfully")
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=file
            )
        logger.info("Transcription completed successfully")
        return transcript.text
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        raise