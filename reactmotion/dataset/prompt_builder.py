# reactmotion/dataset/prompt_builder.py
def build_prompt(
    speaker_transcription: str,
    speaker_audio: str,
    speaker_emotion: str,
    use_transcription: bool = True,
    use_audio: bool = False,
    use_emotion: bool = True,
) -> str:
    t = (speaker_transcription or "").strip()
    a = (speaker_audio or "").strip()
    e = (speaker_emotion or "").strip()

    lines = []
    lines.append("You are modeling a speaker-listener dyadic interaction.\n\n")
    lines.append("Input:\n")
    lines.append(f"- SPEAKER_TRANSCRIPTION: {t if use_transcription else ''}\n")
    lines.append(f"- SPEAKER_AUDIO: {a if use_audio else ''}\n")
    if use_emotion and e:
        lines.append(f"- SPEAKER_EMOTION: <Emotion> {e} </Emotion>\n")
    lines.append("\nOutput:\n")
    lines.append("Return ONLY a sequence of listener motion tokens in the exact format:\n")
    lines.append("<Motion Tokens> <Motion Token i> ... </Motion Tokens>\n")
    lines.append("Do NOT output any other words.\n")
    return "".join(lines).strip()
