import asyncio
import os
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins import openai, deepgram
from livekit import rtc
from livekit.rtc import TrackKind
from graph import app as langgraph_app

load_dotenv()

async def entrypoint(ctx: JobContext):
    # --- 1. SETUP OUTPUT ---
    source = rtc.AudioSource(24000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("bot_voice", source)
    
    # --- 2. INIT AI ---
    ai_ready_event = asyncio.Event()
    stt = None
    tts = None
    
    async def initialize_ai():
        nonlocal stt, tts
        print("   ⏳ Initializing AI Modules...")
        try:
            # FIX: 'endpointing=2000' tells Deepgram to wait 2 seconds of silence
            # before submitting the sentence. Ideal for non-native speakers.
            stt = deepgram.STT(
                interim_results=True, 
                endpointing_ms=2000,
                smart_format=True
            )
            tts = openai.TTS(model="tts-1", voice="nova")
            ai_ready_event.set()
            print("   ✅ AI Modules Ready!")
        except Exception as e:
            print(f"   ❌ Error Loading AI: {e}")

    asyncio.create_task(initialize_ai())

    # --- 3. PIPELINE LOGIC ---
    current_answer_task = None

    async def run_langgraph_pipeline(text, participant_id):
        if not ai_ready_event.is_set(): await ai_ready_event.wait()

        try:
            async def text_iterator():
                inputs = {
                    "question": text, 
                    "user_id": participant_id, 
                    "destination": "conversational_node",
                    "validation": None, 
                    "generated_answer": None
                }
                async for event in langgraph_app.astream_events(inputs, version="v1"):
                     if event["event"] == "on_chat_model_stream":
                        node = event["metadata"].get("langgraph_node")
                        if node in ["generate_node", "conversational_node"]:
                            chunk = event["data"]["chunk"]
                            if chunk.content:
                                yield chunk.content

            buffer = ""
            async for char in text_iterator():
                buffer += char
                if len(buffer) > 5 and buffer[-1] in [".", "!", "?", "\n", ",", ";"]:
                    async for audio_chunk in tts.synthesize(text=buffer):
                        await source.capture_frame(audio_chunk.frame)
                    buffer = "" 
            
            if buffer.strip():
                async for audio_chunk in tts.synthesize(text=buffer):
                    await source.capture_frame(audio_chunk.frame)

        except Exception as e:
            print(f"   ❌ Error in pipeline: {e}")

    async def process_track(track, participant):
        nonlocal current_answer_task
        if not ai_ready_event.is_set(): await ai_ready_event.wait()
        
        print(f"   👂 LISTENING to {participant.identity}")
        
        audio_stream = rtc.AudioStream(track)
        push_task = None 

        while True:
            try:
                # Cleanup old task on retry
                if push_task and not push_task.done():
                    push_task.cancel()
                    
                stt_stream = stt.stream()
                
                async def push_audio():
                    try:
                        async for event in audio_stream:
                            stt_stream.push_frame(event.frame)
                    except asyncio.CancelledError:
                        pass
                    finally:
                        stt_stream.end_input()
                
                push_task = asyncio.create_task(push_audio())

                async for speech_event in stt_stream:
                    # Logic: Interrupt on WORDS (Interim), Answer on SENTENCE (Final)
                    
                    if speech_event.type == "interim_transcript":
                        text = speech_event.alternatives[0].text.strip()
                        if text:
                            if current_answer_task and not current_answer_task.done():
                                print(f"   ⚡ Interrupting (Heard: '{text}')...")
                                current_answer_task.cancel()

                    if speech_event.type == "final_transcript":
                        user_text = speech_event.alternatives[0].text.strip()
                        # Only answer if text is substantial
                        if user_text and len(user_text) > 1:
                            print(f"   👤 User Said: {user_text}")
                            if current_answer_task and not current_answer_task.done():
                                current_answer_task.cancel()
                            current_answer_task = asyncio.create_task(run_langgraph_pipeline(user_text, participant.identity))
                
                break 

            except Exception as e:
                print(f"   ⚠️ STT Glitch: {e}. Retrying...")
                await asyncio.sleep(1) 
                continue
        
        if push_task and not push_task.done():
            push_task.cancel()

    # --- 4. EVENT LISTENERS ---
    @ctx.room.on("track_subscribed")
    def on_track(track, publication, participant):
        if track.kind == TrackKind.KIND_AUDIO:
            print(f"   🎤 Audio Track Subscribed! Attaching AI...")
            asyncio.create_task(process_track(track, participant))

    # --- 5. CONNECT ---
    print("🔌 Connecting to Room...")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await ctx.room.local_participant.publish_track(track)
    print("   ✅ Connected & Voice Published!")

    for p in ctx.room.remote_participants.values():
        for t_pub in p.track_publications.values():
            if t_pub.track and t_pub.track.kind == TrackKind.KIND_AUDIO:
                asyncio.create_task(process_track(t_pub.track, p))

    await asyncio.Event().wait()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))