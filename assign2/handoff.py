import streamlit as st
from dotenv import load_dotenv
import os
import asyncio


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig


load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

# ✅ Setup Gemini Client (Google Gemini using OpenAI-compatible wrapper)
client = AsyncOpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",  # Gemini OpenAI-compatible endpoint
)

# ✅ Define the Gemini model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",  # or gemini-1.5-pro, or gemini-2.0-pro if enabled
    openai_client=client,
)

# ✅ Runner Configuration
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True,
)

# ✅ Agent 1: Mood Detector
mood_detector = Agent(
    name="Mood Detector",
    model=model,
    instructions=(
        "You are a mood detection expert. Based on the user's message, "
        "return ONLY the mood in one lowercase word: happy, sad, stressed, excited, angry, etc. "
        "Do NOT explain. Just return the one word."
    ),
)

# ✅ Agent 2: Activity Suggester
activity_suggester = Agent(
    name="Activity Suggester",
    model=model,
    instructions=(
        "You are a comforting assistant. If the user is sad or stressed, suggest an activity like walking, "
        "meditating, listening to music, or talking to a friend. Keep it short and supportive. "
        "If mood is not sad or stressed, say something positive like 'You're doing great!'"
    ),
)

# ✅ Streamlit UI
st.set_page_config(page_title="Mood Analyzer with Gemini", layout="centered")
st.title("🧠 Mood Analyzer with Gemini 🔮")

user_input = st.text_input("How are you feeling today?", placeholder="e.g., I'm feeling tired and worried")

if st.button("Analyze Mood"):
    if not user_input.strip():
        st.warning("Please describe how you're feeling.")
    else:
        with st.spinner("Detecting your mood..."):
            try:
                # ▶️ Agent 1: Detect mood
                mood_result = Runner.run_sync(mood_detector, user_input, run_config=config)
                mood = mood_result.final_output.strip().lower()

                st.success(f"🧠 Detected Mood: **{mood}**")

                # ▶️ Agent 2: If sad or stressed, suggest activity
                if mood in ["sad", "stressed"]:
                    activity_result = Runner.run_sync(activity_suggester, mood, run_config=config)
                    st.info(f"💡 Suggested Activity: {activity_result.final_output}")
                else:
                    st.balloons()
                    st.info("🎉 You're feeling good! Keep it up! 🌞")

            except Exception as e:
                st.error(f"❌ Error: {e}")
