"""
Learnify Backend v2 — AI-powered contextual learning
Stack: FastAPI + OpenAI
Themes: Roblox, Football, Fortnite, JJK, Naruto
Run: python main.py
Docs: http://localhost:8000/docs
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS — these are packages Python needs to run the app
# ─────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
import openai
import os
import json
import hashlib
import time

# ─────────────────────────────────────────────────────────────
# APP SETUP 
# Creates the FastAPI application and allows the frontend
# (running on a different port) to talk to this backend.
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Learnify API v2",
    description="Learn any subject in your language",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # In production: change to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# OPENAI CLIENT SETUP
# Reads your API key from the environment variable you set.
# NEVER paste your actual key into this file.
# In terminal run: export OPENAI_API_KEY="sk-your-key-here"
# ─────────────────────────────────────────────────────────────
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ─────────────────────────────────────────────────────────────
# DIFFICULTY LEVELS
# These are the options the student can pick from.
# The value in quotes is what gets sent to the AI in the prompt.
# ─────────────────────────────────────────────────────────────
class DifficultyLevel(str, Enum):
    ks2        = "ks2"
    ks3        = "ks3"
    gcse       = "gcse"
    a_level    = "a_level"
    university = "university"


# ─────────────────────────────────────────────────────────────
# THEME PROFILES
# This is the most important dictionary in the whole app.
# Each theme tells the AI EXACTLY what vocabulary and references
# to use when reframing the lesson. The richer the profile,
# the better and more immersive the output will be.
#
# HOW TO ADD A NEW THEME:
# Copy one block, change the key name (e.g. "minecraft"),
# update the description and example_terms, save the file,
# restart the server. That's it.
# ─────────────────────────────────────────────────────────────
THEME_PROFILES = {

    "roblox": {
        "description": (
            "Use Roblox universe references throughout. Reference Robux (the currency), "
            "game passes, obby (obstacle courses), simulator games, tycoon games, roleplay servers, "
            "avatar customisation, the Roblox catalog, scripting in Lua, Roblox Studio (the game builder), "
            "popular games like Adopt Me, Blox Fruits, Brookhaven, Jailbreak, Tower of Hell. "
            "Reference the idea of developers building experiences, earning Robux, and levelling up. "
            "Use the language a Roblox player would naturally use."
        ),
        "example_terms": "Robux, obby, tycoon, game pass, Roblox Studio, Lua script, simulator, Adopt Me, Blox Fruits, Tower of Hell, avatar, catalog"
    },

    "football": {
        "description": (
            "Use football (soccer) analogies throughout. Reference player positions (striker, midfielder, "
            "goalkeeper, centre-back, winger, full-back), tactical concepts (pressing, offside trap, "
            "formations like 4-3-3 or 4-4-2, counter-attack, high line, set pieces, clean sheets), "
            "match situations (injury time, VAR, penalty shootout, relegation battle, title race), "
            "and famous clubs and competitions (Premier League, Champions League, World Cup). "
            "Use the language a football fan would speak naturally."
        ),
        "example_terms": "formation, pressing, offside trap, clean sheet, transfer window, matchday, dribbling, assist, VAR, Champions League, penalty, relegation"
    },

    "fortnite": {
        "description": (
            "Use Fortnite battle royale references throughout. Reference the Storm circle closing in, "
            "building and editing structures under pressure, getting a Victory Royale, named drop locations "
            "(Tilted Towers, Pleasant Park, Loot Lake, The Agency), weapon rarities (grey common to gold legendary), "
            "shield potions, harvesting materials (wood, brick, metal), the Battle Pass, box fighting, "
            "third-partying, rotating to the zone, and famous skins. "
            "Use the slang and energy of a Fortnite player."
        ),
        "example_terms": "Storm, Victory Royale, building, box fighting, drop zone, shield, loot, Battle Pass, double pump, zone, rotate, third-party, material, edit"
    },

    "jujutsu_kaisen": {
        "description": (
            "Use Jujutsu Kaisen universe references throughout. Reference Cursed Energy and how it flows, "
            "Cursed Techniques (innate and learned), Domain Expansion (a perfect closed environment), "
            "the difference between Sorcerers and Cursed Spirits, Jujutsu High Tokyo and Kyoto, "
            "sorcerer grades (Grade 4 weakest up to Special Grade strongest), "
            "key characters: Yuji Itadori, Megumi Fushiguro, Nobara Kugisaki, Satoru Gojo (Infinity, Six Eyes), "
            "Ryomen Sukuna, Suguru Geto. Reference Black Flash, Simple Domain, Binding Vows, Heavenly Restriction. "
            "Use the tone of the manga/anime — serious, intense, with moments of humour."
        ),
        "example_terms": "Cursed Energy, Domain Expansion, Cursed Technique, Jujutsu High, Special Grade, Black Flash, Infinity, Six Eyes, Sukuna, binding vow, sorcerer, cursed spirit"
    },

    "naruto": {
        "description": (
            "Use Naruto universe references throughout. Reference Chakra (the energy that powers everything), "
            "chakra natures (Fire, Water, Earth, Wind, Lightning), hand seals, jutsu types "
            "(ninjutsu, genjutsu, taijutsu), the ninja rank system (Academy Student, Genin, Chunin, Jonin, Kage), "
            "Hidden Villages (Konoha, Sand, Mist, Cloud, Stone), the Akatsuki, Tailed Beasts, "
            "key characters: Naruto, Sasuke, Sakura, Kakashi, Itachi, Madara, Minato. "
            "Reference concepts like the Will of Fire, never giving up, and exceeding your limits. "
            "Use the tone of the anime — motivational, epic, with rivalry and growth."
        ),
        "example_terms": "chakra, jutsu, ninjutsu, hand seals, Genin, Jonin, Hokage, Hidden Leaf, Sharingan, Sage Mode, Tailed Beast, Akatsuki, Will of Fire"
    }
}


# ─────────────────────────────────────────────────────────────
# REQUEST & RESPONSE MODELS
# Pydantic models define the exact shape of data going IN
# and coming OUT of the API. FastAPI validates this automatically
# so if a required field is missing, it returns a helpful error.
# ─────────────────────────────────────────────────────────────

class LearnRequest(BaseModel):
    topic: str                        # What the student wants to learn
    subject: str                      # The broader subject area
    difficulty: DifficultyLevel       # Which level (ks2 through university)
    theme: str                        # Which theme to use (e.g. "naruto")
    include_quiz: bool = True         # Whether to generate quiz questions
    quiz_question_count: int = 5      # How many questions (capped at 10)


class QuizQuestion(BaseModel):
    question: str
    options: list[str]                # Always 4 options
    correct_answer: str
    explanation: str                  # Explains why, still in theme language


class LearnResponse(BaseModel):
    topic: str
    subject: str
    difficulty: str
    theme: str
    explanation: str
    key_points: list[str]
    quiz: list[QuizQuestion] | None
    estimated_read_time_minutes: int


# ─────────────────────────────────────────────────────────────
# PROMPT ENGINEERING
# This is the brain of the whole app.
# Two functions: one sets the AI's overall personality and rules,
# the other builds the specific request for each lesson.
# ─────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    """
    Runs on every single request. Sets who the AI is and its non-negotiable rules.
    Think of this as the AI's job description and rulebook.
    """
    return """You are Learnify, the world's best educational tutor. 
Your unique ability is explaining any academic concept with complete accuracy 
but framed entirely in themes, analogies, and language from worlds the student already loves.

YOUR RULES — NEVER BREAK THESE:

RULE 1 - ACCURACY FIRST.
The educational content must be 100% correct at the specified level.
A student will use this to study and revise. Getting it wrong is not an option.
Never sacrifice factual correctness to make a theme analogy work better.

RULE 2 - FULL THEME IMMERSION.
Once you pick up the theme, never drop it. The entire explanation should feel like
it was written by someone who lives and breathes that theme. Don't just sprinkle
theme words at the start and end — weave them throughout every paragraph.

RULE 3 - MATCH THE LEVEL EXACTLY.
KS2 must be simple, short, fun, and use zero jargon.
University must be rigorous, reference theory, and assume high prior knowledge.
Don't pitch KS2 content at A-Level or vice versa.

RULE 4 - STRUCTURED JSON ONLY.
Respond ONLY with valid JSON matching the exact format requested.
No markdown, no code blocks, no text before or after the JSON.
"""


def build_user_prompt(request: LearnRequest) -> str:
    """
    Builds the specific lesson request dynamically.
    Injects: topic + subject + difficulty + theme profile + quiz settings.
    This is what changes with every request.
    """

    # Look up the theme profile, or build a generic instruction for custom themes
    theme_key = request.theme.lower().replace(" ", "_")
    if theme_key in THEME_PROFILES:
        profile = THEME_PROFILES[theme_key]
        theme_block = f"""
THEME PROFILE — {request.theme.upper()}:
{profile['description']}

Key vocabulary to weave in naturally: {profile['example_terms']}
"""
    else:
        # Handles any custom theme the user types that isn't in our profiles
        theme_block = f"""
THEME: {request.theme}
Use your knowledge of {request.theme} to build rich, specific analogies.
Reference actual characters, concepts, locations, and terminology from {request.theme}.
Make the student feel like they're reading about {request.theme}, not just studying.
"""

    # Quiz instructions — only added if the student wants a quiz
    quiz_block = ""
    if request.include_quiz:
        count = min(request.quiz_question_count, 10)
        quiz_block = f"""
QUIZ — Generate exactly {count} multiple choice questions:
- Every question must be answerable directly from the explanation you just wrote
- Questions and answer options should use the theme language where natural
- Always provide exactly 4 answer options
- correct_answer must exactly match one of the 4 options word-for-word
- Explanation should confirm why it's correct, still in theme language
"""

    return f"""
Create a complete educational lesson with these exact specifications:

TOPIC: {request.topic}
SUBJECT AREA: {request.subject}
ACADEMIC LEVEL: {request.difficulty.value}

{theme_block}

EXPLANATION REQUIREMENTS:
- Length: 300-400 words for KS2/KS3, 400-550 words for GCSE/A-Level, 500-700 for University
- Open with a hook that immediately puts the student inside the theme world
- Break the concept into clear logical parts — each with its own theme analogy
- Never use a theme reference once and abandon it — maintain it to the end
- Close with why this concept matters in the real world (still in theme language if possible)

KEY POINTS:
- Write 4-5 essential takeaways
- Each must state the actual academic fact wrapped in theme language
- Short and punchy — one sentence each

{quiz_block}

RESPOND WITH THIS EXACT JSON STRUCTURE AND NOTHING ELSE:
{{
  "explanation": "Full explanation here...",
  "key_points": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "quiz": [
    {{
      "question": "Question text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "explanation": "Why this is correct..."
    }}
  ]
}}

Set quiz to null if no quiz was requested.
"""


# ─────────────────────────────────────────────────────────────
# SIMPLE CACHE
# Stores responses in memory so identical requests don't hit
# the OpenAI API again. Saves money at scale.
# In production: replace this dict with Redis.
# ─────────────────────────────────────────────────────────────
_cache: dict = {}

def get_cache_key(req: LearnRequest) -> str:
    raw = f"{req.topic}|{req.subject}|{req.difficulty}|{req.theme}|{req.quiz_question_count}"
    return hashlib.md5(raw.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# These are the URLs the frontend calls.
# GET endpoints just return data.
# POST /learn is the main one that generates lessons.
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Health check — visit this to confirm the server is running"""
    return {"status": "Learnify API v2 is running", "docs": "/docs"}


@app.get("/themes")
async def get_themes():
    """Returns the 5 built-in themes and their descriptions"""
    return {
        "themes": list(THEME_PROFILES.keys()),
        "note": "You can also pass any custom theme string — the AI handles it natively"
    }


@app.get("/levels")
async def get_levels():
    """Returns all difficulty level options"""
    return {level.name: level.value for level in DifficultyLevel}


@app.post("/learn", response_model=LearnResponse)
async def generate_lesson(request: LearnRequest):
    """
    THE MAIN ENDPOINT.
    Takes a topic, subject, difficulty and theme.
    Returns a full themed lesson with key points and optional quiz.

    Test it at http://localhost:8000/docs
    """

    # ── Input validation ──
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    if not request.subject.strip():
        raise HTTPException(status_code=400, detail="Subject cannot be empty.")
    if not request.theme.strip():
        raise HTTPException(status_code=400, detail="Theme cannot be empty.")
    if len(request.topic) > 300:
        raise HTTPException(status_code=400, detail="Topic is too long (max 300 characters).")

    # ── Check cache first ──
    cache_key = get_cache_key(request)
    if cache_key in _cache:
        entry = _cache[cache_key]
        if time.time() - entry["ts"] < 86400:   # 24 hour expiry
            return entry["data"]

    # ── Build prompts ──
    system_prompt = build_system_prompt()
    user_prompt   = build_user_prompt(request)

    # ── Call OpenAI ──
    try:
        response = client.chat.completions.create(
            model="gpt-4o",                         # Best quality model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.72,                       # Slightly creative but grounded
            max_tokens=3500,
            response_format={"type": "json_object"} # Forces valid JSON back
        )

        raw      = response.choices[0].message.content
        parsed   = json.loads(raw)

        # Build quiz objects if the AI returned them
        quiz = None
        if request.include_quiz and parsed.get("quiz"):
            quiz = [
                QuizQuestion(
                    question=q["question"],
                    options=q["options"],
                    correct_answer=q["correct_answer"],
                    explanation=q["explanation"]
                )
                for q in parsed["quiz"]
            ]

        # Estimate reading time (~200 words per minute)
        words     = len(parsed["explanation"].split())
        read_time = max(1, round(words / 200))

        result = LearnResponse(
            topic=request.topic,
            subject=request.subject,
            difficulty=request.difficulty.value,
            theme=request.theme,
            explanation=parsed["explanation"],
            key_points=parsed.get("key_points", []),
            quiz=quiz,
            estimated_read_time_minutes=read_time
        )

        # Store in cache
        _cache[cache_key] = {"data": result, "ts": time.time()}

        return result

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="The AI returned a malformed response. Please try again."
        )
    except openai.RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="OpenAI rate limit reached. Wait a moment and try again."
        )
    except openai.AuthenticationError:
        raise HTTPException(
            status_code=500,
            detail="Invalid OpenAI API key. Check your OPENAI_API_KEY environment variable."
        )
    except openai.APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to OpenAI. Check your internet connection."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# ─────────────────────────────────────────────────────────────
# START THE SERVER
# Running `python main.py` executes this block.
# reload=True means the server restarts automatically every time
# you save a change to main.py — very useful during development.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)