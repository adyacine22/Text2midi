"""Prompt library component with pre-built prompts"""

import streamlit as st


# Pre-built prompts organized by category
PROMPT_LIBRARY = {
    "Piano": [
        "A melodic piano piece in C major with a relaxing atmosphere",
        "A classical piano sonata in D minor with dramatic dynamics",
        "A jazz piano improvisation with swing rhythm and complex chords",
        "A romantic piano ballad in F major, slow and emotional",
        "An upbeat ragtime piano piece with syncopated rhythms",
    ],
    "Electronic": [
        "An energetic electronic dance track with synthesizers and drums, tempo 128 BPM",
        "An ambient electronic soundscape with atmospheric pads and subtle melodies",
        "A synthwave composition with retro 80s sound and driving bassline",
        "A minimal techno track with repetitive patterns and deep bass",
        "An uplifting trance melody with arpeggiated synths and uplifting chords",
    ],
    "Classical": [
        "A classical string quartet in G major with elegant melodies",
        "An orchestral symphony movement with full brass and strings, epic and dramatic",
        "A baroque harpsichord piece with intricate counterpoint",
        "A romantic era waltz for orchestra in 3/4 time",
        "A classical chamber music piece for woodwind quintet",
    ],
    "Jazz": [
        "A bebop jazz composition with saxophone, piano, bass and drums",
        "A smooth jazz ballad with muted trumpet and soft piano",
        "A latin jazz piece with energetic percussion and brass section",
        "A cool jazz composition with walking bass and soft brushes",
        "A fusion jazz track blending jazz harmonies with rock rhythms",
    ],
    "World": [
        "A Celtic folk melody with fiddle and acoustic guitar",
        "A Spanish flamenco piece with guitar and hand percussion",
        "An Indian raga with sitar and tabla, meditative and flowing",
        "A Japanese traditional piece with koto and shakuhachi flute",
        "A Middle Eastern composition with oud and percussion",
    ],
    "Rock/Pop": [
        "An energetic rock song with electric guitar, bass and drums",
        "A pop ballad with piano and strings, emotional and uplifting",
        "A funk groove with electric bass, rhythm guitar and brass",
        "A blues rock piece with guitar solos and organ",
        "An indie rock song with jangly guitars and steady drums",
    ],
    "Mood-based": [
        "A dark and mysterious composition with minor chords and low tones",
        "A bright and cheerful melody that evokes happiness and joy",
        "A sad and melancholic piece with slow tempo and emotional progression",
        "An epic and cinematic composition suitable for film soundtracks",
        "A peaceful and meditative ambient piece for relaxation",
    ],
    "Christmas": [
        "A festive Christmas carol with bells, strings and choir",
        "A jazzy Christmas song with swing rhythm and playful melody",
        "A classical Christmas piece for orchestra with grand arrangement",
        "A cozy holiday tune with acoustic guitar and soft percussion",
        "An upbeat Christmas pop song with cheerful instrumentation",
    ],
}


def get_random_prompt():
    """Get a random prompt from the library"""
    import random

    category = random.choice(list(PROMPT_LIBRARY.keys()))
    prompt = random.choice(PROMPT_LIBRARY[category])
    return prompt


def render_prompt_library():
    """Render prompt library UI"""
    st.markdown("### 💡 Example Prompts")

    # Category tabs
    tabs = st.tabs(list(PROMPT_LIBRARY.keys()))

    for tab, category in zip(tabs, PROMPT_LIBRARY.keys()):
        with tab:
            prompts = PROMPT_LIBRARY[category]
            for prompt in prompts:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"• {prompt}")
                with col2:
                    if st.button("Use", key=f"use_{category}_{prompt[:20]}"):
                        st.session_state.selected_prompt = prompt
                        st.rerun()


def get_example_prompts_for_category(category):
    """Get example prompts for a specific category"""
    return PROMPT_LIBRARY.get(category, [])


def render_quick_examples():
    """Render quick example buttons"""
    st.markdown("**Quick Examples:**")

    quick_examples = [
        ("🎹 Piano", "A melodic piano piece in C major with a relaxing atmosphere"),
        ("🎸 Rock", "An energetic rock song with electric guitar, bass and drums"),
        ("🎻 Classical", "A classical string quartet in G major with elegant melodies"),
        (
            "🎧 Electronic",
            "An ambient electronic soundscape with atmospheric pads and subtle melodies",
        ),
    ]

    cols = st.columns(len(quick_examples))

    for col, (label, prompt) in zip(cols, quick_examples):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.selected_prompt = prompt
                st.rerun()
