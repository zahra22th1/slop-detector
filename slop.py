import os, requests
from dotenv import load_dotenv
from card import make_card

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

def count_corporate_buzzwords(text):
    """How many LinkedIn buzzwords the post uses (humbled, synergy, ...)."""
    BUZZWORDS = ["humbled", "thrilled to announce", "synergy", "leverage",
                 "thought leader", "grateful", "blessed", "move the needle"]
    return sum(text.lower().count(b) for b in BUZZWORDS)

def engagement_bait(text):
    """Comment-bait closers, plus a point if the whole post ends on a question."""
    CLOSERS = ["agree?", "thoughts?", "comment below", "repost if"]
    hits = sum(text.lower().count(c) for c in CLOSERS)
    return hits + 1 if text.strip().endswith("?") else hits

def excess_dashes(text):
    """Dashes beyond a 3-dash grace (em-dashes, en-dashes, spaced hyphens)."""
    DASHES = ["—", "–", " - "]
    total = sum(text.count(d) for d in DASHES)
    return max(0, total - 3)

def anaphora_hits(text):
    """Lines that repeat another line's opening two words (e.g. 'Culture is built...')."""
    from collections import Counter
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    starts = Counter(" ".join(l.lower().split()[:2]) for l in lines if len(l.split()) >= 2)
    return sum(c for c in starts.values() if c >= 2)

def broetry_ratio(text):
    """Fraction of lines that are tiny one-liners (only trusted once there are 6+ lines)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    short = sum(1 for l in lines if len(l.split()) <= 6)
    return short / len(lines) if len(lines) >= 6 else 0

def emoji_bullets(text):
    """How many lines start with an emoji (decorative bullets)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return sum(1 for l in lines if not l[0].isascii())

def whole_uppercase(text):
    """Fraction of words that are all-caps (shouting)."""
    words = [w for w in text.split() if w.isalpha()]
    upper = sum(1 for w in words if w.isupper())
    return upper / len(words) if words else 0


def score_signals(signals):
    """Turn the precomputed signal map into a weighted score and an offense count."""
    score = min(20, signals["broetry"] * 28)
    score += min(14, signals["buzzwords"] * 4)
    score += min(12, signals["closers"] * 6)
    score += min(12, signals["emoji_bullets"] * 2)
    score += min(8, signals["dashes"] * 3)
    score += min(12, signals["anaphora"] * 3)
    score += min(8, signals["all_uppercase"] * 8)

    offenses = sum([
        signals["broetry"] >= 0.4,
        signals["buzzwords"] >= 1,
        signals["closers"] >= 1,
        signals["emoji_bullets"] >= 2,
        signals["dashes"] > 0,
        signals["anaphora"] >= 2,
        signals["all_uppercase"] >= 0.5
    ])

    return min(80, score), offenses

def zero_shot(text, labels, token):
    """Ask the zero-shot classifier for the probability that text matches labels[1]."""
    payload = {"inputs": text, "parameters": {"candidate_labels": labels}}
    r = requests.post(HF_URL, headers={"Authorization": f"Bearer {token}"},
                      json=payload, timeout=30)
    r.raise_for_status()
    scores = {item["label"]: item["score"] for item in r.json()}
    return scores.get(labels[1], 0.0)

def performative_score(text, token):
    """How much the post reads as a self-promotional brag (vs a modest, low-key update)."""
    labels = ["a modest low-key update",
              "a self-promotional brag"]
    return zero_shot(text, labels, token)

def promotional_score(text, token):
    """How much the post reads as a promotional announcement (vs a genuine personal anecdote)."""
    labels = ["a personal anecdote",
              "a promotional announcement"]
    return zero_shot(text, labels, token)

def score_ai_signals(ai_signals):
    """Average the AI signal probabilities into one 0-1 slop 'vibe'."""
    return sum(ai_signals.values()) / len(ai_signals)

def console_safe(text):
    """Replace unsupported Unicode characters so output works in legacy consoles."""
    return text.encode("ascii", "replace").decode("ascii")

def main():
    # paste your own post between the triple quotes!
    text = """
            As programmers, we're trained to solve problems.

Ironically, sometimes the hardest problem isn't in our code—it's in our own minds.

A bug that feels impossible often turns out to be a missing bracket.

An assignment that seems overwhelming becomes manageable once we break it into smaller tasks.

A challenge that keeps us awake at night is often much smaller than we imagined.

Programming has taught me an unexpected lesson:

Don't panic before you understand the problem.

Analyze it.
Break it down.
Solve one piece at a time.

Because whether it's debugging code or dealing with life, the biggest obstacle is often the story we tell ourselves before we even begin.

Sometimes, the problem isn't as big as it looks. 💻
    """

    signals = {
        "broetry": broetry_ratio(text),
        "buzzwords": count_corporate_buzzwords(text),
        "closers": engagement_bait(text),
        "emoji_bullets": emoji_bullets(text),
        "dashes": excess_dashes(text),
        "anaphora": anaphora_hits(text),
        "all_uppercase": whole_uppercase(text),
    }
    ai_signals = {
        "performative": performative_score(text, HF_TOKEN),
        "promotional": promotional_score(text, HF_TOKEN),
    }

    rules, offenses = score_signals(signals)
    vibe = score_ai_signals(ai_signals)

    if offenses == 0:
        # no tells flagged: lean on the AI alone, kept low
        score = round(vibe * 25)
    else:
        # scale the whole blend up to use the full range
        score = round(min(100, (rules + vibe * 40) * 1.4))

    if score >= 70:   label = "Certified Artisanal Slop [slop]"
    elif score >= 50: label = "Peak LinkedIn Cringe [cringe]"
    elif score >= 30: label = "Mildly Insufferable [meh]"
    elif score >= 15: label = "Suspiciously Normal [hmm]"
    else:             label = "An Actual Human Wrote This [human]"

    print(f"\n  Slop Score: {score}/100  -  {console_safe(label)}\n")
    make_card(score, signals, ai_signals)

if __name__ == "__main__":
    main()