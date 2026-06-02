"""Load raw text/PDF documents from data/raw/ into a normalized tuple format(filename, page_number, raw_text)."""

import logging
from pathlib import Path
from typing import Generator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATA_RAW   = Path(__file__).parent / "data" / "raw"

PLACEHOLDER_DOCS = [
    {
        "filename": "nutrition_basics_placeholder.txt",
        "pages": [
            """Macronutrients are the three main categories of nutrients: proteins,
carbohydrates, and fats. Proteins are essential for muscle repair and growth.
The recommended daily intake of protein for sedentary adults is 0.8 grams per
kilogram of body weight. For athletes engaged in resistance training, protein
requirements rise to 1.6–2.2 grams per kilogram of body weight per day.

Carbohydrates are the body's primary energy source. Complex carbohydrates such
as oats, brown rice, and sweet potatoes provide sustained energy and contain
dietary fibre. Simple carbohydrates found in sugary drinks cause rapid blood
glucose spikes followed by crashes, reducing sustained performance.

Dietary fats are critical for hormone production, vitamin absorption, and brain
health. Unsaturated fats from olive oil, avocados, and nuts are preferable to
saturated and trans fats. Adults should aim for fats to constitute 20–35% of
total daily caloric intake.""",

            """Hydration is often overlooked in fitness planning. Water regulates body
temperature, lubricates joints, and transports nutrients. Dehydration of even
1–2% of body weight impairs cognitive and physical performance significantly.
The general recommendation is 2–3 litres of water per day for adults, with
additional intake of 500ml per 30 minutes of moderate-to-vigorous exercise.

Electrolytes—sodium, potassium, magnesium, and calcium—are lost through sweat
and must be replenished during prolonged exercise. Sports drinks are useful
only for sessions exceeding 60 minutes; water is sufficient for shorter bouts.""",
        ]
    },
    {
        "filename": "workout_guide_placeholder.txt",
        "pages": [
            """Progressive overload is the foundational principle of strength training.
It means gradually increasing the stress placed on the musculoskeletal system
over time. This can be achieved by increasing weight, reps, sets, or decreasing
rest periods. Without progressive overload, the body adapts and strength gains
plateau.

Compound movements—squats, deadlifts, bench press, and overhead press—recruit
multiple muscle groups simultaneously and produce greater hormonal responses
than isolation exercises. Beginners should build their programme around compound
lifts before adding isolation work.

Recovery is when growth actually occurs. Muscle fibres are broken down during
training and rebuilt stronger during rest. Beginners require at least 48 hours
of rest between training the same muscle group. Sleep of 7–9 hours per night
is non-negotiable for optimal recovery and muscle protein synthesis.""",

            """Cardiovascular training improves heart efficiency, increases VO2 max, and
burns calories. Zone 2 cardio—sustained effort at 60–70% of maximum heart rate—
builds aerobic base and improves fat oxidation. High-Intensity Interval Training
(HIIT) is effective for improving cardiovascular fitness in less time but is
more taxing on the nervous system and should not exceed 2–3 sessions per week.

A balanced weekly training plan for a general fitness goal might include:
3 resistance training sessions, 2 zone-2 cardio sessions, 1 HIIT session,
and 1 full rest or active recovery day (yoga, walking, stretching).""",
        ]
    },
]


def _load_txt(path: Path) -> Generator[tuple[str, int, str], None, None]:
    """Yield (filename, page=0, full_text) for a .txt file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            yield (path.name, 0, text)
    except Exception as e:
        log.warning(f"Could not read {path.name}: {e}")


def _load_pdf(path: Path) -> Generator[tuple[str, int, str], None, None]:
    """Yield (filename, page_number, page_text) for each page of a .pdf."""
    try:
        import pypdf  # optional dependency; install with: pip install pypdf
        reader = pypdf.PdfReader(str(path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                yield (path.name, i, text)
    except ImportError:
        log.warning("pypdf not installed. Skipping PDF. Run: pip install pypdf")
    except Exception as e:
        log.warning(f"Could not read PDF {path.name}: {e}")


def load_documents() -> list[tuple[str, int, str]]:
    """
    Returns a list of (filename, page_number, raw_text) tuples.

    Scans DATA_RAW for .txt and .pdf files.
    Falls back to PLACEHOLDER_DOCS if the folder is empty.

    """
    docs = []

    raw_files = list(DATA_RAW.glob("*.txt")) + list(DATA_RAW.glob("*.pdf"))

    if not raw_files:
        log.warning(
            f"[PLACEHOLDER] No files found in {DATA_RAW}. "
            "Using built-in placeholder corpus. "
            "Drop your fitness/nutrition docs into data/raw/ to use real content."
        )
        for doc in PLACEHOLDER_DOCS:
            for page_num, page_text in enumerate(doc["pages"]):
                docs.append((doc["filename"], page_num, page_text.strip()))
        return docs

    for path in raw_files:
        before = len(docs)
        if path.suffix == ".txt":
            docs.extend(_load_txt(path))
        elif path.suffix == ".pdf":
            docs.extend(_load_pdf(path))
        added = len(docs) - before
        log.info(f"Loaded {path.name}: {added} page(s)")

    log.info(f"Total pages loaded: {len(docs)}")
    return docs


if __name__ == "__main__":
    pages = load_documents()
    for fname, pg, text in pages:
        print(f"\n{'='*60}")
        print(f"Source: {fname}  |  Page: {pg}")
        print(text[:300] + "..." if len(text) > 300 else text)
