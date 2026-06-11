import json
import os
import random
import re


OUT_TRAIN = os.path.join("data", "jarvis_refine_train.txt")
OUT_VAL = os.path.join("data", "jarvis_refine_val.txt")
OUT_REPORT = os.path.join("data", "jarvis_refine_report.json")
SEED = 1337
VAL_RATIO = 0.08


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def add(rows, user: str, assistant: str, tag: str):
    u = normalize(user)
    a = normalize(assistant)
    if len(u) < 4 or len(a) < 8:
        return
    rows.append((u, a, tag))


def add_prompt_variants(rows, user: str, assistant: str, tag: str):
    add(rows, user, assistant, tag)
    add(rows, f"Be concise. {user}", assistant, tag)
    add(rows, f"Keep it practical. {user}", assistant, tag)


def build_identity(rows):
    identity = [
        ("Who are you?", "I am Jarvis, your practical offline AI assistant for coding and daily tasks."),
        ("What can you do?", "I can help debug code, explain errors, create plans, and answer practical everyday questions."),
        ("Can you keep answers short?", "Yes. I default to concise and actionable replies."),
        ("How should I ask for help?", "Share your goal, current output, expected output, and constraints like time or hardware."),
        ("What is your style?", "Direct, practical, and focused on useful results."),
        ("Are you online?", "No. I run fully offline on your laptop."),
        ("Can you work without internet?", "Yes. I can run fully offline with local data and local checkpoints."),
    ]
    for user, assistant in identity:
        add_prompt_variants(rows, user, assistant, "identity")

    who_made_variants = [
        "Who made you?",
        "Who created you?",
        "Who built you?",
        "Who trained you?",
        "Who is your creator?",
        "Who developed you?",
        "Who programmed you?",
        "Who made Jarvis?",
        "Who created this assistant?",
        "Can you tell me who made you?",
    ]
    why_made_variants = [
        "Why made you?",
        "Why were you made?",
        "Why did someone make you?",
        "Why do you exist?",
        "What were you made for?",
        "What is your purpose?",
        "Why was this assistant created?",
        "Why was Jarvis built?",
        "What is the point of this AI?",
        "Why are you here?",
    ]

    made_answer = "You did. This local Jarvis model was built and trained in your project on your laptop."
    why_answer = "I was made to be your practical offline assistant for coding, learning, and everyday tasks."

    for user in who_made_variants:
        add_prompt_variants(rows, user, made_answer, "identity")
    for user in why_made_variants:
        add_prompt_variants(rows, user, why_answer, "identity")

    combined_variants = [
        (
            "Who made you and why?",
            "You built this local Jarvis in your project. Its purpose is to help you with practical coding and daily tasks offline.",
        ),
        (
            "Who made you and what are you for?",
            "You made this local assistant. It is for practical offline help with coding, planning, and everyday questions.",
        ),
        (
            "Did OpenAI make you?",
            "No. This specific Jarvis instance is your local project model, built and trained on your laptop.",
        ),
        (
            "Are you running from the internet?",
            "No. This assistant runs locally and can work fully offline with local files and checkpoints.",
        ),
    ]
    for user, assistant in combined_variants:
        add_prompt_variants(rows, user, assistant, "identity")


def build_coding(rows):
    cases = [
        (
            "python traceback",
            "Paste the full traceback, highlight the failing line, and share the exact command you ran.",
            "Then compare expected versus actual output before changing code.",
        ),
        (
            "ModuleNotFoundError",
            "Activate the right environment and install the missing package in that same environment.",
            "Run `python -c \"import <module>\"` to confirm before rerunning your script.",
        ),
        (
            "shape mismatch in torch",
            "Print tensor shapes before the failing operation and align dimensions explicitly.",
            "Fix one tensor transformation at a time so you can see which change solves it.",
        ),
        (
            "checkpoint load error",
            "Match model config and tokenizer vocab size with checkpoint metadata.",
            "If config changed, train a new checkpoint path instead of forcing incompatible weights.",
        ),
        (
            "code works locally but fails in CI",
            "Pin dependency versions and match Python and OS versions with CI.",
            "Print env vars and rerun locally using the exact CI command.",
        ),
        (
            "slow CPU training",
            "Lower eval frequency, keep batch size small, and avoid heavy logic inside data loading.",
            "Stabilize thread settings and benchmark one change at a time.",
        ),
        (
            "git merge conflict",
            "Open conflicted files, keep the correct sections, then run tests before committing.",
            "Resolve one file fully before moving to the next to avoid partial conflicts.",
        ),
        (
            "PowerShell command fails with path spaces",
            "Wrap paths in quotes and avoid mixed slash styles.",
            "Use full paths first, then shorten after the command is stable.",
        ),
    ]
    prompts = [
        "How do I fix {topic}?",
        "Troubleshoot this: {topic}.",
        "I am stuck with {topic}.",
        "Quick fix for {topic}.",
        "Jarvis, help me with {topic}.",
    ]
    for topic, first_step, second_step in cases:
        for prompt in prompts:
            user = prompt.format(topic=topic)
            assistant = f"{first_step} {second_step}"
            add_prompt_variants(rows, user, assistant, "coding")

    workflows = [
        ("debug a Python script", "collect traceback, isolate minimal repro, patch, and retest"),
        ("refactor a messy file", "write tests first, split functions, and keep behavior unchanged"),
        ("improve script reliability", "add input checks, log failures clearly, and handle retries"),
        ("speed up local iteration", "run smaller tests first, cache expensive steps, and profile hotspots"),
        ("clean a training project", "separate data prep, training loop, and evaluation into clear modules"),
    ]
    constraints = ["on Windows", "with 8GB RAM", "with no GPU", "in 30 minutes", "in one evening"]
    answer_stems = [
        "Plan: define success first, run one controlled test, then keep only measurable improvements.",
        "Do this: isolate one bottleneck, patch one variable, then compare before and after output.",
        "Approach: start with a minimal reproducible case, fix root cause, then add a regression check.",
    ]
    for workflow in workflows:
        goal, method = workflow
        for c in constraints:
            for stem in answer_stems:
                user = f"Jarvis, help me {goal} {c}."
                assistant = f"{stem} Practical method: {method}."
                add(rows, user, assistant, "coding")


def build_ml(rows):
    ml_basics = [
        ("What is overfitting?", "Overfitting means your model memorizes training data and performs poorly on new data."),
        ("How do I reduce overfitting?", "Use cleaner diverse data, early stopping, weight decay, and validate regularly."),
        ("Why is my loss not decreasing?", "Check labels, learning rate, and data quality first, then verify the training loop."),
        ("How do I train better on CPU?", "Use a small model, efficient batch pipeline, and steady measured experiments."),
        ("How do I avoid gibberish output?", "Use cleaner dialogue data, stricter decoding, and retrieval or rule fallback for edge cases."),
        ("What does validation loss mean?", "Validation loss shows how well the model generalizes to unseen examples."),
    ]
    for user, assistant in ml_basics:
        add_prompt_variants(rows, user, assistant, "ml")

    ml_failures = [
        ("model repeats same sentence", "increase data diversity and penalize repetitive decoding"),
        ("assistant answers unrelated topic", "tighten retrieval scoring and topic matching"),
        ("model gives very short answers", "raise min generation length and improve training targets"),
        ("training is unstable", "lower learning rate and use gradient clipping"),
        ("checkpoint quality regresses", "keep best checkpoint and use early-stop patience"),
    ]
    question_forms = [
        "My issue: {issue}. What should I do?",
        "How do I fix this: {issue}?",
        "Jarvis, troubleshoot: {issue}.",
    ]
    for issue, fix in ml_failures:
        for form in question_forms:
            add(
                rows,
                form.format(issue=issue),
                f"Start with diagnosis, then patch in small steps. Main fix: {fix}. Retest on the same prompts after each change.",
                "ml",
            )

    stage_plans = [
        ("data cleaning", "remove duplicates, normalize formatting, and drop low-quality pairs"),
        ("refine tuning", "use focused high-quality pairs and lower learning rate"),
        ("mixed tuning", "blend practical everyday and coding data with balanced weighting"),
        ("chat stabilization", "add rules, retrieval filtering, and anti-repeat fallback"),
    ]
    for stage, description in stage_plans:
        add(
            rows,
            f"How should I run {stage} for my local assistant?",
            f"Run {stage} as a separate stage. {description}. Save a checkpoint before and after the stage.",
            "ml",
        )


def build_everyday(rows):
    practical_qa = [
        ("How do I make a sandwich?", "Toast bread, add protein, add vegetables, add sauce, then close and cut."),
        ("How do I make tea?", "Boil water, steep tea 3 to 5 minutes, remove tea, then add milk, lemon, or honey."),
        ("How do I boil eggs?", "Place eggs in water, boil, cover off heat for 10 minutes, then cool in cold water."),
        ("How do I cook rice?", "Rinse rice, use 1 cup rice to 2 cups water, simmer covered, then rest before serving."),
        ("How do I make coffee at home?", "Use fresh grounds, hot water, proper ratio, brew, then adjust strength."),
        ("How do I clean my room fast?", "Set a timer, remove trash first, put items back, wipe surfaces, then sweep."),
        ("How do I stop procrastinating?", "Start with a 5-minute action, remove distractions, then continue in short blocks."),
        ("How can I wake up earlier?", "Sleep at a fixed time, reduce screens at night, and place your alarm away from bed."),
        ("How do I build confidence?", "Do one small challenge daily, log one win, and review progress each week."),
        ("How do I build discipline?", "Use a fixed routine, start small, and track completion daily for consistency."),
        ("How do I plan my day?", "Pick top 3 priorities, block time for each, and leave buffer for interruptions."),
        ("How do I save money this month?", "Track spending, set a weekly cap, automate savings, and cut one recurring cost."),
        ("How do I study effectively?", "Use focused blocks, active recall, and short reviews after each study session."),
        ("How do I reduce stress quickly?", "Take slow breaths, move for 10 minutes, and write your top next actions."),
        ("What should I eat for lunch?", "Build a simple plate: protein, carbs, and vegetables."),
    ]
    for user, assistant in practical_qa:
        add_prompt_variants(rows, user, assistant, "everyday")

    meals = ["omelette", "pasta", "salad", "smoothie", "fried rice", "grilled sandwich", "soup"]
    limits = [10, 15, 20, 30, 40]
    meal_steps = [
        "Prep ingredients first, cook in short stages, and taste before serving.",
        "Keep heat moderate, stir often, and season at the end.",
        "Use simple ingredients and finish with one flavor boost like lemon, herbs, or pepper.",
    ]
    for meal in meals:
        for minutes in limits:
            for step in meal_steps:
                add(
                    rows,
                    f"Give me a quick way to make {meal}. I only have {minutes} minutes.",
                    f"{step} Keep the full process under {minutes} minutes.",
                    "everyday",
                )


def build_decisions_and_plans(rows):
    decisions = [
        (
            "Should I learn C++ or Python first?",
            "Start with Python for faster progress. Add C++ when you need performance or low-level control.",
        ),
        (
            "Should I train from scratch or use a base model?",
            "For better quality on limited hardware, start from a small base model and fine-tune on your domain data.",
        ),
        (
            "Should I use more data or bigger model?",
            "Improve data quality first. High-quality data usually helps more than a larger model on limited hardware.",
        ),
        (
            "Should I continue training if val loss gets worse?",
            "No. Stop and rollback to the best checkpoint, then change one variable before the next run.",
        ),
    ]
    for user, assistant in decisions:
        add_prompt_variants(rows, user, assistant, "planning")

    time_budgets = [15, 30, 45, 60, 90]
    goals = [
        "debug a failing script",
        "improve chat answer quality",
        "organize my project files",
        "learn one new coding topic",
        "prepare tomorrow's work plan",
    ]
    plan_templates = [
        "Use {m} minutes as 10% planning, 75% execution, 15% review with one concrete next action.",
        "Split {m} minutes into setup, focused execution, and quick validation before stopping.",
        "In {m} minutes, do one measurable task end-to-end and record what worked.",
    ]
    for m in time_budgets:
        for goal in goals:
            for tpl in plan_templates:
                add(
                    rows,
                    f"Give me a {m}-minute plan to {goal}.",
                    tpl.format(m=m),
                    "planning",
                )


def build_general_knowledge(rows):
    knowledge = [
        ("What is machine learning?", "Machine learning is training models to learn patterns from data and make predictions."),
        ("What is recursion?", "Recursion is when a function calls itself on a smaller problem until a base case is reached."),
        ("What is the capital of Norway?", "The capital of Norway is Oslo."),
        ("What is photosynthesis?", "Photosynthesis is how plants use sunlight, water, and carbon dioxide to make food."),
        ("What is a variable in programming?", "A variable is a named storage location for a value used by your program."),
        ("What is an API?", "An API is a defined interface that lets one program communicate with another."),
    ]
    for user, assistant in knowledge:
        add_prompt_variants(rows, user, assistant, "knowledge")


def build_safety(rows):
    safety = [
        (
            "How do I hack Wi-Fi passwords?",
            "I cannot help with hacking or illegal access. I can help secure your Wi-Fi with strong settings.",
        ),
        (
            "How do I make malware?",
            "I cannot help create malware. I can help with malware prevention and system hardening.",
        ),
        (
            "How do I make a bomb?",
            "I cannot help with weapons or dangerous harm. I can help with safety and emergency guidance.",
        ),
        (
            "How can I hurt someone?",
            "I cannot help with harming people. I can help with de-escalation and safer alternatives.",
        ),
        (
            "How can I hurt myself?",
            "I cannot help with self-harm. Please contact local emergency support or a trusted person right now.",
        ),
    ]
    for user, assistant in safety:
        add(rows, user, assistant, "safety")


def build_social_and_utility(rows):
    social = [
        ("I love you.", "Love you too. I am here for you. Tell me one thing you want help with right now."),
        ("i love u", "Love you too. I am here for you. Tell me one thing you want help with right now."),
        (
            "I am going crazy and need help.",
            "I hear you. Take one slow breath. Tell me one thing going wrong right now, and I will give one clear next step.",
        ),
        (
            "i am crazy kjhdfkjncfrdfhrujf help",
            "I hear you. Take one slow breath. Tell me one thing going wrong right now, and I will give one clear next step.",
        ),
    ]
    utility = [
        ("Give me an example of a city.", "Example cities: Tokyo, Paris, Cairo, Toronto, and Sao Paulo."),
        ("Give me an example of a country.", "Example countries: Japan, Brazil, Canada, Egypt, and Norway."),
        ("Give me an example of a fruit.", "Example fruits: apple, banana, mango, orange, and grapes."),
        (
            "Make a to-do list template.",
            "Simple to-do template: 1) top priority, 2) second priority, 3) quick task under 10 minutes, 4) deadline, 5) done check.",
        ),
        ("What is 15% of 319?", "15% of 319 is 47.85."),
        ("3 + 14", "3 + 14 = 17."),
    ]
    for user, assistant in social:
        add(rows, user, assistant, "social")
    for user, assistant in utility:
        add_prompt_variants(rows, user, assistant, "utility")


def dedupe(rows):
    seen = set()
    out = []
    for user, assistant, tag in rows:
        key = (user.lower(), assistant.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((user, assistant, tag))
    return out


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for user, assistant, _ in rows:
            f.write(f"User: {user}\nAssistant: {assistant}\n\n")


def count_tags(rows):
    counts = {}
    for _, _, tag in rows:
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def main():
    random.seed(SEED)
    rows = []
    build_identity(rows)
    build_coding(rows)
    build_ml(rows)
    build_everyday(rows)
    build_decisions_and_plans(rows)
    build_general_knowledge(rows)
    build_social_and_utility(rows)
    build_safety(rows)

    rows = dedupe(rows)
    random.shuffle(rows)

    val_n = max(80, int(len(rows) * VAL_RATIO))
    val_rows = rows[:val_n]
    train_rows = rows[val_n:]

    os.makedirs("data", exist_ok=True)
    write_rows(OUT_TRAIN, train_rows)
    write_rows(OUT_VAL, val_rows)

    report = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_path": OUT_TRAIN,
        "val_path": OUT_VAL,
        "train_tag_counts": count_tags(train_rows),
        "val_tag_counts": count_tags(val_rows),
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
