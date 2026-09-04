import nltk
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure meteor data is available if needed
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    meteor_score = None

REFERENCE_EXAMPLES = [
    {
        "generated": "Deploy frontend application to AWS S3 by Friday afternoon.",
        "reference": "Deploy client web interface to S3 cloud storage before end of week."
    },
    {
        "generated": "Database migration script failed due to connection timeout error.",
        "reference": "Migration script crashed because of DB server connection timeout."
    },
    {
        "generated": "Schedule weekly sync meeting with product management team every Tuesday.",
        "reference": "Hold weekly status meeting with product owners on Tuesdays."
    },
    {
        "generated": "Fix authentication token expiration bug in API service layer.",
        "reference": "Resolve API auth token expiry issue in backend system."
    },
    {
        "generated": "Update UI color theme and brand typography according to design system.",
        "reference": "Apply new design guidelines, color scheme and fonts to frontend."
    },
    {
        "generated": "Refactor database query logic to reduce load time by 30 percent.",
        "reference": "Optimize SQL query execution to boost response speed by 30%."
    },
    {
        "generated": "Prepare quarterly project roadmap presentation slides for leadership review.",
        "reference": "Create slide deck covering Q3 roadmap for executive management."
    },
    {
        "generated": "Add automated unit tests covering user profile module endpoints.",
        "reference": "Write comprehensive unit test suite for user profile API."
    },
    {
        "generated": "Configure CI CD deployment pipeline on GitHub Actions platform.",
        "reference": "Set up continuous integration and delivery on GitHub Actions."
    },
    {
        "generated": "Conduct security audit on third party API dependencies and libraries.",
        "reference": "Review security vulnerabilities in external packages and modules."
    },
    # Expanded Reference Set (+15 items extracted from project datasets)
    {
        "generated": "The homepage takes almost five seconds to load.",
        "reference": "Homepage loading speed is unacceptably slow at 5 seconds."
    },
    {
        "generated": "We could compress the images or use WebP format.",
        "reference": "Recommend image compression and WebP conversion to improve speed."
    },
    {
        "generated": "I will investigate WebP support and check image sizes.",
        "reference": "Action assigned to inspect image sizes and test WebP compatibility."
    },
    {
        "generated": "The team agreed to compress all large homepage images.",
        "reference": "Decision reached to compress large images across the site."
    },
    {
        "generated": "Email open rates are currently twenty percent for marketing.",
        "reference": "Current marketing metrics indicate a 20% email open rate."
    },
    {
        "generated": "The latest release introduced usability problems on mobile.",
        "reference": "Mobile user experience degraded following the recent system release."
    },
    {
        "generated": "We should improve the mobile experience and layout.",
        "reference": "Propose mobile UX enhancements and responsive layout refactoring."
    },
    {
        "generated": "We decided to change the email marketing campaign strategy.",
        "reference": "Executive agreement to revise and launch a new email strategy."
    },
    {
        "generated": "The API response time is too slow under high load.",
        "reference": "Performance issue: API endpoints latency spikes during traffic peaks."
    },
    {
        "generated": "The venue does not have enough seating for attendees.",
        "reference": "Event planning constraint: venue capacity insufficient for attendee list."
    },
    {
        "generated": "The current campaign is not generating enough sales revenue.",
        "reference": "Marketing issue: campaign conversion rate fails to meet revenue targets."
    },
    {
        "generated": "The authentication service fails when invalid credentials are entered.",
        "reference": "Critical security bug: authentication crash on malformed credential input."
    },
    {
        "generated": "The current event budget is set at two hundred thousand.",
        "reference": "Financial note: total allocated budget for event is $200,000."
    },
    {
        "generated": "We agreed to reduce the JavaScript bundle size across pages.",
        "reference": "Decision made to minimize JS bundle payload to accelerate rendering."
    },
    {
        "generated": "I will test the new optimization settings on staging.",
        "reference": "Task owner will benchmark performance tweaks in staging environment."
    }
]


def evaluate_generated_notes(data_pairs=None):
    if data_pairs is None:
        data_pairs = REFERENCE_EXAMPLES

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    results = []
    for pair in data_pairs:
        gen = pair["generated"]
        ref = pair["reference"]

        # ROUGE
        rouge_scores = scorer.score(ref, gen)

        # BLEU
        ref_tokens = [ref.lower().split()]
        gen_tokens = gen.lower().split()
        bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smooth)

        # METEOR
        meteor = 0.0
        if meteor_score is not None:
            try:
                meteor = meteor_score([ref.split()], gen.split())
            except Exception:
                meteor = 0.0

        results.append({
            "generated": gen,
            "reference": ref,
            "rouge1_f1": rouge_scores["rouge1"].fmeasure,
            "rouge2_f1": rouge_scores["rouge2"].fmeasure,
            "rougeL_f1": rouge_scores["rougeL"].fmeasure,
            "bleu": bleu,
            "meteor": meteor
        })

    df = pd.DataFrame(results)
    summary = {
        "mean_rouge1_f1": df["rouge1_f1"].mean(),
        "mean_rouge2_f1": df["rouge2_f1"].mean(),
        "mean_rougeL_f1": df["rougeL_f1"].mean(),
        "mean_bleu": df["bleu"].mean(),
        "mean_meteor": df["meteor"].mean()
    }
    return df, summary


if __name__ == "__main__":
    df, summary = evaluate_generated_notes()
    print("=== GENERATED STICKY NOTE EVALUATION METRICS ===")
    for k, v in summary.items():
        print(f"{k:18s}: {v:.4f}")
