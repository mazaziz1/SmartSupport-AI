# SWE015 final submission checklist

## Before training
- [ ] Use Python 3.10 or 3.11 and install `requirements.txt`.
- [ ] Read the MASSIVE dataset card and cite it.
- [ ] Decide whether to keep MASSIVE as a general-intent proxy or add 50–100 domain-specific support examples as permitted by the assignment.

## After training
- [ ] Run `python -m src.train` successfully.
- [ ] Confirm all four models exist under `models/`.
- [ ] Confirm `models/results.csv` has macro-F1, accuracy, model size, mean latency over 100 runs, p50 latency and hardware.
- [ ] Confirm all four results use the same held-out test set.
- [ ] Review a classification report and record at least one real failure example.
- [ ] Test the Streamlit app with at least two new inputs, including Turkish and a difficult case.
- [ ] Verify that the final route is produced by the quantized INT8 model.

## Before uploading
- [ ] Replace name and student ID on slide 1.
- [ ] Replace result placeholders with exact values from your run; do not invent metrics.
- [ ] Add the actual repository URL and demo URL to slide 1.
- [ ] Keep the presentation between 8 and 12 slides (this draft has 10).
- [ ] Record a narrated demo no longer than 2:00 showing live inputs, architecture and measured results.
- [ ] Upload model weights separately (Hugging Face Hub or Google Drive); do not commit large weights to Git.
- [ ] Ensure README has a working setup command and one run command.
- [ ] Test repository and video links in a private/incognito browser.
- [ ] Ensure the AI usage slide is accurate and includes every AI tool actually used.
