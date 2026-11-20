# Two-minute demo video script (maximum 2:00)

**0:00–0:15 — Problem**  
“SmartSupport AI routes customer support messages in Turkish and English. It predicts the intent, shows confidence, and sends uncertain messages to a human agent.”

**0:15–1:15 — Live application**  
1. Open the running Streamlit application. Type: `My dishwasher is not draining water and there is water at the bottom.` Click **Classify ticket**. Show all four model predictions, confidence scores and latency; point out the INT8 model's final route.
2. Enter a Turkish example such as: `Bulaşık makinem su boşaltmıyor.` Show the new live prediction.
3. Enter a difficult/ambiguous example from your actual test run. If the quantized model confidence is below the selected threshold, show the Human Agent escalation. Do not fake an error or confidence score.
4. Enter `Hello` to demonstrate the rule-based greeting response.

**1:15–1:45 — Architecture and results**  
Show the architecture slide and explain preprocessing → four model predictions → confidence threshold/team mapping → interface. Show the results table populated from your own `models/results.csv` and state the hardware used for latency measurement.

**1:45–2:00 — Conclusion**  
“The application runs its final routing decision with the distilled INT8 Transformer. The comparison shows the trade-off between classification quality, model size and latency. The main limitation is [your real limitation].”

## Recording checklist
- Record the actual application running, not screenshots or notebook output.
- Use at least two different new inputs, including one difficult case.
- Narrate clearly and keep the final video at or under 2:00.
- Upload as unlisted YouTube or Google Drive with “anyone with the link”.
- Test both video and repository links in a private/incognito browser.
