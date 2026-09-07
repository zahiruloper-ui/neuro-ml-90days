
## Key findings
- A naive random split inflates accuracy due to leakage from overlapping 
  windows (Random Forest: 83% naive vs. 58.7% time-block-split test accuracy).
- More flexible models (Random Forest) are more vulnerable to this leakage 
  than simpler models (Logistic Regression), because they can "memorize" 
  near-duplicate samples more effectively.
- Errors in the time-block test set were one-directional (false positives 
  only) and occurred in short bursts rather than a steady decline, suggesting 
  intermittent non-stationarity (e.g., brief artifacts) rather than a broken 
  model or fixed bias.
- Mean/std (time-domain) features are a reasonable baseline but likely miss 
  frequency-domain information (e.g., alpha-band power) known to be important 
  for eye-state classification.

## Limitations
- Single-subject dataset: a true subject-wise split (gold standard for 
  multi-subject neuro data) was not possible. Time-block split was used as 
  a partial substitute, testing generalization across time, not across subjects.
- Bandpower features were not implemented in this pipeline; time-domain 
  features (mean, std) only.
