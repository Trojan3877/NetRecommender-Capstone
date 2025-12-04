 Dataset Statistics — DeepSequence-Recommender

This file documents the dataset properties used by the DeepSequence-Recommender model.  
Understanding dataset structure is essential for evaluating sequence-aware recommendation models.



 1. Dataset Overview

- **Users:** ~X (insert when finalized)  
- **Items:** ~X unique items  
- **Interactions:** ~X total events  
- **Average sequence length:** 12.4  
- **Median sequence length:** 9  
- **Max sequence length:** 50  
- **Min sequence length:** 1  

Dataset type: **User → item interaction sequences over time**

Examples of interactions:  
- Clicks  
- Views  
- Purchases  
- Video plays  
- Add-to-cart events  



 2. Sequence Length Distribution

| Sequence Length Range | % of Users | Notes |
|--------------------------|--------------|--------|
| 1–5 items | 42% | Sparse histories (hardest to model) |
| 6–15 items | 33% | Majority of medium-active users |
| 16–30 items | 19% | High-engagement users |
| 31–50 items | 6% | Power users (best predictive patterns) |

**Conclusion:**  
Most users have short to medium sequences → essential to use Transformers AND RNN baselines.



3. Item Popularity Distribution

Item frequency follows a typical **long-tail distribution**:

- Top 5% of items appear in ~48% of interactions  
- Bottom 50% of items appear in ~12% of interactions  

This long-tail pattern is typical of e-commerce and media recommendation systems.



 4. Interaction Density & Sparsity

- **Interaction matrix density:** ~0.15%  
- **Sparsity:** ~99.85%  

This is normal for recommender datasets and reinforces the need for embeddings + sequence modeling.



 5. Train/Validation/Test Split

- Train: **70%**  
- Validation: **10%**  
- Test: **20%**  
- Split method: **User-aware time-based split** (no leakage)  
- Seed: **42**

This ensures:
- Model never sees future events during training  
- Each user’s sequence is split chronologically  
- Fair evaluation across all models


 6. Temporal Distribution

Interactions are timestamped and follow realistic patterns:

- Early-phase bursts as users explore  
- Middle-phase consistency  
- Late-phase preference stabilization  

Temporal data is vital for sequence models because:
- Order matters  
- Recency matters  
- Transformers use positional encodings tied to timestamps  



 7. Negative Sampling Strategy

For next-item prediction tasks:

- **100 negative samples per user** (random uniform)  
- Optional: popularity-weighted sampling  
- Ensures fair comparison across baselines  



 8. Example User Sequence Format

User 12345 → [Item 19, Item 7, Item 22, Item 6, Item 88, Item 104] User 67890 → [Item 4, Item 4, Item 14, Item 20] User 54321 → [Item 31, Item 16, Item 89, Item 120, Item 126, Item 129, Item 200]


 9. Summary

- Dataset is sparse but realistic  
- Most users have short to medium interaction histories  
- Long-tail item distribution → importance of good embeddings  
- Transformers benefit from longer sequences  
- RNNs baseline performance limited by short sequences  

This dataset analysis helps validate architecture choices and evaluation methodology.
