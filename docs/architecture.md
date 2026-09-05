# Architecture

Full detail (literature review, problem statement, diagrams) lives in the DA1 report submitted for BCSE306L — copy/export the relevant sections here as the team builds, so the repo is self-contained and doesn't depend on the Word doc.

Pipeline (see DA1 Fig. 1):

Interaction Stream -> Fact/Attribute Extraction -> Candidate Memory Item
  -> Selective Memory Controller (4 joint signals) -> Decision
  -> Tiered Memory Store (STM -> MTM -> LTM) -> Two-Stage Retrieval
  -> Agent Response -> Evaluation Feedback (loops back to Controller)
