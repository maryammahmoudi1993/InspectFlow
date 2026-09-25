# InspectFlow: a review-and-release workflow for AI defect detection

*A self-directed demonstration project. It shows how a small manufacturing team could review a defect-detection model's output, correct it, compare two model versions fairly, and record a documented go/no-go decision.*

## The problem and who it is for

Teams that start using AI to spot product defects run into the same questions. Which predictions can we trust? Who checked the doubtful ones? Is the new model version really better than the old one, or was it just tested on easier pictures? Who decided to switch, and why?

InspectFlow is aimed at quality leads and operations managers in small factories, and at the analysts who support them. They need something they can read and audit without being machine-learning specialists.

## What I built

- **Django backend.** Data models for batches, images, model versions, predictions, reviews, evaluation sets, and release decisions. Uploads are validated, images are only served to logged-in users, and every label change is recorded in an audit history.
- **A responsive interface.** A dashboard, an upload page with honest progress feedback, a review queue that puts the least certain predictions first, and a comparison page. On phones the review queue turns into readable cards.
- **A model evaluation workflow.** Two model versions are scored on the *same* reviewed images. The page shows overall accuracy, per-class precision and recall, confusion matrices, and the photos where the models disagree. Missing labels or missing predictions are called out instead of being quietly dropped.
- **Testing.** 22 automated tests plus browser walk-throughs at desktop and phone widths.

## The walkthrough

1. **Upload.** Add a batch of photos. They are checked for type and size, a progress bar shows the transfer, and the stand-in classifier scores each image.
2. **Review.** The queue lists images with the least certain predictions first and can be filtered by batch, class, model version, and review status.
3. **Correct.** The reviewer opens an image, sets the correct label, and adds a note. The history keeps who changed it, when, and from what to what.
4. **Compare.** Pick two versions and one evaluation set. Fixing a label there changes the results straight away: in the seeded demo, one correction moved the accuracies from 0.767 and 0.933 to 0.733 and 0.967.
5. **Decide.** The release page gathers the metrics, review coverage, known limitations, and example images. A person records approve, reject, or needs more review, with a rationale. The tool supplies evidence; it does not certify anything.

## What this demo shows, and what it does not

**It shows** that I can take a workflow from data model to polished interface, keep the logic that scores models separate from the demo model so a real one could be plugged in later, and build in auditability and honest labelling.

**It does not show** real-world accuracy. Everything is synthetic:

- The images are generated shapes, not product photos.
- The "classifier" is a deterministic stand-in, not a trained model.
- All accuracy, precision, recall, and confidence figures come from that stand-in and are labelled as synthetic in the app.
- There is no real client, deployment, or measured business impact.
- Login is a single shared demo account, and images are classified during upload rather than in a background queue.

## Technology

Python, Django, SQLite, Pillow, HTML/CSS with Bootstrap, and a little JavaScript for upload feedback. Tested with Django's test framework and Playwright in Edge.

## Suggested Upwork portfolio description

InspectFlow is a self-directed demo of a visual-inspection review workflow that I designed and built with Django. Upload a batch of product photos, see a stand-in classifier's predictions with confidence scores, and review the uncertain ones first. Reviewers correct labels, every change is logged with who, when, and old and new value, and two model versions are compared on the same reviewed images (accuracy, precision, recall, confusion matrices, and the exact photos where the models disagree). The last step is a documented release decision (approve, reject, or needs more review) with written rationale. It is fully responsive and login-protected. All images, labels, and scores are synthetic and are not real production results.

## Screenshots for the portfolio entry

Files are in `docs/screenshots/`. Use them in this order:

| # | File | Caption |
|---|------|---------|
| 1 | `05-model-comparison.png` | Two model versions scored on the same reviewed images, with per-class results and confusion matrices. |
| 2 | `03-review-queue.png` | The review queue puts the least certain predictions first. |
| 3 | `04-review-detail.png` | A reviewer confirms or corrects the label, and the change is added to an audit history. |
| 4 | `06-comparison-after-correction.png` | After one label is corrected, the comparison recalculates straight away. |
| 5 | `07-release-decision.png` | The release page brings together evidence and limitations, and records a human decision with its rationale. |
| 6 | `08-mobile-review-queue.png` | The review queue as readable cards on a phone. |

The other captures (`01-dashboard.png`, `02-batch-uploaded.png`, `09-mobile-review-detail.png`) are available if a gallery allows more images. None of them show credentials.
