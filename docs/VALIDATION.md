# Release validation

The release was checked against the frozen supplementary experiment before publication.

- The four attributed backbone files differ only in package-relative imports.
- Three CPU behavior tests cover all-one baseline equivalence, spatial guidance weights and nonzero source residuals with zero direct guidance.
- On the first sorted PIE-Bench sample (`000000000000`, selected independently of results), the public single-image pipeline reproduced both the default FocusEdit output and the unmodulated baseline pixel for pixel.
- The seven-method generation runner reproduced all seven edited outputs, both source references and both mask images pixel for pixel on that sample.
- The public evaluator reproduced all 84 per-sample metric values across those seven methods and two references; the maximum absolute difference is recorded in `results/release_validation.json`.
- All 84 full-dataset means and valid counts were independently reconstructed from the 9,800 archived CSV records. The full paired bootstrap summary also matched the archived summary exactly.
- Python wheel construction and command-line help entry points passed.

The GPU check is a one-sample packaging regression check, not a rerun of the full 700-edit experiment. The 700-edit numerical checks aggregate the completed archived experiment.

Before publishing, tracked content and Git history are scanned for credentials, private infrastructure addresses and internal filesystem paths. Public GitHub and Hugging Face references are retained for attribution and model/data access. The scan reports categories and file locations without printing any matching secret values.
