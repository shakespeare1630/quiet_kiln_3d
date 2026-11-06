# quiet_kiln_3d

Image-to-Three.js generator for the 404-GEN (Bittensor subnet 17) closed-model competition; plan: lite-deep (fifteen rewrite rounds).

Entry point: `main.py` exports `async generate(image, seed, witness) -> str | None`, implemented by
`crown/funnel.py`: concurrent drafts, judge-style selection on renders, rounds of rewrites against the
module's own renders, and a final fit. All model calls and renders go through the supplied witness.

Live run (after creating `witness.json` next to `main.py`, see `witness.example.json`):

```sh
python main.py --prompts prompts.txt --seed PUBLISHED_SEED --wallet-name NAME --wallet-hotkey HOTKEY
```
